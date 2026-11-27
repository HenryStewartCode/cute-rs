#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Drive scripts/gpu-validate.sh on a rented GPU machine over SSH, from your own machine.
#
#   scripts/pod-run.sh start  <port> <user@host> [--worktree] [-- validate options]
#   scripts/pod-run.sh status <port> <user@host>
#   scripts/pod-run.sh fetch  <port> <user@host>
#
# start   copies the source (the committed HEAD, or with --worktree the working tree's
#         tracked and untracked files, minus personal ones) to ~/cute-rs-validate on the
#         machine and starts gpu-validate.sh there under nohup, so a dropped connection
#         does not stop it. Options after `--` go to gpu-validate.sh (--quick, --no-full,
#         --no-bench).
# status  prints the run's summary so far, and whether it is still running.
# fetch   copies the results archive back into ./gpu-results/ here.
#
# SSH_KEY picks the key (default ~/.ssh/runpod_ed25519), POD_DIR the directory on the machine
# (default ~/cute-rs-validate), POD_ENV extra environment for the run ("VAR=value ..."), and
# POD_EXCLUDE an extended regular expression of paths that --worktree leaves behind (untracked
# notes, say: POD_EXCLUDE='^notes/|^TODO\.md$').
# The machine needs only an NVIDIA driver, curl and apt (Ubuntu or Debian): gpu-validate.sh installs Rust and mlir-opt.
#
# Typical rental:  start  ->  status every few minutes  ->  fetch  ->  stop the pod.
# Nothing is left running on the machine once gpu-validate.sh exits.

set -euo pipefail

usage() { sed -n '3,24p' "$0"; exit 2; }
[ $# -ge 3 ] || usage
CMD=$1 PORT=$2 HOST=$3
shift 3
KEY=${SSH_KEY:-$HOME/.ssh/runpod_ed25519}
SSH=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "$HOST")
REMOTE=${POD_DIR:-'~/cute-rs-validate'}

cd "$(git rev-parse --show-toplevel)"

case "$CMD" in
start)
    WORKTREE=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --worktree) WORKTREE=1; shift ;;
            --) shift; break ;;
            *) usage ;;
        esac
    done
    if "${SSH[@]}" "pgrep -f '[g]pu-validate.sh' >/dev/null"; then
        echo "a validation is already running there; use status" >&2; exit 1
    fi
    COMMIT=$(git rev-parse --short HEAD)
    echo "copying $([ $WORKTREE -eq 1 ] && echo "the working tree (HEAD $COMMIT plus local changes)" || echo "HEAD $COMMIT") to $HOST:$REMOTE"
    SKIP='^(target/|gpu-results/)|(^|/)\.DS_Store$'
    [ -n "${POD_EXCLUDE:-}" ] && SKIP="$SKIP|$POD_EXCLUDE"
    if [ $WORKTREE -eq 1 ]; then
        # Tracked and untracked files, without build output, results and what POD_EXCLUDE matches.
        git ls-files -z --cached --others --exclude-standard |
            grep -zv -E "$SKIP" |
            while IFS= read -r -d '' f; do [ -e "$f" ] && printf '%s\0' "$f"; done |
            COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata --uid 0 --gid 0 --null -T - -czf - |
            "${SSH[@]}" "rm -rf $REMOTE && mkdir -p $REMOTE && tar --no-same-owner -xzf - -C $REMOTE && echo '$COMMIT+worktree' > $REMOTE/.cute-rs-commit"
    else
        git archive --format=tar.gz HEAD |
            "${SSH[@]}" "rm -rf $REMOTE && mkdir -p $REMOTE && tar --no-same-owner -xzf - -C $REMOTE && echo '$COMMIT' > $REMOTE/.cute-rs-commit"
    fi
    "${SSH[@]}" "cd $REMOTE && (nohup env ${POD_ENV:-} bash scripts/gpu-validate.sh $* > validate.log 2>&1 < /dev/null &) && echo started"
    echo "follow it with: $0 status $PORT $HOST"
    ;;
status)
    "${SSH[@]}" "cd $REMOTE 2>/dev/null || { echo 'nothing started there'; exit 1; }
        if pgrep -f '[g]pu-validate.sh' >/dev/null; then echo '== running'; else echo '== finished'; fi
        s=\$(ls -t gpu-results/*/SUMMARY.txt 2>/dev/null | head -1)
        if [ -n \"\$s\" ]; then cat \"\$s\"; else tail -20 validate.log; fi"
    ;;
fetch)
    mkdir -p gpu-results
    LATEST=$("${SSH[@]}" "ls -t $REMOTE/gpu-results/*.tar.gz 2>/dev/null | head -1")
    [ -n "$LATEST" ] || { echo "no results archive yet (is it still running? try status)" >&2; exit 1; }
    scp -i "$KEY" -P "$PORT" "$HOST:$LATEST" gpu-results/
    scp -i "$KEY" -P "$PORT" "$HOST:$REMOTE/validate.log" "gpu-results/$(basename "$LATEST" .tar.gz).validate.log"
    echo "saved gpu-results/$(basename "$LATEST")"
    ;;
*) usage ;;
esac
