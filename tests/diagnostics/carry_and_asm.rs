//! Carry chains and inline PTX: operand naming, constraints, simulations and validation.

use super::*;
/// The message of a kernel build over a `u32` parameter `x`, which `body`
/// gets with the thread index and must fail to build.
fn carry_misuse(body: impl FnOnce(Tensor<u32>, Int32)) -> String {
    build_panic(move || {
        Kernel::builder("carry_misuse").block_dim([32, 1, 1]).build(|| {
            let x = param_tensor::<u32>("x", Layout::col_major(32));
            let t = thread_idx().x;
            body(x, t);
        });
    })
}

/// A carry-in instruction needs a carry-out before it in the same block of
/// straight-line code: the validator names it, the line, and what came
/// between, if anything.
#[test]
fn carry_chains_stay_in_straight_line_code() {
    let here = |line: u32| format!("tests/diagnostics/carry_and_asm.rs:{line}:");
    // No carry-out at all.
    let mut line = 0;
    let msg = carry_misuse(|x, t| {
        line = line!() + 1;
        x.store(t, addc(x.load(t), 1u32));
    });
    assert!(
        msg.contains("addc reads the carry flag, but nothing before it in this block of straight-line code sets it")
    );
    assert!(msg.contains(&here(line)), "{msg}");
    // A chain broken by an if_.
    let (mut set, mut broken) = (0, 0);
    let msg = carry_misuse(|x, t| {
        let a = x.load(t);
        set = line!() + 1;
        let low = add_cc(a, a);
        broken = line!() + 1;
        if_(t.eq(0), || x.store(t, low));
        line = line!() + 1;
        x.store(t, addc(a, a));
    });
    assert!(msg.contains(&format!("addc reads the carry flag set by the add_cc at {}", here(set))), "{msg}");
    assert!(msg.contains(&format!("but the runtime if at {}", here(broken))), "{msg}");
    assert!(msg.contains(&here(line)) && msg.contains("keep each chain in straight-line code"), "{msg}");
    // The flag does not reach into a branch.
    let msg = carry_misuse(|x, t| {
        let a = x.load(t);
        let low = add_cc(a, a);
        line = line!() + 1;
        if_(t.eq(0), || x.store(t, subc(low, a)));
    });
    assert!(msg.contains("subc reads the carry flag, but nothing before it in this block"), "{msg}");
    assert!(msg.contains(&here(line)), "{msg}");
    // Loops, calls, inline PTX that uses the flag or branches, and a 64-bit
    // division or remainder by a runtime value (which LLVM lowers with a
    // branch) lose it too.
    let division = "the 64-bit integer division or remainder by a runtime value (which LLVM lowers with a branch) at";
    for (what, breaker) in [
        ("the runtime loop at", 0),
        ("the printf (a call) at", 1),
        ("the inline PTX that uses the carry flag at", 2),
        ("the inline PTX that branches, calls or exits at", 3),
        (division, 4),
        (division, 5),
        (division, 6),
    ] {
        let msg = carry_misuse(|x, t| {
            let a = x.load(t);
            let low = mad_cc(a, a, a);
            let wide = a.to::<u64>() + 1u64;
            match breaker {
                0 => for_range(0, 2, 1, |i| x.store(i, low)),
                1 => printf("%u\n", &[low.into()]),
                2 => inline_ptx_effect("{ .reg .b32 r; add.cc.u32 r, $0, 1; }", &[low.into()]),
                3 => inline_ptx_effect("{ .reg .pred p; setp.ne.u32 p, $0, 0; @p bra.uni skip; skip: }", &[low.into()]),
                4 => x.store(t, (wide * 5u64 / wide).to::<u32>()),
                5 => x.store(t, (wide.bitcast::<i64>() % wide.bitcast::<i64>()).to::<u32>()),
                _ => x.store(t, (7u64 / wide).to::<u32>()),
            }
            x.store(t, madc_cc(a, a, low));
        });
        assert!(msg.contains("madc_cc reads the carry flag set by the mad_cc at") && msg.contains(what), "{msg}");
    }
    // A chain cannot switch between adding and subtracting: the GPU keeps a
    // borrow as the inverted carry.
    let (mut set, mut read) = (0, 0);
    let msg = carry_misuse(|x, t| {
        let a = x.load(t);
        set = line!() + 1;
        let low = add_cc(a, 3u32);
        read = line!() + 1;
        x.store(t, subc(low, a));
    });
    assert!(msg.contains(&format!("subc reads the carry flag set by the add_cc at {}", here(set))), "{msg}");
    assert!(msg.contains("but a subtraction reads it as a borrow, which an addition or multiply-add does not leave"));
    assert!(msg.contains("start a new chain (sub_cc)") && msg.contains(&here(read)), "{msg}");
    let msg = carry_misuse(|x, t| {
        let a = x.load(t);
        let low = sub_cc(a, 3u32);
        x.store(t, madc_cc(low, a, a));
    });
    assert!(msg.contains("madc_cc reads the carry flag set by the sub_cc at"), "{msg}");
    assert!(msg.contains("but an addition reads it as a carry, which a subtraction does not leave"), "{msg}");
    // Inline PTX that leaves the flag alone, loads, stores and arithmetic do not.
    Kernel::builder("carry_ok").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<u32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        let low = add_cc(x.load(t), 5u32);
        x.store(t, low * 3u32);
        inline_ptx_effect("{ .reg .b32 r; add.u32 r, $0, 1; }", &[low.into()]);
        x.store((t + 1) % 32, addc(x.load(t), 7u32));
        // Adds and multiply-adds are one kind.
        let low = mad_cc(low, 3u32, 1u32);
        x.store(t, addc_cc(low, 2u32) ^ madc(low, low, low));
        // A 64-bit division by a constant (a multiplication in the PTX) and
        // a 32-bit division by a runtime value (one div.u32) keep the flag.
        let wide = x.load(t).to::<u64>();
        let low = add_cc(x.load(t), 9u32);
        let quotient = (wide / 10u64 + (wide.bitcast::<i64>() % -7i64).bitcast::<u64>()).to::<u32>();
        let narrow = x.load(t) / (x.load(t) | 1u32);
        x.store(t, addc(low, quotient ^ narrow));
    });
}

/// The validator also checks carry instructions built as IR by hand.
#[test]
fn carry_instructions_are_validated() {
    let kernel = Kernel::builder("carry_ir").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<u32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        x.store(t, add_cc(x.load(t), x.load(t)));
    });
    let mutate = |f: &dyn Fn(&mut cute::ir::Op)| {
        let mut ir = kernel.ir().clone();
        for inst in &mut ir.body.insts {
            if matches!(inst.op, cute::ir::Op::CarryInst { .. }) {
                f(&mut inst.op);
            }
        }
        Kernel::from_ir(ir).map(|_| ()).unwrap_err().to_string()
    };
    let e = mutate(&|op| {
        if let cute::ir::Op::CarryInst { func, .. } = op {
            func.carry_out = false;
        }
    });
    assert!(
        e.contains(
            "add_cc: an extended-precision instruction reads or writes the carry flag, and this one does neither"
        ),
        "{e}"
    );
    let e = mutate(&|op| {
        if let cute::ir::Op::CarryInst { args, .. } = op {
            args.pop();
        }
    });
    assert!(e.contains("add_cc takes 2 operands, not 1"), "{e}");
    let e = mutate(&|op| {
        if let cute::ir::Op::CarryInst { func, .. } = op {
            func.carry_in = true;
        }
    });
    assert!(e.contains("addc_cc reads the carry flag, but nothing before it"), "{e}");
}

/// The message of a kernel build that must fail, over a `u32` parameter
/// `x`, and the line of the statement it fails at: the first `::new(` call
/// (`InlinePtx::new`, `InlineAsm::new`, `MlirOp::new`) after the line
/// `body` passes to `mark`, which is where the statement is located however
/// the formatter wraps it.
fn asm_misuse(body: impl FnOnce(Tensor<u32>, Int32, &mut dyn FnMut(u32))) -> (String, u32) {
    let mut line = 0;
    let msg = build_panic(|| {
        Kernel::builder("asm_misuse").block_dim([32, 1, 1]).build(|| {
            let x = param_tensor::<u32>("x", Layout::col_major(32));
            let t = thread_idx().x;
            body(x, t, &mut |l| line = l);
        });
    });
    let source = include_str!("carry_and_asm.rs");
    let at = source.lines().enumerate().skip(line as usize).find(|(_, l)| l.contains("::new(")).expect("a statement");
    (msg, at.0 as u32 + 1)
}

/// Named inline PTX: every name must have an operand and every operand a
/// name, a named template takes no positional operands, and a predicate
/// guards one instruction. The build fails at the template's line.
#[test]
fn inline_ptx_names_its_operands() {
    let cases: Vec<((String, u32), &str)> = vec![
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 =
                    InlinePtx::new("add.u32 {$w0}, {$r0}, {$r3};").write::<u32>().read(x.load(t)).read(1u32).emit();
                x.store(t, y);
            }),
            "inline_ptx: `{$r3}` names no operand: there are 2 read-only operands (.read, .read_ptr), {$r0} to {$r1}",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 =
                    InlinePtx::new("mov.u32 {$w0}, {$r0};").write::<u32>().read(x.load(t)).read(5u32).emit();
                x.store(t, y);
            }),
            "operand {$r1} (one of the read-only operands (.read, .read_ptr)) does not appear in the template",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlinePtx::new("mov.u32 {$w0}, {$rw0};").write::<u32>().emit();
                x.store(t, y);
            }),
            "`{$rw0}` names no operand: there are no read-write operands (.read_write)",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlinePtx::new("mov.u32 $0, {$r0};").write::<u32>().read(x.load(t)).emit();
                x.store(t, y);
            }),
            "`$0` is a positional operand, but this template names its operands",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlinePtx::new("mov.u32 {$out}, {$r0};").write::<u32>().read(x.load(t)).emit();
                x.store(t, y);
            }),
            "`{$out}` is not an operand name: inline PTX names its operands {$wN}",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlinePtx::new("add.u32 {$rw0}, {$rw0}, 1; add.u32 {$rw0}, {$rw0}, 2;")
                    .read_write(x.load(t))
                    .predicate(t.lt(4))
                    .emit();
                x.store(t, y);
            }),
            "a predicate guards one instruction (PTX's @p), but the template",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlinePtx::new("@{$r0} add.u32 {$rw0}, {$rw0}, 1;")
                    .read_write(x.load(t))
                    .read(t.lt(8))
                    .predicate(t.lt(4))
                    .emit();
                x.store(t, y);
            }),
            "has its own guard, and a predicate adds another (PTX's @p takes one)",
        ),
    ];
    for ((msg, line), want) in cases {
        assert!(msg.contains(want), "expected {want:?} in {msg}");
        assert!(
            msg.contains(&format!("tests/diagnostics/carry_and_asm.rs:{line}:")),
            "the error should point at line {line}: {msg}"
        );
    }
}

/// Raw inline assembly: the constraint string must match the outputs and
/// operands given, each code must be one NVPTX takes, each register class
/// must hold its operand's type, an immediate must be an integer constant,
/// an address must be 64-bit (or 32-bit into shared memory), a tied input
/// must have its output's type, and every `$n` must name an operand.
#[test]
fn inline_asm_constraints_match_their_operands() {
    let cases: Vec<((String, u32), &str)> = vec![
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlineAsm::new("add.u32 $0, $1, $2;", "=r,r,r").output::<u32>().arg(x.load(t)).emit();
                x.store(t, y);
            }),
            "reads 2 operand(s) (0 + entries and 2 input(s)), but 1 were given with .arg",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let (a, b) =
                    InlineAsm::new("mov.u32 $0, $1;", "=r,r").output::<u32>().output::<u32>().arg(x.load(t)).emit();
                x.store(t, a + b);
            }),
            "has 1 output(s) (= and + entries), but 2 type(s) were given with .output::<T>()",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlineAsm::new("add.u32 $0, $0, 1;", "+r").output::<u32>().arg(t).emit();
                x.store(t, y);
            }),
            "inline assembly: input $1 (i32) is tied to output $0 (u32), so they must have one type",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint64 =
                    InlineAsm::new("mov.b64 $0, $1;", "=l,r").output::<u64>().arg(x.load(t).to::<u64>()).emit();
                x.store(t, y.to::<u32>());
            }),
            "inline assembly: input $1 is u64, which a 32-bit integer register (constraint \"r\") does not hold; use \"l\"",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlineAsm::new("shl.b32 $0, $1, $2;", "=r,r,n")
                    .output::<u32>()
                    .arg(x.load(t))
                    .arg(x.load(t))
                    .emit();
                x.store(t, y);
            }),
            "inline assembly: input $2 has the immediate constraint \"n\", so it must be a constant",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 =
                    InlineAsm::new("add.u32 $0, $1, $3;", "=r,r,r").output::<u32>().arg(x.load(t)).arg(1u32).emit();
                x.store(t, y);
            }),
            "inline assembly: the template refers to $3, but there are 3 operand(s) ($0 to $2)",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlineAsm::new("mov.u32 $0, $L;", "=r").output::<u32>().emit();
                x.store(t, y);
            }),
            "inline assembly: `$` must be followed by an operand number (or written `$$`)",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                InlineAsm::new("st.global.u32 [$0], $1;", "l,r").arg_ptr(&x.ptr()).arg(t).emit();
            }),
            "inline assembly with no outputs, no side effects and no memory clobber does nothing",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlineAsm::new("mov.u32 $0, $1;", "r,=r").output::<u32>().arg(x.load(t)).emit();
                x.store(t, y);
            }),
            "must list the outputs, then the inputs, then the clobbers",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlineAsm::new("mov.u32 $0, $1;", "=r,r,").output::<u32>().arg(x.load(t)).emit();
                x.store(t, y);
            }),
            "inline_asm: entry 2 of the constraint string \"=r,r,\" is empty",
        ),
        // Codes NVPTX does not take, which LLVM 23 would drop with the
        // statement (LLVM 21 rejects them).
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 =
                    InlineAsm::new("add.u32 $0, $1, $2;", "=r,r,x").output::<u32>().arg(x.load(t)).arg(t).emit();
                x.store(t, y);
            }),
            "inline assembly: input $2 has the constraint \"x\", which is not an NVPTX register class: b (Boolean), c \
             (8-bit), h (16-bit), r (32-bit), l (64-bit), f (Float32) or d (Float64), or the immediate n (an integer \
             constant); use \"r\" for i32",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlineAsm::new("mov.u32 $0, $1;", "=x,r").output::<u32>().arg(x.load(t)).emit();
                x.store(t, y);
            }),
            "inline assembly: output $0 has the constraint \"x\", which is not an NVPTX register class: b (Boolean), \
             c (8-bit), h (16-bit), r (32-bit), l (64-bit), f (Float32) or d (Float64); use \"r\" for u32",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlineAsm::new("mov.u32 $0, 1;", "=n").output::<u32>().emit();
                x.store(t, y);
            }),
            "inline assembly: output $0 has the immediate constraint \"n\", but an output is a register; use \"r\" for u32",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint64 =
                    InlineAsm::new("mov.b64 $0, $1;", "=q,l").output::<u64>().arg(x.load(t).to::<u64>()).emit();
                x.store(t, y.to::<u32>());
            }),
            "inline assembly: output $0 has the constraint \"q\", a 128-bit register, and cute-rs has no 128-bit type; \
             use \"l\" for u64",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlineAsm::new("ld.u32 $0, $1;", "=r,m").output::<u32>().arg(x.load(t)).emit();
                x.store(t, y);
            }),
            "inline assembly: input $1 has the constraint \"m\", which is not an NVPTX register class",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlineAsm::new("mov.u32 $0, $1;", "=r,&r").output::<u32>().arg(x.load(t)).emit();
                x.store(t, y);
            }),
            "inline assembly: input $1 has the constraint \"&r\", which is not an NVPTX register class",
        ),
        // An immediate is an integer constant: LLVM rejects a float (LLVM 23
        // only prints its error) and prints a Boolean `true` as -1.
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Float32 = InlineAsm::new("add.f32 $0, $1, $2;", "=f,f,n")
                    .output::<f32>()
                    .arg(x.load(t).to::<f32>())
                    .arg(1.5f32)
                    .emit();
                x.store(t, y.to::<u32>());
            }),
            "inline assembly: input $2 has the immediate constraint \"n\", which takes an integer constant, and this \
             one is f32; pass it in a register (\"f\")",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 =
                    InlineAsm::new("add.u32 $0, $1, $2;", "=r,r,i").output::<u32>().arg(x.load(t)).arg(true).emit();
                x.store(t, y);
            }),
            "inline assembly: input $2 has the immediate constraint \"i\", which takes an integer constant, and this \
             one is i1; pass it in a register (\"b\")",
        ),
        // Addresses are 64-bit, and 32-bit only into shared memory.
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlineAsm::new("ld.global.u32 $0, [$1];", "=r,r")
                    .output::<u32>()
                    .arg_ptr(&x.ptr().offset(t))
                    .side_effects()
                    .emit();
                x.store(t, y);
            }),
            "inline assembly: input $1 is the address of global memory, which needs a 64-bit register (constraint \
             \"l\"); only a shared-memory address fits in 32 bits (\"r\")",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let smem = alloc_smem::<u32>(32, 16);
                let y: Uint32 =
                    InlineAsm::new("ld.shared.u32 $0, [$1];", "=r,h").output::<u32>().arg_ptr(&smem.offset(t)).emit();
                x.store(t, y);
            }),
            "inline assembly: input $1 is an address, which is a 64-bit register (constraint \"l\"), or a 32-bit one \
             (\"r\") for shared memory, not \"h\"",
        ),
    ];
    for ((msg, line), want) in cases {
        assert!(msg.contains(want), "expected {want:?} in {msg}");
        assert!(
            msg.contains(&format!("tests/diagnostics/carry_and_asm.rs:{line}:")),
            "the error should point at line {line}: {msg}"
        );
    }
    // What NVPTX takes builds: an early-clobber output, the immediate `i`, a
    // shared address in 32 or 64 bits, clobbers LLVM ignores, and a memory
    // clobber written in the constraint string, which keeps a statement
    // without outputs (with side effects) as .memory_clobber() does.
    let kernel = Kernel::builder("constraints_ok").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<u32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        let smem = alloc_smem::<u32>(32, 16);
        let y: Uint32 =
            InlineAsm::new("add.u32 $0, $1, $2;", "=&r,r,i").output::<u32>().arg(x.load(t)).arg(-5i32).emit();
        InlineAsm::new("st.shared.u32 [$0], $1;", "r,r,~{memory}").arg_ptr(&smem.offset(t)).arg(y).emit();
        InlineAsm::new("st.shared.u32 [$0], $1;", "l,r, ~{cc}, ~{memory}").arg_ptr(&smem.offset(t)).arg(y).emit();
        x.store(t, make_tensor(&smem, Layout::col_major(32)).load(t));
    });
    let asm: Vec<String> =
        kernel.to_mlir().lines().filter(|l| l.contains("llvm.inline_asm")).map(String::from).collect();
    assert_eq!(asm.len(), 3, "{asm:#?}");
    assert!(asm[1..].iter().all(|l| l.contains("llvm.inline_asm has_side_effects")), "{asm:#?}");
}

/// A hand-written MLIR operation: every `%n` must name an operand and every
/// operand appear, and the text is one operation without its results.
#[test]
fn mlir_op_text_is_checked() {
    let cases: Vec<((String, u32), &str)> = vec![
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 =
                    MlirOp::new("llvm.intr.bswap(%1) : (i32) -> i32").operand(x.load(t)).result::<u32>().emit();
                x.store(t, y);
            }),
            "mlir_op: `%1` names no operand: there are 1 (%0 to %0)",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = MlirOp::new("llvm.intr.bswap(%0) : (i32) -> i32")
                    .operand(x.load(t))
                    .operand(t)
                    .result::<u32>()
                    .emit();
                x.store(t, y);
            }),
            "mlir_op: operand %1 does not appear in",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Int32 = MlirOp::new("%r = nvvm.read.ptx.sreg.nsmid : i32").result::<i32>().emit();
                x.store(t, y.bitcast::<u32>());
            }),
            "`%r` is not an operand: the operands are %0, %1, ... in order (write the operation without its results",
        ),
    ];
    for ((msg, line), want) in cases {
        assert!(msg.contains(want), "expected {want:?} in {msg}");
        assert!(
            msg.contains(&format!("tests/diagnostics/carry_and_asm.rs:{line}:")),
            "the error should point at line {line}: {msg}"
        );
    }
}

/// Without a simulation closure, hand-written PTX and MLIR can be compiled
/// but not simulated, and a closure must return one value per write-only
/// output; the errors name the line.
#[test]
fn inline_statements_need_a_simulation() {
    let run = |kernel: Kernel| simulate_err(&kernel, 32, &mut [KernelArg::buffer(&[3u32; 32])]);
    let line = line!() + 4;
    let kernel = Kernel::builder("unsimulated").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<u32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        let y: Uint32 = InlinePtx::new("popc.b32 {$w0}, {$r0};").write::<u32>().read(x.load(t)).pure().emit();
        x.store(t, y);
    });
    let e = run(kernel);
    assert!(e.message.contains("cannot be simulated; describe it with a Rust closure (InlinePtx::sim"), "{e}");
    assert!(e.to_string().contains(&format!("tests/diagnostics/carry_and_asm.rs:{line}:")), "{e}");
    let kernel = Kernel::builder("unsimulated_mlir").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<u32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        let y: Uint32 = MlirOp::new("llvm.intr.ctpop(%0) : (i32) -> i32").operand(x.load(t)).result::<u32>().emit();
        x.store(t, y);
    });
    assert!(
        run(kernel).message.contains("the MLIR operation `llvm.intr.ctpop(%0) : (i32) -> i32` cannot be simulated")
    );
    let kernel = Kernel::builder("short_simulation").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<u32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        let (a, b) = InlinePtx::new("mov.b64 {{$w0}, {$w1}}, {$r0};")
            .write::<u32>()
            .write::<u32>()
            .read(x.load(t).to::<u64>())
            .sim(|r, _| vec![r[0]])
            .emit();
        x.store(t, a ^ b);
    });
    assert!(run(kernel).message.contains("inline PTX simulation returned 1 values, not 2"));
}

/// The validator checks inline assembly built as IR by hand too: an empty
/// entry of the constraint string, which LLVM rejects, is named.
#[test]
fn inline_asm_ir_is_validated() {
    let kernel = Kernel::builder("asm_ir").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<u32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        let y: Uint32 = InlineAsm::new("add.u32 $0, $1, $2;", "=r,r,r").output::<u32>().arg(x.load(t)).arg(t).emit();
        x.store(t, y);
    });
    let with_constraints = |constraints: &str| {
        let mut ir = kernel.ir().clone();
        for inst in &mut ir.body.insts {
            if let cute::ir::Op::InlinePtx { constraints: c, .. } = &mut inst.op {
                *c = constraints.to_string();
            }
        }
        Kernel::from_ir(ir).map(|_| ()).err().map(|e| e.to_string())
    };
    let e = with_constraints("=r,,r").expect("an empty entry is rejected");
    assert!(e.contains("inline assembly: entry 1 of the constraint string \"=r,,r\" is empty"), "{e}");
    let e = with_constraints("=r,r,r,").expect("a trailing comma is rejected");
    assert!(e.contains("entry 3 of the constraint string"), "{e}");
    let e = with_constraints("=r,r,l").expect("a 64-bit register for an Int32 is rejected");
    assert!(e.contains("input $2 is i32, which a 64-bit integer register (constraint \"l\") does not hold"), "{e}");
    assert_eq!(with_constraints("=r,r,r"), None);
}
