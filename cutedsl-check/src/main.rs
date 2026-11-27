// SPDX-License-Identifier: MIT OR Apache-2.0

//! Replays in cute-rs the layout-algebra calls that `reference/gen_layouts.py` recorded from
//! NVIDIA's CuTe DSL, and compares the answers.
//!
//! ```text
//! cutedsl-check layouts LAYOUTS.tsv [--show N]
//! ```
//!
//! Each line of the file is a call and CuTe DSL's answer as text. The same call is made here
//! and its answer printed the same way; the two texts must be equal. The exit status is 1 when
//! any call ends in one of the columns that count as a disagreement (see [`Verdict`]).
//! Calls that CuTe DSL answers but that are ill-posed, and that cute-rs refuses,
//! have their own column; [`ill_posed`] says how they are recognised.

mod ill_posed;

use std::collections::BTreeMap;
use std::panic::{self, AssertUnwindSafe};
use std::process::ExitCode;

use cute::{
    IndexExpr, IntTuple, Layout, Tiler, Tree, blocked_product, dice, flat_product, is_congruent, is_weakly_congruent,
    logical_product, make_composed_layout, make_layout_tv, make_ordered_layout, make_swizzle, max_common_layout,
    max_common_vector, raked_product, recast_layout, tile_to_shape_ordered, tiled_product, zipped_product,
};

/// How many indices `values` and `swizzle_values` evaluate (the generator's `VALUES`).
const VALUES: usize = 48;

/// How one call compares.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Verdict {
    /// Both answer, with the same text.
    Same,
    /// Both refuse the call (CuTe DSL with an error or a crash).
    BothRefuse,
    /// Both answer with layouts that print differently but map every index to the same offset.
    SameFunction,
    /// CuTe DSL refuses with an error; cute-rs answers.
    OnlyDslRefuses,
    /// The call ends CuTe DSL's process; cute-rs answers.
    DslCrashes,
    /// CuTe DSL answers a call that is ill-posed (see [`ill_posed`]), and
    /// cute-rs refuses it.
    IllPosed,
    /// CuTe DSL answers; cute-rs refuses.
    OnlyOursRefuses,
    /// Both answer, differently.
    Differ,
}

impl Verdict {
    const ALL: [Verdict; 8] = [
        Verdict::Same,
        Verdict::BothRefuse,
        Verdict::SameFunction,
        Verdict::OnlyDslRefuses,
        Verdict::DslCrashes,
        Verdict::IllPosed,
        Verdict::OnlyOursRefuses,
        Verdict::Differ,
    ];

    fn heading(self) -> &'static str {
        match self {
            Verdict::Same => "same",
            Verdict::BothRefuse => "both refuse",
            Verdict::SameFunction => "same map",
            Verdict::OnlyDslRefuses => "DSL refuses",
            Verdict::DslCrashes => "DSL crashes",
            Verdict::IllPosed => "ill-posed",
            Verdict::OnlyOursRefuses => "we refuse",
            Verdict::Differ => "differ",
        }
    }

    /// Whether the call counts against cute-rs.
    fn is_disagreement(self) -> bool {
        matches!(self, Verdict::OnlyOursRefuses | Verdict::Differ)
    }
}

fn layout(s: &str) -> Layout {
    s.parse().unwrap_or_else(|e| panic!("unreadable layout {s}: {e}"))
}

fn tuple(s: &str) -> IntTuple {
    s.parse().unwrap_or_else(|e| panic!("unreadable tuple {s}: {e}"))
}

fn int(s: &str) -> usize {
    s.parse().unwrap_or_else(|e| panic!("unreadable integer {s}: {e}"))
}

/// A flat tuple of integers, as a list: `(2,0,1)`.
fn list(s: &str) -> Vec<usize> {
    entries(s).into_iter().map(|e| int(&e)).collect()
}

/// The entries of a flat tuple: `(3,_,1)` gives `3`, `_`, `1`.
fn entries(s: &str) -> Vec<String> {
    let inner = s.strip_prefix('(').and_then(|s| s.strip_suffix(')')).unwrap_or_else(|| panic!("unreadable tuple {s}"));
    if inner.is_empty() { Vec::new() } else { inner.split(',').map(str::to_string).collect() }
}

/// `a:b;c:d`, a tiler of one layout per mode.
fn tilers(s: &str) -> Tiler {
    Tree::Branch(s.split(';').map(|l| Tree::Leaf(Some(layout(l)))).collect())
}

fn values(n: usize, at: impl Fn(usize) -> usize) -> String {
    let v: Vec<String> = (0..n.min(VALUES)).map(|i| at(i).to_string()).collect();
    format!("({})", v.join(","))
}

/// cute-rs's answer to one recorded call, as CuTe DSL would print it.
fn answer(op: &str, a: &[&str]) -> String {
    match op {
        "size" => layout(a[0]).size().to_string(),
        "cosize" => layout(a[0]).cosize().to_string(),
        "rank" => layout(a[0]).rank().to_string(),
        "depth" => layout(a[0]).depth().to_string(),
        "values" => {
            let l = layout(a[0]);
            values(l.size(), |i| l.call(i))
        }
        "coalesce" => layout(a[0]).coalesce().to_string(),
        "flatten" => layout(a[0]).flatten().to_string(),
        "filter" => layout(a[0]).filter().to_string(),
        "filter_zeros" => layout(a[0]).filter_zeros().to_string(),
        "nullspace" => layout(a[0]).nullspace().to_string(),
        "right_inverse" => layout(a[0]).right_inverse().to_string(),
        "left_inverse" => layout(a[0]).left_inverse().to_string(),
        "idx2crd" => Layout::col_major(tuple(a[1])).idx2crd(int(a[0])).to_string(),
        "crd2idx" => layout(a[1]).crd2idx(&tuple(a[0])).to_string(),
        "composition" => layout(a[0]).compose(&layout(a[1])).to_string(),
        "complement" => layout(a[0]).complement(int(a[1])).to_string(),
        "logical_divide" => layout(a[0]).logical_divide(&Tiler::from(layout(a[1]))).to_string(),
        "zipped_divide" => layout(a[0]).zipped_divide(&tilers(a[1])).to_string(),
        "tiled_divide" => layout(a[0]).tiled_divide(&tilers(a[1])).to_string(),
        "flat_divide" => layout(a[0]).flat_divide(&tilers(a[1])).to_string(),
        "logical_product" => logical_product(&layout(a[0]), layout(a[1])).to_string(),
        "blocked_product" => blocked_product(&layout(a[0]), &layout(a[1])).to_string(),
        "raked_product" => raked_product(&layout(a[0]), &layout(a[1])).to_string(),
        "zipped_product" => zipped_product(&layout(a[0]), &layout(a[1])).to_string(),
        "tiled_product" => tiled_product(&layout(a[0]), &layout(a[1])).to_string(),
        "flat_product" => flat_product(&layout(a[0]), &layout(a[1])).to_string(),
        "tile_to_shape" => tile_to_shape_ordered(&layout(a[0]), tuple(a[1]), &list(a[2])).to_string(),
        "make_layout_tv" => {
            let (tiler, tv) = make_layout_tv(&layout(a[0]), &layout(a[1]));
            format!("({tiler},{tv})")
        }
        "recast_layout" => recast_layout(int(a[0]), int(a[1]), &layout(a[2])).to_string(),
        "max_common_vector" => max_common_vector(&layout(a[0]), &layout(a[1])).to_string(),
        "max_common_layout" => max_common_layout(&layout(a[0]), &layout(a[1])).to_string(),
        "append" => layout(a[0]).append(layout(a[1])).to_string(),
        "prepend" => layout(a[0]).prepend(layout(a[1])).to_string(),
        "group_modes" => layout(a[0]).group(int(a[1]), int(a[2])).to_string(),
        "select" => layout(a[0]).select(&list(a[1])).to_string(),
        // CuTe DSL's `dice` keeps the modes the coordinate fixes, `slice_` the ones it leaves open.
        "dice" => {
            let mask: Vec<bool> = entries(a[1]).iter().map(|e| e != "_").collect();
            dice(&layout(a[0]), &mask).to_string()
        }
        "slice" => {
            let coord: Vec<Tree<Option<IndexExpr>>> = entries(a[1])
                .iter()
                .map(|e| Tree::Leaf(if e == "_" { None } else { Some(IndexExpr::from(int(e) as i64)) }))
                .collect();
            layout(a[0]).slice(&Tree::Branch(coord)).0.to_string()
        }
        "make_ordered_layout" => make_ordered_layout(tuple(a[0]), &list(a[1])).to_string(),
        "is_congruent" => is_congruent(&tuple(a[0]), &tuple(a[1])).to_string(),
        "is_weakly_congruent" => is_weakly_congruent(&tuple(a[0]), &tuple(a[1])).to_string(),
        "swizzle_values" => {
            let shift: i32 = a[2].parse().unwrap_or_else(|e| panic!("unreadable shift {}: {e}", a[2]));
            let outer = layout(a[3]);
            let n = outer.size();
            let composed = make_composed_layout(make_swizzle(int(a[0]) as u32, int(a[1]) as u32, shift), 0, outer);
            values(n, |i| composed.call(i))
        }
        _ => panic!("unknown operation {op}"),
    }
}

/// Whether two answers are layouts of the same size that send every index to the same offset.
fn same_function(a: &str, b: &str) -> bool {
    let (Ok(a), Ok(b)) = (a.parse::<Layout>(), b.parse::<Layout>()) else { return false };
    panic::catch_unwind(AssertUnwindSafe(|| {
        a.is_static()
            && b.is_static()
            && a.size() == b.size()
            && (0..a.size().min(4096)).all(|i| a.call(i) == b.call(i))
    }))
    .unwrap_or(false)
}

fn run(path: &str, show: usize) -> Result<bool, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    // cute-rs refuses a call by panicking; keep the messages out of the report.
    panic::set_hook(Box::new(|_| {}));

    let mut ops: Vec<String> = Vec::new();
    let mut counts: BTreeMap<(String, Verdict), usize> = BTreeMap::new();
    let mut examples: BTreeMap<(Verdict, String), Vec<String>> = BTreeMap::new();
    let mut header = String::new();
    let mut classes: BTreeMap<ill_posed::Class, usize> = BTreeMap::new();
    for (n, line) in text.lines().enumerate() {
        if let Some(h) = line.strip_prefix("# ") {
            header = h.to_string();
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        let arrow = fields.iter().position(|f| *f == "=>").ok_or_else(|| format!("{path}:{}: no `=>`", n + 1))?;
        let (op, args, theirs) = (fields[0], &fields[1..arrow], fields[arrow + 1]);
        if ops.last().map(String::as_str) != Some(op) && !ops.iter().any(|o| o == op) {
            ops.push(op.to_string());
        }
        let ours = panic::catch_unwind(AssertUnwindSafe(|| answer(op, args).replace(' ', "")));
        if let Err(e) = &ours {
            let why = panic_text(e.as_ref());
            if why.starts_with("unreadable") || why.starts_with("unknown operation") {
                return Err(format!("{path}:{}: {why}", n + 1));
            }
        }
        let verdict = match (theirs, &ours) {
            ("ERR" | "CRASH", Err(_)) => Verdict::BothRefuse,
            ("ERR", Ok(_)) => Verdict::OnlyDslRefuses,
            ("CRASH", Ok(_)) => Verdict::DslCrashes,
            (_, Err(_)) => match ill_posed::class(op, args) {
                Some(c) => {
                    *classes.entry(c).or_default() += 1;
                    Verdict::IllPosed
                }
                None => Verdict::OnlyOursRefuses,
            },
            (t, Ok(o)) if t == o => Verdict::Same,
            (t, Ok(o)) if same_function(t, o) => Verdict::SameFunction,
            _ => Verdict::Differ,
        };
        *counts.entry((op.to_string(), verdict)).or_default() += 1;
        if verdict != Verdict::Same && verdict != Verdict::BothRefuse {
            let list = examples.entry((verdict, op.to_string())).or_default();
            if list.len() < show {
                let ours = match &ours {
                    Ok(o) => o.clone(),
                    Err(e) => format!("refused: {}", panic_text(e.as_ref())),
                };
                list.push(format!("{op}({})\n      CuTe DSL: {theirs}\n      cute-rs:  {ours}", args.join(", ")));
            }
        }
    }
    let _ = panic::take_hook();

    println!("reference: {header}");
    print!("{:<22}{:>7}", "operation", "calls");
    for v in Verdict::ALL {
        print!("{:>13}", v.heading());
    }
    println!();
    let mut totals = [0usize; Verdict::ALL.len()];
    for op in &ops {
        let row: Vec<usize> =
            Verdict::ALL.iter().map(|v| counts.get(&(op.clone(), *v)).copied().unwrap_or(0)).collect();
        print!("{:<22}{:>7}", op, row.iter().sum::<usize>());
        for (i, c) in row.iter().enumerate() {
            totals[i] += c;
            print!("{:>13}", c);
        }
        println!();
    }
    print!("{:<22}{:>7}", "total", totals.iter().sum::<usize>());
    for c in totals {
        print!("{:>13}", c);
    }
    println!();
    println!("\nill-posed calls by class:");
    for c in ill_posed::Class::ALL {
        println!("  {:<54}{:>7}", c.name(), classes.get(&c).copied().unwrap_or(0));
    }
    for ((verdict, op), list) in examples.iter().filter(|(_, list)| !list.is_empty()) {
        println!("\n{} / {op}:", verdict.heading());
        for e in list {
            println!("  {e}");
        }
    }
    Ok(Verdict::ALL.iter().zip(totals).all(|(v, c)| !v.is_disagreement() || c == 0))
}

/// The message of a caught panic.
fn panic_text(e: &(dyn std::any::Any + Send)) -> String {
    let text = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()));
    text.unwrap_or_else(|| "a panic".to_string())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: cutedsl-check layouts LAYOUTS.tsv [--show N]";
    let (Some("layouts"), Some(path)) = (args.first().map(String::as_str), args.get(1)) else {
        eprintln!("{usage}");
        return ExitCode::from(2);
    };
    let show = match args.get(2).map(String::as_str) {
        None => 0,
        Some("--show") => match args.get(3).and_then(|n| n.parse().ok()) {
            Some(n) => n,
            None => {
                eprintln!("{usage}");
                return ExitCode::from(2);
            }
        },
        Some(_) => {
            eprintln!("{usage}");
            return ExitCode::from(2);
        }
    };
    match run(path, show) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(2)
        }
    }
}
