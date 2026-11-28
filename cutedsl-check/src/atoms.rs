// SPDX-License-Identifier: MIT OR Apache-2.0

//! The questions `reference/gen_atoms.py` asks CuTe DSL about MMA and copy atoms, tiled MMAs,
//! tiled copies and tensor partitions, asked of cute-rs.
//!
//! A line is `query arch args... => answer`. The architecture is the one CuTe DSL traced for
//! and does not change cute-rs's answer. An operation is written `Name(arg,...)` with CuTe DSL's
//! class name; [`mma_op`] and [`copy_op`] give the cute-rs operation for each. A tensor argument
//! is a layout: the tensor is the coordinate tensor `make_tensor(0, layout)`, whose iterator is
//! an integer, so that a partition prints the offset of its first element. A query or operation
//! with no counterpart here panics with "unknown operation", which ends the replay with an error.

use cute::nvgpu::tcgen05;
use cute::nvgpu::warpgroup;
use cute::nvgpu::{OperandMajorMode, warp};
use cute::tensor::Ident;

use crate::ill_posed::Class;
use cute::{
    BF16, CopyAtom, CopyOp, DType, F4E2M1FN, F6E2M3FN, F6E3M2FN, F8E4M3FN, F8E5M2, F8E8M0FNU, F16, IndexExpr, IntTuple,
    Kernel, Layout, MmaOp, Numeric, SliceCoord, TF32, Tensor, TiledCopy, TiledMma, Tiler, Tree, flat_divide,
    local_partition, local_partition_proj, local_tile, local_tile_proj, logical_divide, make_copy_atom,
    make_cotiled_copy, make_mma_atom, make_tensor, make_tiled_copy_a, make_tiled_copy_b, make_tiled_copy_c,
    make_tiled_copy_c_atom, make_tiled_copy_d, make_tiled_copy_s, make_tiled_copy_tv, make_tiled_mma, tiled_divide,
    zipped_divide,
};

/// Run `$body` with `$t` the host type of the element type `$d`.
macro_rules! with_type {
    ($d:expr, $t:ident => $body:expr) => {
        match $d {
            DType::F16 => {
                type $t = F16;
                $body
            }
            DType::BF16 => {
                type $t = BF16;
                $body
            }
            DType::F32 => {
                type $t = f32;
                $body
            }
            DType::F64 => {
                type $t = f64;
                $body
            }
            DType::TF32 => {
                type $t = TF32;
                $body
            }
            DType::I8 => {
                type $t = i8;
                $body
            }
            DType::U8 => {
                type $t = u8;
                $body
            }
            DType::I32 => {
                type $t = i32;
                $body
            }
            DType::F8E4M3FN => {
                type $t = F8E4M3FN;
                $body
            }
            DType::F8E5M2 => {
                type $t = F8E5M2;
                $body
            }
            DType::F8E8M0FNU => {
                type $t = F8E8M0FNU;
                $body
            }
            DType::F6E2M3FN => {
                type $t = F6E2M3FN;
                $body
            }
            DType::F6E3M2FN => {
                type $t = F6E3M2FN;
                $body
            }
            DType::F4E2M1FN => {
                type $t = F4E2M1FN;
                $body
            }
            other => panic!("unknown operation: no element type {other} here"),
        }
    };
}

// ------------------------------------------------------------------ reading

fn layout(s: &str) -> Layout {
    s.parse().unwrap_or_else(|e| panic!("unreadable layout {s}: {e}"))
}

fn tuple(s: &str) -> IntTuple {
    s.parse().unwrap_or_else(|e| panic!("unreadable tuple {s}: {e}"))
}

fn int(s: &str) -> usize {
    s.parse().unwrap_or_else(|e| panic!("unreadable integer {s}: {e}"))
}

/// The entries of `s` split at its top-level commas.
fn split_top(s: &str) -> Vec<String> {
    let (mut out, mut depth, mut cur) = (Vec::new(), 0i32, String::new());
    for c in s.chars() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(std::mem::take(&mut cur));
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// The entries of a tuple: `(3,_,(1,2))` gives `3`, `_`, `(1,2)`.
fn entries(s: &str) -> Vec<String> {
    let inner = s.strip_prefix('(').and_then(|s| s.strip_suffix(')')).unwrap_or_else(|| panic!("unreadable tuple {s}"));
    split_top(inner)
}

/// An operation, `Name(arg,...)`: its name and arguments.
fn spec(s: &str) -> (&str, Vec<String>) {
    let open = s.find('(').unwrap_or_else(|| panic!("unreadable operation {s}"));
    (&s[..open], entries(&s[open..]))
}

fn dtype(s: &str) -> DType {
    DType::from_cute_name(s).unwrap_or_else(|| panic!("unreadable type {s}"))
}

fn shape3(s: &str) -> (usize, usize, usize) {
    let e: Vec<usize> = entries(s).iter().map(|x| int(x)).collect();
    match e[..] {
        [m, n, k] => (m, n, k),
        _ => panic!("unreadable instruction shape {s}"),
    }
}

fn boolean(s: &str) -> bool {
    match s {
        "true" => true,
        "false" => false,
        _ => panic!("unreadable boolean {s}"),
    }
}

fn major(s: &str) -> OperandMajorMode {
    match s {
        "K" => OperandMajorMode::K,
        "MN" => OperandMajorMode::MN,
        _ => panic!("unreadable major mode {s}"),
    }
}

fn cta(s: &str) -> tcgen05::CtaGroup {
    match s {
        "ONE" => tcgen05::CtaGroup::ONE,
        "TWO" => tcgen05::CtaGroup::TWO,
        _ => panic!("unreadable CTA group {s}"),
    }
}

fn umma_src(s: &str) -> tcgen05::OperandSource {
    match s {
        "SMEM" => tcgen05::OperandSource::SMEM,
        "TMEM" => tcgen05::OperandSource::TMEM,
        _ => panic!("unreadable operand source {s}"),
    }
}

fn wgmma_src(s: &str) -> warpgroup::OperandSource {
    match s {
        "SMEM" => warpgroup::OperandSource::SMEM,
        "RMEM" => warpgroup::OperandSource::RMEM,
        _ => panic!("unreadable operand source {s}"),
    }
}

/// `x1` to `x128`.
fn repetition(s: &str) -> tcgen05::Repetition {
    use tcgen05::Repetition::*;
    match s {
        "x1" => X1,
        "x2" => X2,
        "x4" => X4,
        "x8" => X8,
        "x16" => X16,
        "x32" => X32,
        "x64" => X64,
        "x128" => X128,
        _ => panic!("unreadable repetition {s}"),
    }
}

/// `_` or a number of bits.
fn unpack_bits(s: &str) -> Option<u32> {
    if s == "_" { None } else { Some(int(s) as u32) }
}

/// The coordinate tensor `make_tensor(0, layout)`.
fn counting(s: &str) -> Tensor<Ident> {
    make_tensor(0usize, layout(s))
}

/// A slice coordinate, `_` for a mode left open.
fn slice_coord(s: &str) -> SliceCoord {
    Tree::Branch(
        entries(s)
            .iter()
            .map(|e| Tree::Leaf(if e == "_" { None } else { Some(IndexExpr::from(int(e) as i64)) }))
            .collect(),
    )
}

/// A projection, `1` for a mode kept and `_` for one dropped.
fn projection(s: &str) -> Vec<bool> {
    entries(s)
        .iter()
        .map(|e| match e.as_str() {
            "1" => true,
            "_" => false,
            _ => panic!("unreadable projection {s}"),
        })
        .collect()
}

/// `a:b;c:d`, a tiler of one layout per mode.
fn tilers(s: &str) -> Tiler {
    Tree::Branch(s.split(';').map(|l| Tree::Leaf(Some(layout(l)))).collect())
}

// ----------------------------------------------------------------- printing

/// A coordinate tensor as CuTe DSL prints one with an integer iterator.
fn tensor_text(t: &Tensor<Ident>) -> String {
    format!("tensor<{}o{}>", t.origin(), t.layout())
}

/// The first element and the layout of a printed tensor, `tensor<768o(4,2):(1,32)>`.
pub fn split_tensor(s: &str) -> Option<(&str, &str)> {
    let inner = s.strip_prefix("tensor<")?.strip_suffix('>')?;
    let o = inner.find('o')?;
    Some((&inner[..o], &inner[o + 1..]))
}

/// A tiler as CuTe DSL prints a tiled copy's: `(8:1,32:1)`.
fn tiler_text(t: &Tiler) -> String {
    match t {
        Tree::Leaf(Some(l)) => l.to_string(),
        Tree::Leaf(None) => "_".to_string(),
        Tree::Branch(v) => format!("({})", v.iter().map(tiler_text).collect::<Vec<_>>().join(",")),
    }
}

// ------------------------------------------------------------------- MMAs

/// The cute-rs operation of a CuTe DSL MMA operation.
fn mma_op(s: &str) -> MmaOp {
    let (name, a) = spec(s);
    let a: Vec<&str> = a.iter().map(String::as_str).collect();
    match name {
        "warp.MmaF16BF16Op" => cute::MmaF16BF16Op::new(dtype(a[0]), dtype(a[1]), shape3(a[2])).into(),
        "warp.MmaTF32Op" => cute::MmaTF32Op::new(shape3(a[0])).into(),
        "warp.MmaFP8Op" => cute::MmaFP8Op::new(dtype(a[0]), dtype(a[1]), shape3(a[2])).into(),
        "MmaUniversalOp" => MmaOp::UniversalFma(dtype(a[0])),
        "warp.MmaMXF4Op" => warp::MmaMXF4Op::new(dtype(a[0]), dtype(a[1]), dtype(a[2])).into(),
        "warp.MmaMXF4NVF4Op" => warp::MmaMXF4NVF4Op::new(dtype(a[0]), dtype(a[1]), dtype(a[2])).into(),
        "warp.MmaMXF8Op" => warp::MmaMXF8Op::new(dtype(a[0]), dtype(a[1]), dtype(a[2])).into(),
        "warp.MmaMXF8F6F4Op" => warp::MmaMXF8F6F4Op::new(dtype(a[0]), dtype(a[1]), dtype(a[2]), dtype(a[3])).into(),
        "warpgroup.MmaF16BF16Op" => warpgroup::MmaF16BF16Op::new(
            dtype(a[0]),
            dtype(a[1]),
            shape3(a[2]),
            wgmma_src(a[3]),
            major(a[4]),
            major(a[5]),
        )
        .into(),
        "warpgroup.MmaF8Op" => warpgroup::MmaF8Op::new(
            dtype(a[0]),
            dtype(a[1]),
            dtype(a[2]),
            shape3(a[3]),
            wgmma_src(a[4]),
            major(a[5]),
            major(a[6]),
        )
        .into(),
        "warpgroup.MmaI8Op" => warpgroup::MmaI8Op::new(
            dtype(a[0]),
            dtype(a[1]),
            dtype(a[2]),
            shape3(a[3]),
            wgmma_src(a[4]),
            major(a[5]),
            major(a[6]),
        )
        .into(),
        "tcgen05.MmaF16BF16Op" => tcgen05::MmaF16BF16Op::new(
            dtype(a[0]),
            dtype(a[1]),
            shape3(a[2]),
            cta(a[3]),
            umma_src(a[4]),
            major(a[5]),
            major(a[6]),
        )
        .into(),
        "tcgen05.MmaTF32Op" => {
            tcgen05::MmaTF32Op::new(shape3(a[0]), cta(a[1]), umma_src(a[2]), major(a[3]), major(a[4])).into()
        }
        "tcgen05.MmaI8Op" => {
            tcgen05::MmaI8Op::new(dtype(a[0]), shape3(a[1]), cta(a[2]), umma_src(a[3]), major(a[4]), major(a[5])).into()
        }
        "tcgen05.MmaF8F6F4Op" => tcgen05::MmaF8F6F4Op::new(
            dtype(a[0]),
            dtype(a[1]),
            dtype(a[2]),
            shape3(a[3]),
            cta(a[4]),
            umma_src(a[5]),
            major(a[6]),
            major(a[7]),
        )
        .into(),
        "tcgen05.MmaMXF8F6F4Op" => tcgen05::MmaMXF8F6F4Op::new(
            dtype(a[0]),
            dtype(a[1]),
            shape3(a[2]),
            cta(a[3]),
            umma_src(a[4]),
            major(a[5]),
            major(a[6]),
        )
        .into(),
        "tcgen05.MmaMXF4Op" => tcgen05::MmaMXF4Op::new(shape3(a[0]), cta(a[1]), umma_src(a[2])).into(),
        "tcgen05.MmaMXF4NVF4Op" => {
            tcgen05::MmaMXF4NVF4Op::new(dtype(a[0]), shape3(a[1]), cta(a[2]), umma_src(a[3])).into()
        }
        "tcgen05.MmaF16BF16SparseOp" => {
            assert_eq!(a[7], "TID", "unreadable sparse metadata format {}", a[7]);
            tcgen05::MmaF16BF16SparseOp::new(
                dtype(a[0]),
                dtype(a[1]),
                shape3(a[2]),
                cta(a[3]),
                umma_src(a[4]),
                major(a[5]),
                major(a[6]),
                tcgen05::SparseMetadataFormat::Tid,
            )
            .into()
        }
        _ => panic!("unknown operation {name}"),
    }
}

/// `make_tiled_mma(op, atom_layout, permutation_mnk)`; the permutation is `(_,_,_)` for none.
fn tiled_mma(op: &str, atom_layout: &str, perm: &str) -> TiledMma {
    let tm = make_tiled_mma(make_mma_atom(mma_op(op)), layout(atom_layout));
    let p: Vec<Option<Layout>> = entries(perm).iter().map(|e| if e == "_" { None } else { Some(layout(e)) }).collect();
    if p.iter().all(Option::is_none) { tm } else { tm.with_permutation_mnk([p[0].clone(), p[1].clone(), p[2].clone()]) }
}

fn mma_atom_query(q: &str, a: &[&str]) -> String {
    let atom = make_mma_atom(mma_op(a[0]));
    match q {
        "shape_mnk" => {
            let [m, n, k] = atom.shape_mnk();
            format!("({m},{n},{k})")
        }
        "thr_id" => atom.thr_id().to_string(),
        "tv_A" => atom.layout_a().to_string(),
        "tv_B" => atom.layout_b().to_string(),
        "tv_C" => atom.layout_c().to_string(),
        _ => panic!("unknown operation mma_atom.{q}"),
    }
}

fn tiled_mma_query(q: &str, a: &[&str]) -> String {
    let tm = tiled_mma(a[0], a[1], a[2]);
    let slice = || tm.get_slice(int(a[4]) as i64);
    match q {
        "thr_layout_vmnk" => tm.thr_layout_vmnk().to_string(),
        "tile_size" => format!("({},{},{})", tm.tile_size(0), tm.tile_size(1), tm.tile_size(2)),
        "size" => tm.num_threads().to_string(),
        "tv_A" => tm.layout_a_tv().to_string(),
        "tv_B" => tm.layout_b_tv().to_string(),
        "tv_C" => tm.layout_c_tv().to_string(),
        "partition_shape_A" => tm.partition_shape_a(tuple(a[3])).to_string(),
        "partition_shape_B" => tm.partition_shape_b(tuple(a[3])).to_string(),
        "partition_shape_C" => tm.partition_shape_c(tuple(a[3])).to_string(),
        "partition_A" => tensor_text(&slice().partition_a(&counting(a[3]))),
        "partition_B" => tensor_text(&slice().partition_b(&counting(a[3]))),
        "partition_C" => tensor_text(&slice().partition_c(&counting(a[3]))),
        "fragment_A" | "fragment_B" | "fragment_C" => {
            let (ta, tb, tc) = tm.atom().dtypes();
            let t = counting(a[3]);
            let s = slice();
            // Fragments are registers (or tensor memory) of a kernel: make them while one is built.
            let mut out = String::new();
            Kernel::build("fragment", || {
                out = match q {
                    "fragment_A" => with_type!(ta, T => s.partition_fragment_a::<T, _>(&t).layout().to_string()),
                    "fragment_B" => with_type!(tb, T => s.partition_fragment_b::<T, _>(&t).layout().to_string()),
                    _ => with_type!(tc, T => s.partition_fragment_c::<T, _>(&t).layout().to_string()),
                }
            });
            out
        }
        _ => panic!("unknown operation tiled_mma.{q}"),
    }
}

// ------------------------------------------------------------------ copies

/// The cute-rs operation of a CuTe DSL copy operation and the atom's element type.
fn copy_op(s: &str) -> (CopyOp, DType) {
    let (name, a) = spec(s);
    let a: Vec<&str> = a.iter().map(String::as_str).collect();
    let ty = dtype(a[0]);
    let op = match name {
        "CopyUniversalOp" => CopyOp::Universal { bits: int(a[1]) },
        "cpasync.CopyG2SOp" => CopyOp::CpAsync {
            bytes: int(a[1]) / 8,
            bypass_l1: match a[2] {
                "ALWAYS" => false,
                "GLOBAL" => true,
                m => panic!("unknown operation: cache mode {m}"),
            },
        },
        "CopyG2ROp" => CopyOp::G2R { bits: int(a[1]), attrs: Default::default() },
        "CopyR2GOp" => CopyOp::R2G { bits: int(a[1]), attrs: Default::default() },
        "CopyS2ROp" => CopyOp::S2R { bits: int(a[1]), attrs: Default::default() },
        "CopyR2SOp" => CopyOp::R2S { bits: int(a[1]), attrs: Default::default() },
        "warp.LdMatrix8x8x16bOp" => CopyOp::LdMatrix { num: int(a[2]), trans: boolean(a[1]) },
        "warp.StMatrix8x8x16bOp" => CopyOp::StMatrix { num: int(a[2]), trans: boolean(a[1]) },
        "warp.LdMatrix8x16x8bOp" => CopyOp::LdMatrix8x16x8b { num: int(a[1]), unpack_bits: unpack_bits(a[2]) },
        "warp.LdMatrix16x16x8bOp" => {
            CopyOp::LdMatrix16x16x8b { num: int(a[2]), trans: boolean(a[1]), unpack_bits: unpack_bits(a[3]) }
        }
        "warp.LdMatrix16x8x8bOp" => {
            CopyOp::LdMatrix16x8x8b { num: int(a[2]), trans: boolean(a[1]), unpack_bits: unpack_bits(a[3]) }
        }
        "warp.StMatrix16x8x8bOp" => CopyOp::StMatrix16x8x8b { num: int(a[2]), trans: boolean(a[1]) },
        _ if name.starts_with("tcgen05.Ld") => {
            let pack = match a[2] {
                "NONE" => tcgen05::Pack::None,
                "PACK_16b_IN_32b" => tcgen05::Pack::Pack16bIn32b,
                p => panic!("unreadable packing {p}"),
            };
            let r = repetition(a[1]);
            match &name[8..] {
                "Ld32x32bOp" => tcgen05::Ld32x32bOp::new(r).with_pack(pack).into(),
                "Ld16x64bOp" => tcgen05::Ld16x64bOp::new(r).with_pack(pack).into(),
                "Ld16x128bOp" => tcgen05::Ld16x128bOp::new(r).with_pack(pack).into(),
                "Ld16x256bOp" => tcgen05::Ld16x256bOp::new(r).with_pack(pack).into(),
                "Ld16x32bx2Op" => tcgen05::Ld16x32bx2Op::new(r).with_pack(pack).into(),
                _ => panic!("unknown operation {name}"),
            }
        }
        _ if name.starts_with("tcgen05.St") => {
            let unpack = match a[2] {
                "NONE" => tcgen05::Unpack::None,
                "UNPACK_32b_IN_16b" => tcgen05::Unpack::Unpack32bIn16b,
                p => panic!("unreadable unpacking {p}"),
            };
            let r = repetition(a[1]);
            match &name[8..] {
                "St32x32bOp" => tcgen05::St32x32bOp::new(r).with_unpack(unpack).into(),
                "St16x64bOp" => tcgen05::St16x64bOp::new(r).with_unpack(unpack).into(),
                "St16x128bOp" => tcgen05::St16x128bOp::new(r).with_unpack(unpack).into(),
                "St16x256bOp" => tcgen05::St16x256bOp::new(r).with_unpack(unpack).into(),
                "St16x32bx2Op" => tcgen05::St16x32bx2Op::new(r).with_unpack(unpack).into(),
                _ => panic!("unknown operation {name}"),
            }
        }
        _ => panic!("unknown operation {name}"),
    };
    (op, ty)
}

fn atom_of<T: Numeric>(op: CopyOp) -> CopyAtom<T> {
    make_copy_atom::<T>(op)
}

fn copy_atom_query<T: Numeric>(op: CopyOp, q: &str) -> String {
    let atom = atom_of::<T>(op);
    match q {
        "thr_id" => atom.thr_id().to_string(),
        "src_tv" => atom.layout_src_tv().to_string(),
        "dst_tv" => atom.layout_dst_tv().to_string(),
        _ => panic!("unknown operation copy_atom.{q}"),
    }
}

/// A query of a tiled copy; `rest` are the query's own arguments.
fn tiled_copy_query<T: Numeric>(tc: &TiledCopy<T>, q: &str, rest: &[&str]) -> String {
    match q {
        "layout_tv" => tc.layout_tv_tiled().to_string(),
        "tiler" => tiler_text(tc.tiler_mn()),
        "src_tv_tiled" => tc.layout_src_tv_tiled().to_string(),
        "dst_tv_tiled" => tc.layout_dst_tv_tiled().to_string(),
        "size" => tc.size().to_string(),
        "partition_S" => tensor_text(&tc.get_slice(int(rest[1]) as i64).partition_s(&counting(rest[0]))),
        "partition_D" => tensor_text(&tc.get_slice(int(rest[1]) as i64).partition_d(&counting(rest[0]))),
        "retile" => tc.retile(&counting(rest[0])).layout().to_string(),
        _ => panic!("unknown operation of a tiled copy: {q}"),
    }
}

fn copy_query<T: Numeric>(op: CopyOp, family: &str, q: &str, a: &[&str]) -> String {
    let atom = atom_of::<T>(op);
    match family {
        "copy_atom" => copy_atom_query::<T>(op, q),
        "tiled_copy_tv" => tiled_copy_query(&make_tiled_copy_tv(&atom, &layout(a[1]), &layout(a[2])), q, &a[3..]),
        "tiled_copy_A" | "tiled_copy_B" | "tiled_copy_C" | "tiled_copy_C_atom" => {
            let tm = tiled_mma(a[1], a[2], a[3]);
            let tc = match family {
                "tiled_copy_A" => make_tiled_copy_a(&atom, &tm),
                "tiled_copy_B" => make_tiled_copy_b(&atom, &tm),
                "tiled_copy_C" => make_tiled_copy_c(&atom, &tm),
                _ => make_tiled_copy_c_atom(&atom, &tm),
            };
            tiled_copy_query(&tc, q, &a[4..])
        }
        "cotiled_copy" => tiled_copy_query(&make_cotiled_copy(&atom, &layout(a[1]), &layout(a[2])), q, &a[3..]),
        "tiled_copy_S" | "tiled_copy_D" => {
            let (inner_op, inner_ty) = copy_op(a[1]);
            with_type!(inner_ty, U => {
                let inner = make_tiled_copy_tv(&atom_of::<U>(inner_op), &layout(a[2]), &layout(a[3]));
                let tc = if family == "tiled_copy_S" { make_tiled_copy_s(&atom, &inner) } else { make_tiled_copy_d(&atom, &inner) };
                tiled_copy_query(&tc, q, &a[4..])
            })
        }
        _ => panic!("unknown operation {family}.{q}"),
    }
}

// ------------------------------------------------------------- partitions

fn partition_query(op: &str, a: &[&str]) -> String {
    let t = counting(a[0]);
    let out = match op {
        "local_tile" => local_tile(&t, &tuple(a[1]), slice_coord(a[2])),
        "local_tile_proj" => local_tile_proj(&t, tuple(a[1]), slice_coord(a[2]), &projection(a[3])),
        "local_partition" => local_partition(&t, &layout(a[1]), int(a[2]) as i64),
        "local_partition_proj" => local_partition_proj(&t, &layout(a[1]), int(a[2]) as i64, &projection(a[3])),
        "tensor.logical_divide" => logical_divide(&t, tilers(a[1])),
        "tensor.zipped_divide" => zipped_divide(&t, tilers(a[1])),
        "tensor.tiled_divide" => tiled_divide(&t, tilers(a[1])),
        "tensor.flat_divide" => flat_divide(&t, tilers(a[1])),
        "tensor.slice" => {
            let s = t.slice(slice_coord(a[1]));
            // A coordinate that fixes every mode gives the element itself.
            if s.layout().rank() == 0 || entries(a[1]).iter().all(|e| e != "_") {
                return s.origin().to_string();
            }
            s
        }
        _ => panic!("unknown operation {op}"),
    };
    tensor_text(&out)
}

/// cute-rs's answer to one recorded question, as CuTe DSL prints it. `a[0]` is the
/// architecture CuTe DSL traced for.
pub fn answer(op: &str, a: &[&str]) -> String {
    let a = &a[1..];
    let (family, q) = op.split_once('.').unwrap_or((op, ""));
    match family {
        "mma_atom" => mma_atom_query(q, a),
        "tiled_mma" => tiled_mma_query(q, a),
        "copy_atom" | "tiled_copy_tv" | "tiled_copy_A" | "tiled_copy_B" | "tiled_copy_C" | "tiled_copy_C_atom"
        | "cotiled_copy" | "tiled_copy_S" | "tiled_copy_D" => {
            let (cop, ty) = copy_op(a[0]);
            with_type!(ty, T => copy_query::<T>(cop, family, q, a))
        }
        _ => partition_query(op, a),
    }
}

// ----------------------------------------------------------------- ill-posed

/// The class of an ill-posed question (see [`Class`]), from its arguments alone. The value
/// counts of atoms and the tile sizes of tiled MMAs are read from cute-rs, whose answers to
/// those are compared on their own lines.
pub fn ill_posed(op: &str, a: &[&str]) -> Option<Class> {
    let a = &a[1..];
    let (family, q) = op.split_once('.').unwrap_or((op, ""));
    let caught =
        |f: &dyn Fn() -> Option<Class>| std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).ok().flatten();
    match family {
        "mma_atom" | "tiled_mma" => mma_class(a[0])
            .or_else(|| (family == "tiled_mma" && q == "fragment_C").then(|| caught(&|| tmem_lane_gap(a))).flatten()),
        "tiled_copy_A" | "tiled_copy_B" | "tiled_copy_C" | "tiled_copy_C_atom" => mma_class(a[1]).or_else(|| {
            caught(&|| {
                let tm = tiled_mma(a[1], a[2], a[3]);
                let tv = match family {
                    "tiled_copy_A" => tm.layout_a_tv(),
                    "tiled_copy_B" => tm.layout_b_tv(),
                    _ => tm.layout_c_tv(),
                };
                too_few_values(a[0], tv.mode(1).size())
            })
        }),
        "tiled_copy_tv" => caught(&|| too_few_values(a[0], layout(a[2]).size())),
        "tiled_copy_S" | "tiled_copy_D" => caught(&|| too_few_values(a[0], layout(a[3]).size())),
        "cotiled_copy" => {
            let (atv, data) = (layout(a[1]), layout(a[2]));
            let held: std::collections::HashSet<usize> = (0..data.size()).map(|i| data.call(i)).collect();
            (0..atv.size()).any(|i| !held.contains(&atv.call(i))).then_some(Class::CotiledOutsideData)
        }
        "local_partition" | "local_partition_proj" => {
            let thr = layout(a[1]);
            let mut seen = vec![false; thr.size()];
            let compact = (0..thr.size()).all(|i| {
                let o = thr.call(i);
                o < seen.len() && !std::mem::replace(&mut seen[o], true)
            });
            (!compact).then_some(Class::ThreadLayoutNotCompact)
        }
        _ => None,
    }
}

/// The class of an MMA operation that is ill-posed whatever is asked of it.
fn mma_class(s: &str) -> Option<Class> {
    let (name, a) = spec(s);
    match name {
        "warpgroup.MmaF16BF16Op" if a[3] == "RMEM" && a[4] == "MN" => Some(Class::RegisterAMnMajor),
        "warpgroup.MmaF8Op" | "warpgroup.MmaI8Op" if a[4] == "RMEM" && a[5] == "MN" => Some(Class::RegisterAMnMajor),
        "tcgen05.MmaI8Op" => {
            let (_, n, _) = shape3(&a[1]);
            (!(n == 8 || n.is_multiple_of(16) && (16..=256).contains(&n))).then_some(Class::UmmaInt8N)
        }
        _ => None,
    }
}

/// Whether a tiled copy's threads hold a number of values that is not a multiple of the atom's.
fn too_few_values(copy: &str, tiled_vals: usize) -> Option<Class> {
    let (op, ty) = copy_op(copy);
    let atom_vals = with_type!(ty, T => atom_of::<T>(op).layout_ref_tv().mode(1).size());
    (!tiled_vals.is_multiple_of(atom_vals)).then_some(Class::CopyTooFewValues)
}

/// Whether the tensor-memory accumulator of an `M = 64`, one-CTA tcgen05 MMA is tiled over a
/// first rest mode (of extent above 1, in column-major order) of odd extent.
fn tmem_lane_gap(a: &[&str]) -> Option<Class> {
    let (name, args) = spec(a[0]);
    if !name.starts_with("tcgen05.") {
        return None;
    }
    let shape = args.iter().find(|x| x.starts_with('(')).map(|x| shape3(x))?;
    if shape.0 != 64 || args.iter().any(|x| x == "TWO") {
        return None;
    }
    let tm = tiled_mma(a[0], a[1], a[2]);
    let c = tm.get_slice(int(a[4]) as i64).partition_c(&counting(a[3]));
    let rest: Vec<usize> =
        (1..c.layout().rank()).flat_map(|i| c.layout().mode(i).flat_modes()).map(|(e, _)| e).collect();
    let first = rest.into_iter().find(|&e| e > 1)?;
    (first % 2 == 1).then_some(Class::TmemLaneGap)
}
