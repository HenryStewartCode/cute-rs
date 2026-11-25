//! TMA atom configurations (roadmap part m7-tma-atom), shared by the host tests
//! (`tests/tma_atom.rs`) and the GPU tests (`tests/gpu_tma.rs`, which has the driver encode
//! every descriptor), and the placement check: for every element of a shared-memory tile,
//! the descriptor's box-and-swizzle model must place the global element that the CuTe
//! layouts say belongs there.

#![allow(dead_code)]

use std::collections::BTreeMap;

use cute::cpasync::TmaInfo;
use cute::ir::ValueId;
use cute::prelude::*;

/// Where the global tensor starts in the model.
pub const GBASE: u64 = 0x1000_0000;
/// Where shared memory starts in the model: 1024-byte aligned, as 128-byte-swizzled
/// boxes need.
pub const SBASE: u64 = 0x4_0000;

/// Values of the kernel's scalar parameters for a launch.
#[derive(Clone, Debug, Default)]
pub struct Env(pub Vec<(ValueId, i64)>);

impl Env {
    pub fn with(mut self, v: &Int32, x: i64) -> Env {
        match IndexExpr::from(*v) {
            IndexExpr::Dynamic(id) => self.0.push((id, x)),
            e => panic!("{e:?} is not a runtime value"),
        }
        self
    }

    pub fn get(&self, v: ValueId) -> Option<i64> {
        self.0.iter().find(|(id, _)| *id == v).map(|(_, x)| *x)
    }

    pub fn eval(&self, e: &IndexExpr) -> i64 {
        e.eval_signed(&|v| self.get(v).unwrap_or_else(|| panic!("no value for %{}", v.0)))
    }
}

/// Run `f` while a kernel is being built (the atoms declare parameters) and return what
/// it returns.
pub fn in_kernel<R>(f: impl FnOnce() -> R) -> R {
    let mut out = None;
    Kernel::build("tma_atom_case", || out = Some(f()));
    out.unwrap()
}

/// The GMMA shared-memory layout atoms, by the bytes of their rows.
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Atom {
    SW128,
    SW64,
    SW32,
    INTER,
}

impl Atom {
    pub const ALL: [Atom; 4] = [Atom::SW128, Atom::SW64, Atom::SW32, Atom::INTER];

    /// Bytes of one row (the contiguous run): 128, 64, 32, or 16 unswizzled.
    pub fn row_bytes(self) -> usize {
        match self {
            Atom::SW128 => 128,
            Atom::SW64 => 64,
            Atom::SW32 => 32,
            Atom::INTER => 16,
        }
    }

    /// `B` of the byte swizzle `Swizzle<B,4,3>`.
    pub fn swizzle_bits(self) -> u32 {
        match self {
            Atom::SW128 => 3,
            Atom::SW64 => 2,
            Atom::SW32 => 1,
            Atom::INTER => 0,
        }
    }

    pub fn tma_swizzle(self) -> TensorMapSwizzle {
        match self {
            Atom::SW128 => TensorMapSwizzle::s128b,
            Atom::SW64 => TensorMapSwizzle::s64b,
            Atom::SW32 => TensorMapSwizzle::s32b,
            Atom::INTER => TensorMapSwizzle::none,
        }
    }

    pub fn kind(self, k_major: bool) -> cute::warpgroup::SmemLayoutAtomKind {
        use cute::warpgroup::SmemLayoutAtomKind::*;
        match (self, k_major) {
            (Atom::SW128, true) => K_SW128,
            (Atom::SW64, true) => K_SW64,
            (Atom::SW32, true) => K_SW32,
            (Atom::INTER, true) => K_INTER,
            (Atom::SW128, false) => MN_SW128,
            (Atom::SW64, false) => MN_SW64,
            (Atom::SW32, false) => MN_SW32,
            (Atom::INTER, false) => MN_INTER,
        }
    }
}

/// The byte swizzle `Swizzle<B,4,3>` of `atom` restated on elements of `elem_bytes` bytes,
/// the unit of cute-rs's composed layouts: `Swizzle<B,4-log2(elem_bytes),3>`.
pub fn element_swizzle(atom: Atom, elem_bytes: usize) -> Swizzle {
    if atom.swizzle_bits() == 0 {
        make_swizzle(0, 4, 3)
    } else {
        make_swizzle(atom.swizzle_bits(), 4 - elem_bytes.trailing_zeros(), 3)
    }
}

/// A `(bm, bk)` shared-memory tile built by hand from a GMMA atom of `elem_bytes`-byte
/// elements: K-major, `Sw o (8,W):(W,1)` tiled column-major; M-major, `Sw o (W,8):(1,W)`
/// tiled along K first (the order `make_smem_layout_a` uses), with `W` the atom row in
/// elements.
pub fn gmma_tile(atom: Atom, k_major: bool, elem_bytes: usize, bm: usize, bk: usize) -> ComposedLayout {
    let w = atom.row_bytes() / elem_bytes;
    let sw = element_swizzle(atom, elem_bytes);
    if k_major {
        tile_to_shape_swizzled(&make_composed_layout(sw, 0, Layout::new((8, w), (w, 1))), (bm, bk))
    } else {
        tile_to_shape_swizzled_ordered(&make_composed_layout(sw, 0, Layout::new((w, 8), (1, w))), (bm, bk), &[1, 0])
    }
}

/// The address map of a tile: shared-memory byte address of each element's first byte
/// -> global byte address.
pub type Placement = BTreeMap<u64, u64>;

fn coord_of(shape: &IntTuple, idx: usize) -> Coord {
    Layout::col_major(shape.clone()).idx2crd(idx).map(&mut |&x| IndexExpr::Static(x))
}

/// What the CuTe layouts say: the tile of `a` at `tile_coord` against the shared-memory
/// tile `smem`, element by element (coordinate `c` of the tile is at `smem(c)` and at
/// `a_tile(c)`).
pub fn layout_placement<T: Numeric>(
    a: &Tensor<T>,
    smem: &Tensor<T>,
    tiler: &IntTuple,
    tile_coord: impl Into<SliceCoord>,
    env: &Env,
) -> Placement {
    let eb = T::DTYPE.bytes() as u64;
    let g = local_tile(a, Tiler::from(tiler), tile_coord);
    let mut out = Placement::new();
    for idx in 0..tiler.size() {
        let c = coord_of(tiler, idx);
        let s = SBASE + env.eval(&smem.element_offset(&c)) as u64 * eb;
        let ga = GBASE + env.eval(&g.element_offset(&c)) as u64 * eb;
        assert!(out.insert(s, ga).is_none(), "two tile elements at smem {s:#x}");
    }
    out
}

/// What the descriptor does: each CTA of the multicast partitions the tiles with
/// [`tma_partition`], and each instruction's box, starting at the TMA coordinates its
/// coordinate view holds and at the shared-memory element its shared-memory view holds,
/// is traversed by the descriptor model ([`TmaDesc::box_elements`]) and swizzled
/// ([`TmaDesc::smem_address`]).
pub fn tma_placement<T: Numeric>(
    info: &TmaInfo<T>,
    smem: &Tensor<T>,
    tiler: &IntTuple,
    tile_coord: impl Into<SliceCoord>,
    env: &Env,
) -> Placement {
    let eb = T::DTYPE.bytes() as u64;
    let ti = info.atom.tma_info().unwrap();
    let desc = ti.spec().evaluate(GBASE, &|v| env.get(v)).unwrap_or_else(|e| panic!("{e}"));
    let t = local_tile(&info.tma_tensor, Tiler::from(tiler), tile_coord);
    let r = tiler.rank();
    let (sg, tg) = (smem.group_modes(0, r), t.group_modes(0, r));
    let mc = ti.num_multicast();
    let mut out = Placement::new();
    for cta in 0..mc {
        let (ts, tt) = cpasync::tma_partition(&info.atom, cta, &make_layout(mc, 1), &sg, &tg);
        assert_eq!(ts.rank(), 1, "{ts:?}");
        assert_eq!(tt.layout().mode(0).mode(0).size(), info.atom.num_val_src(), "{tt:?}");
        let iters = tt.layout().mode(0).mode(1).size();
        for i in 0..iters {
            let c: Coord = Tree::Branch(vec![Tree::Branch(vec![Tree::Leaf(0.into()), Tree::Leaf(i.into())])]);
            let start: Vec<i64> = tt.coord(c.clone()).leaves().iter().map(|e| env.eval(e)).collect();
            assert_eq!(start.len(), desc.rank as usize, "TMA coordinates {start:?} for a rank-{} map", desc.rank);
            // The instruction's destination is the box start's address before the swizzle
            // (C++ CuTe passes `raw_pointer_cast` of the swizzled smem pointer, which is the
            // unswizzled address): TMA swizzles by address itself. The swizzle is an
            // involution, so applying it to the physical offset gives the linear one.
            let phys = env.eval(&ts.element_offset(&c)) as usize;
            let linear = ts.swizzle().map_or(phys, |sw| sw.apply(phys));
            let s0 = SBASE + linear as u64 * eb;
            // TMA's shared-memory destination is 128-byte aligned.
            assert!(s0.is_multiple_of(128), "box {i} of CTA {cta} starts at smem {s0:#x}");
            for e in desc.box_elements(&start) {
                let s = desc.smem_address(s0, e.box_byte);
                let g = e.global_address.unwrap_or_else(|| panic!("box {i} of CTA {cta} leaves the tensor at {e:?}"));
                assert!(out.insert(s, g).is_none(), "two box elements land at smem {s:#x}");
            }
        }
    }
    out
}

/// Check the placement of the tile at `tile_coord` of `a` (made into `info`) in a
/// 1024-byte-aligned shared-memory tile over one stage of the atom's layout.
pub fn check_placement<T: Numeric>(
    info: &TmaInfo<T>,
    a: &Tensor<T>,
    tiler: &IntTuple,
    tile_coord: impl Into<SliceCoord> + Clone,
    env: &Env,
) {
    let smem = make_smem_tensor_aligned::<T>(info.atom.tma_info().unwrap().smem_layout().clone(), 1024);
    let want = layout_placement(a, &smem, tiler, tile_coord.clone(), env);
    let got = tma_placement(info, &smem, tiler, tile_coord, env);
    assert_eq!(got.len(), want.len(), "the boxes move {} elements, the tile has {}", got.len(), want.len());
    for (s, g) in &want {
        assert_eq!(got.get(s), Some(g), "smem {s:#x}: TMA places {:x?}, the layouts {g:#x}", got.get(s));
    }
}

/// A descriptor with the scalar values of its launch, for the driver to encode.
pub struct DescCase {
    pub name: String,
    pub spec: TmaDescSpec,
    pub env: Env,
}

/// The K- and M-major GMMA tiles of f16, bf16 and f32 (128x64 and 64x64, every atom) over
/// a `(m, k)` tensor with runtime extents, and their descriptors at `m = 512, k = 256`.
pub fn gmma_desc_cases() -> Vec<DescCase> {
    fn one<T: Numeric>(out: &mut Vec<DescCase>) {
        let eb = T::DTYPE.bytes();
        for atom in Atom::ALL {
            for k_major in [true, false] {
                for (bm, bk) in [(128, 64), (64, 64)] {
                    out.push(in_kernel(|| {
                        let m = param_extent("m", 8);
                        let k = param_extent("k", 8);
                        let layout = if k_major { Layout::new((m, k), (k, 1)) } else { Layout::new((m, k), (1, m)) };
                        let a = param_tensor_aligned::<T>("a", layout, 16);
                        let smem = gmma_tile(atom, k_major, eb, bm, bk);
                        let info = cpasync::make_tiled_tma_atom(
                            cpasync::CopyBulkTensorTileG2SOp::default(),
                            &a,
                            smem,
                            (bm, bk),
                            1,
                        );
                        DescCase {
                            name: format!(
                                "{} {atom:?} {}-major {bm}x{bk}",
                                T::DTYPE.cute_name(),
                                if k_major { "K" } else { "M" }
                            ),
                            spec: info.atom.tma_info().unwrap().spec().clone(),
                            env: Env::default().with(&m, 512).with(&k, 256),
                        }
                    }));
                }
            }
        }
    }
    let mut out = Vec::new();
    one::<F16>(&mut out);
    one::<BF16>(&mut out);
    one::<f32>(&mut out);
    out
}
