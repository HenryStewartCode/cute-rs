// SPDX-License-Identifier: Apache-2.0

//! Block-sparse attention, after FlashInfer's `BlockSparseAttentionWrapper` and
//! `VariableBlockSparseAttentionWrapper` (`flashinfer/sparse.py`, their `fa2` path). This is
//! cute-rs's own code, written after FlashInfer's published design; no FlashInfer source
//! was translated.
//!
//! FlashInfer runs block-sparse attention as a paged prefill: the sparse structure becomes
//! a page table, and the [`prefill`](crate::prefill) kernel's CTA body (`prefill::body_hd`)
//! runs it unchanged.
//!
//! * **BSR** ([`Problem::from_bsr`], FlashInfer's `BlockSparseAttentionWrapper`): an `M x
//!   N` attention mask in blocks of `R x C`, as BSR `indptr [M / R + 1]` / `indices
//!   [nnz]` (block columns). Each block row is one prefill request of `R` queries, the KV
//!   cache `[N, num_kv_heads, head_dim]` is read as `N / C` pages of `C` tokens (NHD), and
//!   block row `i`'s page list is its block columns `indices[indptr[i] .. indptr[i + 1]]`,
//!   every page full. GQA packs `R * group` rows per request, as the prefill kernel does.
//!   An optional custom mask `[nnz, R, C]` refines the blocks: as FlashInfer's
//!   `convert_bsr_mask_layout`, request `i`'s bits are its `R x (nnz_i * C)` matrix, packed
//!   little-endian with `segment_packbits`; the kernel is then built with the custom-mask
//!   variant (every tile masked).
//! * **Variable blocks** ([`Problem::from_variable_blocks`], FlashInfer's
//!   `VariableBlockSparseAttentionWrapper`): per KV head, block-row sizes `[num_kv_heads,
//!   MB]`, block-column sizes `[num_kv_heads, NB]` and a block mask `[num_kv_heads, MB,
//!   NB]`. As FlashInfer, the KV heads become requests: request `(h, i)` is block row `i`
//!   of head `h`, the kernel sees one KV head with `group` query heads, Q is rearranged
//!   from `[num_qo_heads, qo_len, D]` to `[(num_kv_heads qo_len), group, D]`, K and V
//!   `[num_kv_heads, kv_len, D]` are read as one-token pages, and request `(h, i)`'s page
//!   list is the tokens of its kept block columns. Per-head block sizes and patterns are
//!   both covered.
//!
//! The plan is the batch prefill planner's ([`prefill::Plan`]), with page size `C` (BSR)
//! or 1 (variable blocks): FlashInfer's `PrefillSplitQOKVIndptr`, so query tiles of 128,
//! 64 or 16 rows from the average packed request length, and split-KV with a merge when
//! the tiles leave the GPU idle. The merge is this module's [`merge_kernel`] (the decode
//! port's, which also writes FlashInfer's `o = 0`, `lse = -inf` for a row with no visible
//! token).
//!
//! Covered: Float16 Q/K/V/O, head dim 64 or 128 (64 through `prefill::body_hd`, with
//! FlashInfer's shared-memory tile rule for 64), any `R`, any page size `C`, GQA, causal
//! off (FlashInfer's default for sparse attention), an optional custom mask on BSR blocks,
//! block rows with no block (output 0, `lse = -inf`).
//!
//! Not covered: FlashInfer's CUDA-core decode path for `R * group < 4` without a mask
//! (cute-rs's decode kernel is built for group 4, so these run on the prefill kernel's
//! 16-row tensor-core tile); causal masking, soft-cap, RoPE and FP8; the VSA / cuTile /
//! Cake backends; a query row whose custom mask hides every token of a non-empty block row
//! (the prefill kernel divides by a zero sum there, where FlashInfer writes 0).
//!
//! On an RTX 2000 Ada (CUDA-graph replays, the shapes of `sparse_flashinfer_bench.py`):
//! BSR 64 x 64 blocks, 32 / 8 heads, M = N = 4K-16K, 50% and 90% sparse: 34.9-39.0 TFLOPS,
//! within 2% of FlashInfer (head dim 128: 3558 vs 3527 us at 4K/0.5, 55984 vs 56457 us at
//! 16K/0.5; head dim 64 at 8K 6-7% faster); with a custom mask at 4K 4303 vs 6292 us.
//! Variable blocks (FlashInfer's 20 x 50, 32 / 32 heads): within 3% at 16K and at 8K/0.5,
//! 9% slower at 8K/0.1 (3615 vs 3305 us: 640 short requests of ~410 queries over ~820
//! one-token pages; the kernel's instruction mix is the paged prefill's).
//!
//! `examples/sparse.rs` runs it on the simulator and, with `--features cuda` and `GPU=1`,
//! checks it on the GPU and benchmarks it against `examples/sparse_flashinfer_bench.py`.

use cute::prelude::*;

use crate::prefill::{self, CtaCoords, KernelSpec, Params, Plan, TileConfig, Variant};

/// The per-request page tables and inputs the prefill kernel runs: block rows (BSR) or
/// (KV head, block row) pairs (variable blocks) as requests.
///
/// `kv_indices` has one padding entry at each end (`kv_indptr` counts from 1), so the
/// kernel's look-ahead page reads of an empty request stay in bounds.
#[derive(Clone, Debug)]
pub struct Problem {
    pub head_dim: usize,
    /// KV heads the kernel sees (1 for variable blocks).
    pub num_kv_heads: usize,
    pub group: usize,
    /// Tokens per page: `C` (BSR) or 1 (variable blocks).
    pub page_size: usize,
    pub qo_lens: Vec<usize>,
    /// KV tokens per request.
    pub kv_lens: Vec<usize>,
    pub qo_indptr: Vec<i32>,
    pub kv_indptr: Vec<i32>,
    pub kv_indices: Vec<i32>,
    pub kv_last_page_len: Vec<i32>,
    /// `q [total_q, num_kv_heads * group, head_dim]`, `k`, `v [pages * page_size,
    /// num_kv_heads, head_dim]`.
    pub q: Vec<F16>,
    pub k: Vec<F16>,
    pub v: Vec<F16>,
    pub sm_scale: f32,
    /// The custom mask (empty for none): request `b`'s `qo_len * kv_len` bits from
    /// `qk_indptr[b]`, packed from byte `mask_indptr[b]` of `packed_mask`.
    pub mask_bits: Vec<bool>,
    pub qk_indptr: Vec<i32>,
    pub packed_mask: Vec<u8>,
    pub mask_indptr: Vec<i32>,
    /// For variable blocks, `(num_kv_heads, qo_len)` of FlashInfer's HND layout.
    pub hnd: Option<(usize, usize)>,
}

/// FlashInfer's block-sparse pattern hash, shared with `sparse_flashinfer_bench.py`: block
/// `(h, i, j)` is kept when the hash, mod 1000, is below `density * 1000`.
pub fn keep_block(h: usize, i: usize, j: usize, density: f64) -> bool {
    let mut x = ((h as u64 * 1_000_003 + i as u64) * 1_000_003 + j as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    x % 1000 < (density * 1000.0).round() as u64
}

fn indptr(lens: impl IntoIterator<Item = usize>) -> Vec<i32> {
    let mut v = vec![0i32];
    for l in lens {
        v.push(v.last().unwrap() + l as i32);
    }
    v
}

/// A pseudo-random generator for inputs (large inputs from a repeating block).
fn filler(seed: u64) -> impl FnMut(usize) -> Vec<F16> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    move |n: usize| {
        let mut rnd = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            F16::from_f32(((state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0)
        };
        if n <= 1 << 22 {
            (0..n).map(|_| rnd()).collect()
        } else {
            let block: Vec<F16> = (0..65537).map(|_| rnd()).collect();
            (0..n).map(|i| block[i % block.len()]).collect()
        }
    }
}

impl Problem {
    /// FlashInfer's `BlockSparseAttentionWrapper.plan(indptr, indices, M, N, R, C,
    /// num_qo_heads, num_kv_heads, head_dim, mask)` and `run(q, k, v)`: `q [M, num_qo_heads,
    /// head_dim]`, `k`, `v [N, num_kv_heads, head_dim]` (NHD), and an optional mask `[nnz, R,
    /// C]` (row-major bools); scale `1 / sqrt(head_dim)`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_bsr(
        (m, n, r, c): (usize, usize, usize, usize),
        bsr_indptr: &[i32],
        bsr_indices: &[i32],
        num_qo_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        (q, k, v): (Vec<F16>, Vec<F16>, Vec<F16>),
        mask: Option<&[bool]>,
    ) -> Problem {
        assert!(head_dim == 64 || head_dim == 128, "head dim 64 or 128");
        assert_eq!(num_qo_heads % num_kv_heads, 0);
        assert!(m % r == 0 && n % c == 0, "M and N must be multiples of R and C");
        let mb = m / r;
        assert_eq!(bsr_indptr.len(), mb + 1);
        let nnz = *bsr_indptr.last().unwrap() as usize;
        assert_eq!(bsr_indices.len(), nnz);
        assert!(bsr_indices.iter().all(|&j| (j as usize) < n / c), "indices out of bound");
        assert_eq!(q.len(), m * num_qo_heads * head_dim);
        assert_eq!(k.len(), n * num_kv_heads * head_dim);
        assert_eq!(v.len(), k.len());
        let blocks: Vec<usize> = (0..mb).map(|i| (bsr_indptr[i + 1] - bsr_indptr[i]) as usize).collect();
        let mut p = Problem {
            head_dim,
            num_kv_heads,
            group: num_qo_heads / num_kv_heads,
            page_size: c,
            qo_lens: vec![r; mb],
            kv_lens: blocks.iter().map(|b| b * c).collect(),
            qo_indptr: indptr(vec![r; mb]),
            kv_indptr: bsr_indptr.iter().map(|x| x + 1).collect(),
            kv_indices: std::iter::once(0).chain(bsr_indices.iter().copied()).chain(std::iter::once(0)).collect(),
            kv_last_page_len: vec![c as i32; mb],
            q,
            k,
            v,
            sm_scale: 1.0 / (head_dim as f32).sqrt(),
            mask_bits: vec![],
            qk_indptr: vec![0],
            packed_mask: vec![],
            mask_indptr: vec![0],
            hnd: None,
        };
        if let Some(mask) = mask {
            assert_eq!(mask.len(), nnz * r * c, "mask is [nnz, R, C]");
            // convert_bsr_mask_layout: request i's blocks (nnz_i, R, C) as (R, nnz_i, C).
            let mut bits = Vec::with_capacity(mask.len());
            for i in 0..mb {
                let (b0, nb) = (bsr_indptr[i] as usize, blocks[i]);
                for row in 0..r {
                    for blk in 0..nb {
                        let at = ((b0 + blk) * r + row) * c;
                        bits.extend_from_slice(&mask[at..at + c]);
                    }
                }
            }
            p = p.with_mask_bits(bits);
        }
        p
    }

    /// FlashInfer's `VariableBlockSparseAttentionWrapper.plan(block_mask_map, block_row_sz,
    /// block_col_sz, num_qo_heads, num_kv_heads, head_dim)` and `run(q, k, v)`:
    /// `block_mask_map [num_kv_heads, MB, NB]`, `block_row_sz [num_kv_heads, MB]`,
    /// `block_col_sz [num_kv_heads, NB]` (each head's sizes summing to `qo_len` and
    /// `kv_len`), `q [num_qo_heads, qo_len, head_dim]`, `k`, `v [num_kv_heads, kv_len,
    /// head_dim]` (HND).
    #[allow(clippy::too_many_arguments)]
    pub fn from_variable_blocks(
        block_mask_map: &[bool],
        block_row_sz: &[usize],
        block_col_sz: &[usize],
        num_qo_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        (q, k, v): (Vec<F16>, Vec<F16>, Vec<F16>),
    ) -> Problem {
        assert!(head_dim == 64 || head_dim == 128, "head dim 64 or 128");
        assert_eq!(num_qo_heads % num_kv_heads, 0);
        let (mb, nb) = (block_row_sz.len() / num_kv_heads, block_col_sz.len() / num_kv_heads);
        assert_eq!(block_mask_map.len(), num_kv_heads * mb * nb);
        let qo_len: usize = block_row_sz[..mb].iter().sum();
        let kv_len: usize = block_col_sz[..nb].iter().sum();
        for h in 0..num_kv_heads {
            assert_eq!(block_row_sz[h * mb..(h + 1) * mb].iter().sum::<usize>(), qo_len, "head {h}'s rows");
            assert_eq!(block_col_sz[h * nb..(h + 1) * nb].iter().sum::<usize>(), kv_len, "head {h}'s columns");
        }
        let g = num_qo_heads / num_kv_heads;
        assert_eq!(q.len(), num_qo_heads * qo_len * head_dim);
        assert_eq!(k.len(), num_kv_heads * kv_len * head_dim);
        assert_eq!(v.len(), k.len());
        // "(num_kv_heads g) qo_len d -> (num_kv_heads qo_len) g d"
        let mut qk = vec![F16::from_f32(0.0); q.len()];
        for h in 0..num_kv_heads {
            for gi in 0..g {
                for t in 0..qo_len {
                    let src = ((h * g + gi) * qo_len + t) * head_dim;
                    let dst = ((h * qo_len + t) * g + gi) * head_dim;
                    qk[dst..dst + head_dim].copy_from_slice(&q[src..src + head_dim]);
                }
            }
        }
        let (mut qo_lens, mut kv_lens, mut kv_indices) = (vec![], vec![], vec![0i32]);
        for h in 0..num_kv_heads {
            let cols = &block_col_sz[h * nb..(h + 1) * nb];
            let col_off: Vec<usize> = indptr(cols.iter().copied()).iter().map(|&x| x as usize).collect();
            for i in 0..mb {
                qo_lens.push(block_row_sz[h * mb + i]);
                let mut len = 0;
                for j in 0..nb {
                    if block_mask_map[(h * mb + i) * nb + j] {
                        let base = h * kv_len + col_off[j];
                        kv_indices.extend((base..base + cols[j]).map(|t| t as i32));
                        len += cols[j];
                    }
                }
                kv_lens.push(len);
            }
        }
        kv_indices.push(0);
        let batch = qo_lens.len();
        Problem {
            head_dim,
            num_kv_heads: 1,
            group: g,
            page_size: 1,
            qo_indptr: indptr(qo_lens.iter().copied()),
            kv_indptr: indptr(kv_lens.iter().copied()).iter().map(|x| x + 1).collect(),
            qo_lens,
            kv_lens,
            kv_indices,
            kv_last_page_len: vec![1; batch],
            q: qk,
            k,
            v,
            sm_scale: 1.0 / (head_dim as f32).sqrt(),
            mask_bits: vec![],
            qk_indptr: vec![0],
            packed_mask: vec![],
            mask_indptr: vec![0],
            hnd: Some((num_kv_heads, qo_len)),
        }
    }

    /// A random BSR problem: `M x N` in `R x C` blocks, block `(i, j)` kept by
    /// [`keep_block`]`(0, i, j, density)`, random inputs from `seed`, and with `mask` a
    /// random custom mask (each bit with probability one half; every query row keeps one
    /// token of its block row's first block).
    #[allow(clippy::too_many_arguments)]
    pub fn random_bsr(
        dims: (usize, usize, usize, usize),
        density: f64,
        num_qo_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        mask: bool,
        seed: u64,
    ) -> Problem {
        let (m, n, r, c) = dims;
        let (mut ind, mut idx) = (vec![0i32], vec![]);
        for i in 0..m / r {
            idx.extend((0..n / c).filter(|&j| keep_block(0, i, j, density)).map(|j| j as i32));
            ind.push(idx.len() as i32);
        }
        let mut fill = filler(seed);
        let q = fill(m * num_qo_heads * head_dim);
        let k = fill(n * num_kv_heads * head_dim);
        let v = fill(n * num_kv_heads * head_dim);
        let bits = mask.then(|| {
            let mut st = seed ^ 0x5DEE_CE66;
            let mut bits = vec![false; idx.len() * r * c];
            let mut first = vec![false; idx.len()];
            for w in ind.windows(2).filter(|w| w[1] > w[0]) {
                first[w[0] as usize] = true;
            }
            for (b, x) in bits.iter_mut().enumerate() {
                st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *x = st >> 40 & 1 == 1;
                // Block-row i's first block keeps token row % C of each row.
                let (blk, row, col) = (b / (r * c), (b / c) % r, b % c);
                if first[blk] && col == row % c {
                    *x = true;
                }
            }
            bits
        });
        Problem::from_bsr(dims, &ind, &idx, num_qo_heads, num_kv_heads, head_dim, (q, k, v), bits.as_deref())
    }

    /// A random variable-block problem like FlashInfer's `bench_block_sparse_attention.py`:
    /// `seq_len` queries and tokens split into `MB` rows and `NB` columns (equal sizes, the
    /// last taking the rest; with `jitter`, each head's sizes shifted by one block-size
    /// pattern of its own), block `(h, i, j)` kept by [`keep_block`].
    #[allow(clippy::too_many_arguments)]
    pub fn random_variable(
        seq_len: usize,
        (mb, nb): (usize, usize),
        density: f64,
        num_qo_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        jitter: bool,
        seed: u64,
    ) -> Problem {
        let sizes = |blocks: usize, h: usize| -> Vec<usize> {
            let base = seq_len / blocks;
            let mut s = vec![base; blocks];
            if jitter && blocks > 1 {
                // Head h moves (h + 1) * 3 rows between neighbouring blocks.
                for i in (0..blocks - 1).step_by(2) {
                    let d = ((h + 1) * 3 + i) % base.max(1);
                    s[i] -= d;
                    s[i + 1] += d;
                }
            }
            s[blocks - 1] = seq_len - s[..blocks - 1].iter().sum::<usize>();
            s
        };
        let rows: Vec<usize> = (0..num_kv_heads).flat_map(|h| sizes(mb, h)).collect();
        let cols: Vec<usize> = (0..num_kv_heads).flat_map(|h| sizes(nb, h + 7)).collect();
        let map: Vec<bool> =
            (0..num_kv_heads * mb * nb).map(|x| keep_block(x / (mb * nb), (x / nb) % mb, x % nb, density)).collect();
        let mut fill = filler(seed);
        let q = fill(num_qo_heads * seq_len * head_dim);
        let k = fill(num_kv_heads * seq_len * head_dim);
        let v = fill(num_kv_heads * seq_len * head_dim);
        Problem::from_variable_blocks(&map, &rows, &cols, num_qo_heads, num_kv_heads, head_dim, (q, k, v))
    }

    /// This problem with the custom mask `bits` (request by request, `qo_len * kv_len` bits
    /// each, row-major), packed with FlashInfer's `segment_packbits` (little bit order).
    pub fn with_mask_bits(mut self, bits: Vec<bool>) -> Problem {
        let qk_indptr = indptr(self.qo_lens.iter().zip(&self.kv_lens).map(|(q, k)| q * k));
        assert_eq!(bits.len(), *qk_indptr.last().unwrap() as usize, "one bit per (query, token)");
        let packer = crate::quantization::Problem { bits, indptr: qk_indptr.clone() };
        self.packed_mask = packer.reference(crate::quantization::Order::Little);
        self.mask_indptr = packer.out_indptr();
        (self.mask_bits, self.qk_indptr) = (packer.bits, qk_indptr);
        self
    }

    pub fn has_mask(&self) -> bool {
        !self.mask_bits.is_empty()
    }

    pub fn num_qo_heads(&self) -> usize {
        self.num_kv_heads * self.group
    }

    pub fn total_q(&self) -> usize {
        *self.qo_indptr.last().unwrap() as usize
    }

    /// Pages per request.
    pub fn pages(&self) -> Vec<usize> {
        self.kv_indptr.windows(2).map(|w| (w[1] - w[0]) as usize).collect()
    }

    /// FlashInfer's plan for `max_grid` concurrent CTAs (the batch prefill planner, smallest
    /// chunk `128 / page_size` pages), with the tile shape of this head dim.
    pub fn plan(&self, max_grid: usize) -> Plan {
        self.plan_with_min_chunk(max_grid, (128 / self.page_size).max(1))
    }

    /// [`Problem::plan`] with another smallest chunk, in pages (tests use 1).
    pub fn plan_with_min_chunk(&self, max_grid: usize, min_chunk_pages: usize) -> Plan {
        let mut plan = Plan::with_min_chunk(
            &self.qo_lens,
            &self.pages(),
            self.page_size,
            self.group,
            self.num_kv_heads,
            max_grid,
            min_chunk_pages,
        );
        plan.tile = tile(plan.tile.cta_tile_q, self.head_dim);
        if plan.split && self.kv_lens.contains(&0) {
            // An empty block row would write its (lse = -inf) state to one scratch row per
            // query that the planner never counted; with split-KV it is left out instead,
            // and the merge writes its rows' o = 0, lse = -inf.
            let keep: Vec<usize> =
                (0..plan.num_ctas()).filter(|&i| self.kv_lens[plan.request_indices[i] as usize] > 0).collect();
            let pick = |v: &[i32]| keep.iter().map(|&i| v[i]).collect::<Vec<i32>>();
            plan.request_indices = pick(&plan.request_indices);
            plan.qo_tile_indices = pick(&plan.qo_tile_indices);
            plan.kv_tile_indices = pick(&plan.kv_tile_indices);
            let chunk_pages = plan.kv_chunk_size / self.page_size;
            let (mut merge_indptr, mut o_indptr) = (vec![0i32], vec![0i32]);
            for (b, pages) in self.pages().into_iter().enumerate() {
                let chunks = if self.kv_lens[b] == 0 { 0 } else { pages.div_ceil(chunk_pages) };
                for _ in 0..self.qo_lens[b] {
                    merge_indptr.push(merge_indptr.last().unwrap() + chunks as i32);
                }
                o_indptr.push(o_indptr[b] + (self.qo_lens[b] * chunks) as i32);
            }
            (plan.merge_indptr, plan.o_indptr) = (merge_indptr, o_indptr);
        }
        plan
    }

    pub fn spec(&self, plan: &Plan) -> Spec {
        let variant = Variant { custom_mask: self.has_mask(), ..Variant::NONE };
        Spec {
            prefill: KernelSpec {
                tile: plan.tile,
                group: self.group,
                page_size: self.page_size,
                causal: false,
                partition: plan.split,
                variant,
            },
            head_dim: self.head_dim,
        }
    }

    pub fn launch(&self, plan: &Plan) -> LaunchConfig {
        LaunchConfig::new([plan.num_ctas() as u32, self.num_kv_heads as u32, 1], plan.tile.block())
    }

    /// The merge's launch: one CTA of `head_dim` threads per (query token, query head).
    pub fn merge_launch(&self) -> LaunchConfig {
        LaunchConfig::new([self.total_q() as u32, self.num_qo_heads() as u32, 1], [self.head_dim as u32, 1, 1])
    }

    /// The kernel's arguments, in the order of [`prefill::p`].
    pub fn args(&self, plan: &Plan) -> Vec<KernelArg> {
        let (n_o, hd, rows) = (self.q.len(), self.head_dim, plan.tmp_rows());
        let variant = Variant { custom_mask: self.has_mask(), ..Variant::NONE };
        let (sm_log2, pre) = variant.scales(self.sm_scale, 0.0);
        vec![
            KernelArg::buffer(&self.q),
            KernelArg::buffer(&self.k),
            KernelArg::buffer(&self.v),
            KernelArg::i32(self.qo_indptr.clone()),
            KernelArg::i32(self.kv_indptr.clone()),
            KernelArg::i32(self.kv_indices.clone()),
            KernelArg::i32(self.kv_last_page_len.clone()),
            KernelArg::i32(plan.request_indices.clone()),
            KernelArg::i32(plan.qo_tile_indices.clone()),
            KernelArg::i32(plan.kv_tile_indices.clone()),
            KernelArg::i32(plan.o_indptr.clone()),
            KernelArg::buffer(&vec![F16::from_f32(0.0); n_o]),
            KernelArg::f32(vec![0.0; n_o / hd]),
            KernelArg::f32(vec![0.0; rows * self.num_qo_heads() * hd]),
            KernelArg::f32(vec![0.0; rows * self.num_qo_heads()]),
            KernelArg::i32(plan.merge_indptr.clone()),
            KernelArg::ScalarI32(self.num_kv_heads as i32),
            KernelArg::ScalarF32(sm_log2),
            KernelArg::ScalarI32(plan.kv_chunk_size as i32),
            KernelArg::ScalarF32(pre),
            KernelArg::ScalarI32(-1),
            KernelArg::f32(vec![0.0]),
            KernelArg::Bytes {
                elem: DType::U8,
                data: if self.packed_mask.is_empty() { vec![0] } else { self.packed_mask.clone() },
            },
            KernelArg::i32(self.mask_indptr.clone()),
            KernelArg::ScalarF32(1.0),
        ]
    }

    /// Attention in f64 on the f16 inputs: `o` and base-2 `lse` in the kernel's layout (a
    /// request with no visible token: `o = 0`, `lse = -inf`).
    pub fn reference(&self) -> (Vec<f32>, Vec<f32>) {
        let (hkv, n_qo, ps, hd) = (self.num_kv_heads, self.num_qo_heads(), self.page_size, self.head_dim);
        let f = |x: F16| x.to_f32() as f64;
        let mut o = vec![0f32; self.q.len()];
        let mut lse = vec![f32::NEG_INFINITY; self.q.len() / hd];
        for (b, (&qo_len, &kv_len)) in self.qo_lens.iter().zip(&self.kv_lens).enumerate() {
            let start = self.kv_indptr[b] as usize;
            let rows: Vec<usize> =
                (0..kv_len).map(|t| self.kv_indices[start + t / ps] as usize * ps + t % ps).collect();
            for i in 0..qo_len {
                let token = self.qo_indptr[b] as usize + i;
                let visible: Vec<usize> = (0..kv_len)
                    .filter(|&t| !self.has_mask() || self.mask_bits[self.qk_indptr[b] as usize + i * kv_len + t])
                    .collect();
                if visible.is_empty() {
                    continue;
                }
                for qh in 0..n_qo {
                    let h = qh / self.group;
                    let qrow = (token * n_qo + qh) * hd;
                    let s: Vec<f64> = visible
                        .iter()
                        .map(|&t| {
                            let a = (rows[t] * hkv + h) * hd;
                            (0..hd).map(|e| f(self.q[qrow + e]) * f(self.k[a + e])).sum::<f64>() * self.sm_scale as f64
                        })
                        .collect();
                    let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let w: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
                    let sum: f64 = w.iter().sum();
                    lse[token * n_qo + qh] = ((mx + sum.ln()) * std::f64::consts::LOG2_E) as f32;
                    for e in 0..hd {
                        let acc: f64 =
                            visible.iter().zip(&w).map(|(&t, wt)| wt * f(self.v[(rows[t] * hkv + h) * hd + e])).sum();
                        o[qrow + e] = (acc / sum) as f32;
                    }
                }
            }
        }
        (o, lse)
    }

    /// `o` (kernel layout) in FlashInfer's output layout: unchanged for BSR (`[M,
    /// num_qo_heads, D]`); `[num_qo_heads, qo_len, D]` for variable blocks.
    pub fn output_layout(&self, o: &[f32]) -> Vec<f32> {
        let Some((hkv, qo_len)) = self.hnd else { return o.to_vec() };
        let (g, hd) = (self.group, self.head_dim);
        let mut out = vec![0f32; o.len()];
        for h in 0..hkv {
            for gi in 0..g {
                for t in 0..qo_len {
                    let src = ((h * qo_len + t) * g + gi) * hd;
                    let dst = ((h * g + gi) * qo_len + t) * hd;
                    out[dst..dst + hd].copy_from_slice(&o[src..src + hd]);
                }
            }
        }
        out
    }

    /// Floating-point operations of the kept (query, token) pairs: `4 * head_dim` per pair
    /// and query head, as `attention_tflops_per_sec_with_actual_seq_lens` counts them.
    pub fn flops(&self) -> f64 {
        let pairs: usize = self.qo_lens.iter().zip(&self.kv_lens).map(|(q, k)| q * k).sum();
        4.0 * pairs as f64 * self.num_qo_heads() as f64 * self.head_dim as f64
    }
}

/// The tile shape for query tile `cta_tile_q` and `head_dim` on an SM89 (Ada) GPU (100 KB
/// of shared memory per SM, 99 KB per block): FlashInfer's rule, which for head dim 64
/// gives KV tiles of 64, 128 and 128 tokens for query tiles of 128, 64 and 16.
pub fn tile(cta_tile_q: usize, head_dim: usize) -> TileConfig {
    TileConfig::new_kv_hd(cta_tile_q, 102400, 101376, prefill::KvDtype::F16, head_dim)
}

/// What a block-sparse kernel is built for: the prefill kernel's spec and the head dim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Spec {
    pub prefill: KernelSpec,
    pub head_dim: usize,
}

impl Spec {
    pub fn name(&self) -> String {
        let base = self.prefill.name();
        format!("flashinfer_block_sparse_d{}_{}", self.head_dim, base.trim_start_matches("flashinfer_batch_prefill_"))
    }

    /// Shared memory per CTA, in bytes.
    pub fn smem_bytes(&self) -> usize {
        2 * self.prefill.tile.smem_elems_hd(self.head_dim)
    }
}

/// The block-sparse kernel: the prefill kernel's CTA body for `spec.head_dim`, grid
/// `(plan CTAs, num_kv_heads)`, block `(32, NUM_WARPS_Q, NUM_WARPS_KV)`; parameters in the
/// order of [`prefill::p`].
pub fn kernel(spec: &Spec) -> Kernel {
    let (ps, hd) = (spec.prefill, spec.head_dim);
    assert!(!ps.causal && ps.page_size != prefill::RAGGED, "a paged, non-causal kernel");
    Kernel::builder(spec.name()).block_dim(ps.tile.block()).build(|| {
        let prm = Params::declare::<F16>("");
        let at = CtaCoords {
            lane: thread_idx().x,
            wq: thread_idx().y,
            wkv: thread_idx().z,
            kv_head: block_idx().y,
            cta: block_idx().x,
        };
        let smem = make_smem_tensor_aligned::<F16>(Layout::col_major(ps.tile.smem_elems_hd(hd)), 16);
        prefill::body_hd::<F16>(&ps, hd, &prm, &smem, at);
    })
}

/// The merge of split-KV states for `head_dim` (FlashInfer's `VariableLengthMergeStates`):
/// grid `(total_q, num_qo_heads)`, `head_dim` threads, parameters `tmp_o`, `tmp_lse`,
/// `merge_indptr`, `o`, `lse`, `num_qo_heads`. The decode port's merge, except that a row
/// with no state or only empty ones (`lse = -inf`) gets FlashInfer's `o = 0`, `lse = -inf`.
pub fn merge_kernel(head_dim: usize) -> Kernel {
    const MERGE_UNROLL: usize = 8;
    Kernel::builder(format!("flashinfer_block_sparse_merge_d{head_dim}")).block_dim([head_dim as u32, 1, 1]).build(
        || {
            let tmp_o = param_ptr::<f32>("tmp_o");
            let tmp_lse = param_ptr::<f32>("tmp_lse");
            let o_indptr = param_ptr::<i32>("o_indptr");
            let o = param_ptr::<F16>("o");
            let lse = param_ptr::<f32>("lse");
            let num_qo_heads = param_scalar::<i32>("num_qo_heads");
            let (e, b, h) = (thread_idx().x, block_idx().x, block_idx().y);
            let hd = head_dim as i32;
            let m = Var::new(Float32::from(-1.0e30f32));
            let d = Var::new(Float32::from(0.0f32));
            let acc = Var::new(Float32::from(0.0f32));
            let (begin, end) = (o_indptr.load_at(b), o_indptr.load_at(b + 1));
            let steps = (end - begin + (MERGE_UNROLL as i32 - 1)) / MERGE_UNROLL as i32;
            for_range(0, steps, 1, |step| {
                let rows: Vec<Int32> =
                    (0..MERGE_UNROLL).map(|u| begin + step * MERGE_UNROLL as i32 + u as i32).collect();
                let loads: Vec<(Float32, Float32)> = rows
                    .iter()
                    .map(|r| {
                        let rc = r.min(end - 1);
                        (tmp_lse.load_at(rc * num_qo_heads + h), tmp_o.load_at((rc * num_qo_heads + h) * hd + e))
                    })
                    .collect();
                let ss: Vec<Float32> = rows
                    .iter()
                    .zip(&loads)
                    .map(|(r, (s, _))| select(r.lt(end), *s, Float32::from(f32::NEG_INFINITY)))
                    .collect();
                let m_new = ss.iter().fold(m.get(), |x, y| fmax(x, *y));
                let correction = (m.get() - m_new).exp2();
                let (mut dn, mut an) = (d.get() * correction, acc.get() * correction);
                for (s, (_, v)) in ss.iter().zip(&loads) {
                    let w = (*s - m_new).exp2();
                    dn = dn + w;
                    an = an + w * *v;
                }
                d.set(dn);
                acc.set(an);
                m.set(m_new);
            });
            let some = d.get().gt(Float32::from(0.0f32));
            let out = select(some, acc.get() / d.get(), Float32::from(0.0f32));
            o.store_at((b * num_qo_heads + h) * hd + e, out.to::<F16>());
            if_(e.eq(0), || {
                let l = select(some, m.get() + d.get().log2(), Float32::from(f32::NEG_INFINITY));
                lse.store_at(b * num_qo_heads + h, l);
            });
        },
    )
}

/// The largest error of `a` against `b`, with equal infinities agreeing and NaN an
/// infinite error.
pub fn max_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            if x == y {
                0.0
            } else if x.is_nan() || y.is_nan() {
                f32::INFINITY
            } else {
                (x - y).abs()
            }
        })
        .fold(0.0, f32::max)
}

/// The merge's arguments, from the kernel's.
pub fn merge_args(p: &Problem, args: &[KernelArg]) -> Vec<KernelArg> {
    use prefill::p;
    vec![
        args[p::TMP_O].clone(),
        args[p::TMP_LSE].clone(),
        args[p::MERGE_INDPTR].clone(),
        args[p::O].clone(),
        args[p::LSE].clone(),
        KernelArg::ScalarI32(p.num_qo_heads() as i32),
    ]
}

/// Run `p` under `plan` on the simulator (the kernel, then the merge if the plan splits);
/// returns `o` (as f32, kernel layout) and `lse`.
pub fn execute(p: &Problem, plan: &Plan) -> (Vec<f32>, Vec<f32>) {
    use prefill::p as pp;
    let mut args = p.args(plan);
    let k = kernel(&p.spec(plan));
    k.simulate(p.launch(plan), &mut args).unwrap_or_else(|e| panic!("{}: {e}", k.name()));
    if plan.split {
        let mut margs = merge_args(p, &args);
        merge_kernel(p.head_dim).simulate(p.merge_launch(), &mut margs).unwrap_or_else(|e| panic!("merge: {e}"));
        args[pp::O] = margs[3].clone();
        args[pp::LSE] = margs[4].clone();
    }
    let o: Vec<f32> = args[pp::O].to_vec::<F16>().into_iter().map(|x| x.to_f32()).collect();
    (o, args[pp::LSE].as_f32().to_vec())
}

/// Simulate `p` for `max_grid` concurrent CTAs (smallest chunk `min_chunk_pages`) and
/// check it: the plan and `(max |o - ref|, max |lse - ref|)`.
pub fn simulate(p: &Problem, max_grid: usize, min_chunk_pages: usize) -> (Plan, f32, f32) {
    let plan = p.plan_with_min_chunk(max_grid, min_chunk_pages);
    let (o, lse) = execute(p, &plan);
    let (ro, rl) = p.reference();
    (plan, max_error(&o, &ro), max_error(&lse, &rl))
}

/// Running on the GPU (`--features cuda`).
#[cfg(feature = "cuda")]
pub mod gpu {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::error::Error;
    use std::rc::Rc;

    use cute::cuda::{CudaContext, CudaFunction, DeviceArg, HardwareInfo};
    use cute::prelude::*;
    use cute::testing::{BenchmarkOptions, benchmark};

    use super::{Problem, Spec, kernel, max_error, merge_kernel};
    use crate::prefill::{Plan, p as pp};

    type Loaded<'c> = Rc<(Kernel, CudaFunction<'c>)>;

    /// Block-sparse and merge kernels, compiled and loaded on first use.
    pub struct Kernels<'c> {
        ctx: &'c CudaContext,
        tc: Toolchain,
        opts: CompileOptions,
        cache: RefCell<HashMap<Spec, Loaded<'c>>>,
        merges: RefCell<HashMap<usize, Loaded<'c>>>,
        /// CTAs the planner assumes run at once: 2 per SM, as FlashInfer's.
        pub max_grid: usize,
    }

    impl<'c> Kernels<'c> {
        pub fn new(ctx: &'c CudaContext, tc: Toolchain) -> Result<Kernels<'c>, Box<dyn Error>> {
            let arch = ctx.best_target(&tc).ok_or("no target for this GPU")?;
            let opts = CompileOptions::new().gpu_arch(arch);
            let sms = HardwareInfo::new(0)?.get_device_multiprocessor_count()? as usize;
            Ok(Kernels {
                ctx,
                tc,
                opts,
                cache: RefCell::new(HashMap::new()),
                merges: RefCell::new(HashMap::new()),
                max_grid: 2 * sms,
            })
        }

        /// The kernel for `spec`, compiled on first use.
        pub fn get(&self, spec: &Spec) -> Result<Loaded<'c>, Box<dyn Error>> {
            if let Some(k) = self.cache.borrow().get(spec) {
                return Ok(k.clone());
            }
            let k = kernel(spec);
            let f = self.ctx.load(&k.compile_with(&self.tc, &self.opts)?)?;
            let loaded = Rc::new((k, f));
            self.cache.borrow_mut().insert(*spec, loaded.clone());
            Ok(loaded)
        }

        /// The merge for `head_dim`, compiled on first use.
        pub fn merge(&self, head_dim: usize) -> Result<Loaded<'c>, Box<dyn Error>> {
            if let Some(k) = self.merges.borrow().get(&head_dim) {
                return Ok(k.clone());
            }
            let k = merge_kernel(head_dim);
            let f = self.ctx.load(&k.compile_with(&self.tc, &self.opts)?)?;
            let loaded = Rc::new((k, f));
            self.merges.borrow_mut().insert(head_dim, loaded.clone());
            Ok(loaded)
        }
    }

    /// Run `p` once on the GPU under FlashInfer's plan; if `check`, compare with the f64
    /// reference; if `time`, time the launches (kernel and merge) replayed from a CUDA
    /// graph, `iterations` times. Returns the plan, the errors and the GPU time in
    /// microseconds.
    #[allow(clippy::type_complexity)]
    pub fn run(
        ks: &Kernels<'_>,
        p: &Problem,
        check: bool,
        time: Option<usize>,
    ) -> Result<(Plan, f32, f32, Option<f64>), Box<dyn Error>> {
        let plan = p.plan(ks.max_grid);
        let (out, t) = launch(ks, p, &plan, check, time)?;
        let (mut eo, mut el) = (0.0, 0.0);
        if let Some((o, lse)) = out {
            let (ro, rl) = p.reference();
            (eo, el) = (max_error(&o, &ro), max_error(&lse, &rl));
        }
        Ok((plan, eo, el, t))
    }

    /// Upload `p`, run it once under `plan`, download `o` and `lse` (kernel layout) if
    /// `fetch`, and time it from a CUDA graph if `time` is `Some(iterations)`.
    #[allow(clippy::type_complexity)]
    pub fn launch(
        ks: &Kernels<'_>,
        p: &Problem,
        plan: &Plan,
        fetch: bool,
        time: Option<usize>,
    ) -> Result<(Option<(Vec<f32>, Vec<f32>)>, Option<f64>), Box<dyn Error>> {
        let ctx = ks.ctx;
        let loaded = ks.get(&p.spec(plan))?;
        let merge = ks.merge(p.head_dim)?;
        let host = p.args(plan);
        let mut bufs = Vec::new();
        for a in &host {
            bufs.push(match a {
                KernelArg::ScalarI32(_) | KernelArg::ScalarF32(_) => None,
                _ => Some(ctx.upload(a)?),
            });
        }
        let dev = |i: usize| match &bufs[i] {
            Some(b) => DeviceArg::Buffer(b),
            None => DeviceArg::Scalar(host[i].clone()),
        };
        let args: Vec<DeviceArg> = (0..host.len()).map(dev).collect();
        let margs = vec![
            dev(pp::TMP_O),
            dev(pp::TMP_LSE),
            dev(pp::MERGE_INDPTR),
            dev(pp::O),
            dev(pp::LSE),
            DeviceArg::Scalar(KernelArg::ScalarI32(p.num_qo_heads() as i32)),
        ];
        let mut run = |s: &cute::cuda::CudaStream<'_>, _w: usize| -> Result<(), Box<dyn Error>> {
            ctx.launch_builder(&loaded.0, &loaded.1, p.launch(plan)).stream(s).args(&args).submit()?;
            if plan.split {
                ctx.launch_builder(&merge.0, &merge.1, p.merge_launch()).stream(s).args(&margs).submit()?;
            }
            Ok(())
        };
        let stream = ctx.stream()?;
        run(&stream, 0)?;
        stream.synchronize()?;
        let out = if fetch {
            let o = bufs[pp::O].as_ref().unwrap().to_vec::<F16>()?;
            let lse = bufs[pp::LSE].as_ref().unwrap().to_vec::<f32>()?;
            Some((o.into_iter().map(|x| x.to_f32()).collect(), lse))
        } else {
            None
        };
        let t = match time {
            Some(iterations) => {
                let opts = BenchmarkOptions::new().warmup_iterations(3).iterations(iterations).use_cuda_graphs(true);
                Some(benchmark(&stream, &opts, &mut run)?)
            }
            None => None,
        };
        Ok((out, t))
    }
}
