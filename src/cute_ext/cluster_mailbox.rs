// SPDX-License-Identifier: MIT OR Apache-2.0

//! cute_ext's DSMEM cluster mailbox (roadmap part m11-cute-ext-sched-tmem;
//! CuTe DSL's `cutlass.cute.experimental.cluster_mailbox`): a typed payload
//! one CTA writes into the shared memory of its pair's peer with
//! `st.async`, delivered through a pipeline stage of the peer.
//!
//! # The b32 encoding
//!
//! A [`B32PayloadCodec`] packs named scalar fields into 32-bit words, in
//! field order: `Boolean`, `Int8`, `Int16`, `Int32`, `Float16`, `BFloat16`
//! and `Float32` take one word each (the narrow types in its low bits:
//! integers sign- or zero-extended as converted to `Int32`, the 16-bit
//! floats' bits sign-extended, a `Float32`'s bits as they are), an `Int64`
//! two (low word first). Other types are refused. It is an encoding, not
//! a struct's natural layout: the storage is `storage_words` `Int32`s (a
//! `MemRange<Int32>` of a [`cute_struct!`](crate::cute_struct)), and a
//! send moves `tx_bytes = 4 * storage_words` bytes, in `v4`, `v2` and
//! scalar stores chosen by each word's address alignment
//! ([`chunks_for_alignment`](B32PayloadCodec::chunks_for_alignment)).
//!
//! # The protocol
//!
//! The mailbox rides on a pipeline whose barriers the peer owns:
//! [`CTA_PAIR_PEER_LOCALE`] (`PipelineLocale::cluster(CtaPairPeer,
//! CtaPairPeer)`). The sending CTA acquires a stage (its own empty barrier,
//! which the peer's consumers release to), then
//! [`send_to_pipeline_stage`](ClusterMailbox::send_to_pipeline_stage)
//! issues from one elected thread an `mbarrier.expect_tx` of `tx_bytes` on
//! the peer's full barrier and one `st.async` per chunk into the peer's
//! copy of the storage, each completing its bytes on that barrier; then the
//! sender commits the stage as usual (its arrival goes to the peer). The
//! peer's consumer waits on its full barrier, [`recv`](ClusterMailbox::recv)s
//! the payload from its own shared memory and releases. Slot acquisition,
//! commit and release stay the caller's.

use std::collections::HashSet;
use std::fmt;

use super::pipeline_ops::{MbarrierOwner, PipelineLocale, StageToken, get_mbarrier};
use crate::arch::{MbarrierOptions, elect_one, mbarrier_expect_tx_with, store_async_dsmem};
use crate::builder;
use crate::control::if_;
use crate::ir::DType;
use crate::smem_struct::{MemRange, MemberKind, StructLayout};
use crate::tensor::Ptr;
use crate::value::{AnyValue, Int32, Int64, IntoValue, Numeric, Value};
use crate::{BF16, F16};

/// The locale of a pipeline a mailbox sends through (CuTe DSL's
/// `CTA_PAIR_PEER_LOCALE`): both barriers owned by the pair's peer.
pub const CTA_PAIR_PEER_LOCALE: PipelineLocale =
    PipelineLocale::Cluster { full: MbarrierOwner::CtaPairPeer, empty: MbarrierOwner::CtaPairPeer };

/// The words a field of `dtype` takes, or `None` for a type the codec does
/// not take.
fn words_of(dtype: DType) -> Option<usize> {
    match dtype {
        DType::Bool | DType::I8 | DType::I16 | DType::I32 | DType::F16 | DType::BF16 | DType::F32 => Some(1),
        DType::I64 => Some(2),
        _ => None,
    }
}

/// One field of a [`B32PayloadCodec`]: its name, type and first word.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PayloadField {
    pub name: String,
    pub dtype: DType,
    /// The index of its first word.
    pub word: usize,
    /// The number of words (2 for `Int64`, else 1).
    pub words: usize,
}

/// Named scalar fields encoded into 32-bit words (CuTe DSL's
/// `B32PayloadCodec`; see the [module docs](self)).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct B32PayloadCodec {
    fields: Vec<PayloadField>,
    storage_words: usize,
    name: String,
}

impl B32PayloadCodec {
    /// A codec for `fields` (name and type, in word order), named `name`
    /// (default `"B32PayloadCodec"`). Panics for no fields, a repeated
    /// name and a type outside Boolean, Int8, Int16, Int32, Int64,
    /// Float16, BFloat16 and Float32.
    #[track_caller]
    pub fn new(fields: &[(&str, DType)], name: Option<&str>) -> B32PayloadCodec {
        assert!(!fields.is_empty(), "cute: experimental::B32PayloadCodec: a payload needs at least one field");
        let mut seen = HashSet::new();
        let mut out = Vec::with_capacity(fields.len());
        let mut word = 0;
        for &(field, dtype) in fields {
            assert!(seen.insert(field), "cute: experimental::B32PayloadCodec: the field name `{field}` is used twice");
            let Some(words) = words_of(dtype) else {
                panic!(
                    "cute: experimental::B32PayloadCodec: field `{field}` is {}, which has no b32 encoding (the \
                     codec takes Boolean, Int8, Int16, Int32, Int64, Float16, BFloat16 and Float32)",
                    dtype.cute_name()
                )
            };
            out.push(PayloadField { name: field.to_string(), dtype, word, words });
            word += words;
        }
        B32PayloadCodec { fields: out, storage_words: word, name: name.unwrap_or("B32PayloadCodec").to_string() }
    }

    /// A codec from a shared-storage struct's members (CuTe DSL's
    /// `from_struct` of a `cute.struct`): their names and scalar types, in
    /// order, named after the struct. Only the names and types are used,
    /// not the struct's byte layout. Panics for a member that is not a
    /// scalar.
    #[track_caller]
    pub fn from_struct(layout: &StructLayout) -> B32PayloadCodec {
        let fields: Vec<(&str, DType)> = layout
            .members()
            .iter()
            .map(|m| match &m.kind {
                MemberKind::Scalar(t) => (m.name.as_str(), *t),
                other => panic!(
                    "cute: experimental::B32PayloadCodec::from_struct: member `{}` of {} is {other}; a payload's \
                     fields are scalars",
                    m.name,
                    layout.name()
                ),
            })
            .collect();
        B32PayloadCodec::new(&fields, Some(layout.name()))
    }

    /// A codec for a [`B32Payload`] type (CuTe DSL's `from_struct` of an
    /// annotated class).
    #[track_caller]
    pub fn from_payload<P: B32Payload>() -> B32PayloadCodec {
        B32PayloadCodec::new(&P::fields(), Some(P::NAME))
    }

    /// The payload's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The fields, in word order.
    pub fn fields(&self) -> &[PayloadField] {
        &self.fields
    }

    /// The field names, in order.
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }

    /// The 32-bit words the encoding takes.
    pub fn storage_words(&self) -> usize {
        self.storage_words
    }

    /// The bytes a send moves (`4 * storage_words`).
    pub fn tx_bytes(&self) -> usize {
        4 * self.storage_words
    }

    /// The storage alignment the widest store wants: 16 bytes from four
    /// words, 8 from two, else 4.
    pub fn required_alignment(&self) -> usize {
        match self.storage_words {
            w if w >= 4 => 16,
            w if w >= 2 => 8,
            _ => 4,
        }
    }

    /// The stores of the words for storage at a `base_align`-byte aligned
    /// address (CuTe DSL's `chunks_for_alignment`): `(size, words)` pairs
    /// in order, each a `v4` (4 words), `v2` (2) or scalar (1) store. Word
    /// `i` is at byte `4 i`; its alignment is the greatest common divisor of
    /// `base_align` and `4 i` (`base_align` for word 0), capped at 16. A
    /// `v4` starts at a 16-byte aligned word with four words left, a `v2` at
    /// an 8-byte aligned one with two.
    pub fn chunks_for_alignment(&self, base_align: usize) -> Vec<(usize, Vec<usize>)> {
        chunk_words(self.storage_words, base_align)
    }

    /// Field `name`. Panics for an unknown name.
    #[track_caller]
    pub fn field(&self, name: &str) -> &PayloadField {
        self.fields.iter().find(|f| f.name == name).unwrap_or_else(|| {
            panic!("cute: experimental::B32PayloadCodec: payload `{}` has no field `{name}`", self.name)
        })
    }

    /// The words of `fields` (CuTe DSL's `encode_fields`): every field
    /// exactly once, in any order. A value of another type than its
    /// field's is converted to it first ([`Value::to`]). Panics, naming
    /// them, for missing or unknown fields.
    #[track_caller]
    pub fn encode_fields(&self, fields: &[(&str, AnyValue)]) -> Vec<Int32> {
        let given: Vec<&str> = fields.iter().map(|(n, _)| *n).collect();
        let expected: HashSet<&str> = self.fields.iter().map(|f| f.name.as_str()).collect();
        let mut missing: Vec<&str> = expected.iter().copied().filter(|n| !given.contains(n)).collect();
        let mut extra: Vec<&str> = given.iter().copied().filter(|n| !expected.contains(n)).collect();
        let mut dup: Vec<&str> =
            given.iter().enumerate().filter(|(i, n)| given[..*i].contains(n)).map(|(_, n)| *n).collect();
        missing.sort_unstable();
        extra.sort_unstable();
        dup.dedup();
        if !(missing.is_empty() && extra.is_empty() && dup.is_empty()) {
            let mut parts = Vec::new();
            if !missing.is_empty() {
                parts.push(format!("no value for {missing:?}"));
            }
            if !extra.is_empty() {
                parts.push(format!("{extra:?} are not fields of it"));
            }
            if !dup.is_empty() {
                parts.push(format!("{dup:?} given more than once"));
            }
            panic!("cute: experimental::B32PayloadCodec::encode_fields: payload `{}`: {}", self.name, parts.join("; "));
        }
        let mut words = Vec::with_capacity(self.storage_words);
        for f in &self.fields {
            let v = fields.iter().find(|(n, _)| *n == f.name).map(|(_, v)| *v).expect("checked above");
            words.extend(encode(v, f.dtype));
        }
        words
    }

    /// The fields decoded from the words at `ptr` (CuTe DSL's
    /// `decode_from_b32_ptr`), a pointer to this CTA's memory.
    #[track_caller]
    pub fn decode_from_b32_ptr(&self, ptr: &Ptr<i32>) -> DecodedPayload {
        let values = self
            .fields
            .iter()
            .map(|f| {
                let w = |i: usize| ptr.offset(f.word + i).load();
                let v = if f.words == 2 { decode_i64(w(0), w(1)) } else { decode(w(0), f.dtype) };
                (f.name.clone(), v)
            })
            .collect();
        DecodedPayload { name: self.name.clone(), values }
    }
}

impl fmt::Display for B32PayloadCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fields: Vec<String> = self.fields.iter().map(|x| format!("{}: {}", x.name, x.dtype.cute_name())).collect();
        write!(f, "{} {{ {} }}", self.name, fields.join(", "))
    }
}

/// Group `total` words into stores by alignment (see
/// [`B32PayloadCodec::chunks_for_alignment`]).
fn chunk_words(total: usize, base_align: usize) -> Vec<(usize, Vec<usize>)> {
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i < total {
        let align = if i == 0 { base_align } else { gcd(base_align, 4 * i) };
        let align = [16, 8, 4].into_iter().find(|&k| align >= k).unwrap_or(align);
        let left = total - i;
        let size = if left >= 4 && align >= 16 {
            4
        } else if left >= 2 && align >= 8 {
            2
        } else {
            1
        };
        out.push((size, (i..i + size).collect()));
        i += size;
    }
    out
}

/// The value of `v` (any type) as `T`.
#[track_caller]
fn as_type<T: Numeric>(v: AnyValue) -> Value<T> {
    let t = builder::with(|b| b.dtype(v.0));
    macro_rules! conv {
        ($($d:ident => $r:ty),*) => {
            match t {
                $(DType::$d => Value::<$r>::from_id(v.0).to::<T>(),)*
                other => panic!(
                    "cute: experimental::B32PayloadCodec: a {} value cannot be converted to the field's {}",
                    other.cute_name(),
                    T::DTYPE.cute_name()
                ),
            }
        };
    }
    if t == T::DTYPE {
        return Value::from_id(v.0);
    }
    conv!(Bool => bool, I8 => i8, I16 => i16, I32 => i32, I64 => i64, U8 => u8, U16 => u16, U32 => u32,
          U64 => u64, F16 => F16, BF16 => BF16, F32 => f32, F64 => f64)
}

/// The words of `v` as a field of `dtype`.
#[track_caller]
fn encode(v: AnyValue, dtype: DType) -> Vec<Int32> {
    match dtype {
        DType::I32 => vec![as_type::<i32>(v)],
        DType::F32 => vec![as_type::<f32>(v).bitcast::<i32>()],
        DType::F16 => vec![as_type::<F16>(v).bitcast::<i16>().to::<i32>()],
        DType::BF16 => vec![as_type::<BF16>(v).bitcast::<i16>().to::<i32>()],
        DType::Bool => vec![as_type::<bool>(v).to::<i32>()],
        DType::I8 => vec![as_type::<i8>(v).to::<i32>()],
        DType::I16 => vec![as_type::<i16>(v).to::<i32>()],
        DType::I64 => {
            let x: Int64 = as_type::<i64>(v);
            vec![(x & 0xFFFF_FFFFi64).to::<i32>(), (x >> 32i64).to::<i32>()]
        }
        other => unreachable!("{} has no encoding", other.cute_name()),
    }
}

/// The field of `dtype` (one word) from `w`.
#[track_caller]
fn decode(w: Int32, dtype: DType) -> AnyValue {
    match dtype {
        DType::I32 => w.into(),
        DType::F32 => w.bitcast::<f32>().into(),
        DType::F16 => w.to::<i16>().bitcast::<F16>().into(),
        DType::BF16 => w.to::<i16>().bitcast::<BF16>().into(),
        DType::Bool => w.to::<bool>().into(),
        DType::I8 => w.to::<i8>().into(),
        DType::I16 => w.to::<i16>().into(),
        other => unreachable!("{} has no one-word encoding", other.cute_name()),
    }
}

/// An `Int64` from its low and high words.
#[track_caller]
fn decode_i64(lo: Int32, hi: Int32) -> AnyValue {
    let mask = 0xFFFF_FFFFi64;
    let low = lo.to::<i64>() & mask;
    let high = (hi.to::<i64>() & mask) << 32i64;
    (low | high).into()
}

/// A decoded payload (CuTe DSL's namespace of fields): each field's value
/// by name.
#[derive(Clone, Debug)]
pub struct DecodedPayload {
    name: String,
    values: Vec<(String, AnyValue)>,
}

impl DecodedPayload {
    /// The fields, in order.
    pub fn values(&self) -> &[(String, AnyValue)] {
        &self.values
    }

    /// Field `name`, untyped. Panics for an unknown name.
    #[track_caller]
    pub fn get_any(&self, name: &str) -> AnyValue {
        self.values.iter().find(|(n, _)| n == name).map(|(_, v)| *v).unwrap_or_else(|| {
            panic!("cute: experimental::DecodedPayload: payload `{}` has no field `{name}`", self.name)
        })
    }

    /// Field `name` as `T`, which must be its type. Panics otherwise.
    #[track_caller]
    pub fn get<T: Numeric>(&self, name: &str) -> Value<T> {
        let v = self.get_any(name);
        let t = builder::with(|b| b.dtype(v.0));
        assert!(
            t == T::DTYPE,
            "cute: experimental::DecodedPayload: field `{name}` of `{}` is {}, not {}",
            self.name,
            t.cute_name(),
            T::DTYPE.cute_name()
        );
        Value::from_id(v.0)
    }
}

/// A Rust type that is a mailbox payload (CuTe DSL's annotated class given
/// to `b32_payload`): its fields' names and types, and its conversion to
/// and from field values. [`b32_payload`] makes its codec.
pub trait B32Payload: Sized {
    /// The payload's name.
    const NAME: &'static str;
    /// The fields, in word order.
    fn fields() -> Vec<(&'static str, DType)>;
    /// The fields' values.
    fn to_fields(&self) -> Vec<(&'static str, AnyValue)>;
    /// The payload from decoded fields.
    fn from_fields(fields: &DecodedPayload) -> Self;
}

/// The codec of a [`B32Payload`] type (CuTe DSL's `b32_payload`).
#[track_caller]
pub fn b32_payload<P: B32Payload>() -> B32PayloadCodec {
    B32PayloadCodec::from_payload::<P>()
}

/// [`b32_payload`] by its older name (CuTe DSL's `dsmem_payload`).
#[track_caller]
pub fn dsmem_payload<P: B32Payload>() -> B32PayloadCodec {
    b32_payload::<P>()
}

/// A one-slot DSMEM mailbox for a typed payload sent to the pair's peer CTA
/// (CuTe DSL's `ClusterMailbox`; see the [module docs](self)).
#[derive(Clone)]
pub struct ClusterMailbox {
    codec: B32PayloadCodec,
    storage: MemRange<i32>,
}

impl fmt::Debug for ClusterMailbox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ClusterMailbox({})", self.codec)
    }
}

impl ClusterMailbox {
    /// A mailbox for `payload` in `storage`, a struct's
    /// `MemRange<Int32>` of `payload.storage_words()` words (align it to 16
    /// bytes, `#[align(16)]`, for `v4` stores). Panics for storage of
    /// another length.
    #[track_caller]
    pub fn new(payload: &B32PayloadCodec, storage: MemRange<i32>) -> ClusterMailbox {
        assert!(
            storage.len() == payload.storage_words(),
            "cute: experimental::ClusterMailbox: payload `{}` takes {} b32 words, but the storage holds {}",
            payload.name(),
            payload.storage_words(),
            storage.len()
        );
        ClusterMailbox { codec: payload.clone(), storage }
    }

    /// The codec.
    pub fn payload(&self) -> &B32PayloadCodec {
        &self.codec
    }

    /// The storage.
    pub fn storage(&self) -> &MemRange<i32> {
        &self.storage
    }

    /// The stores a send issues for this storage's alignment.
    pub fn chunks(&self) -> Vec<(usize, Vec<usize>)> {
        self.codec.chunks_for_alignment(self.storage.data_ptr().max_alignment())
    }

    /// Send `fields` to CTA `peer_rank` through the stage of `stage_token`
    /// (CuTe DSL's `send_to_pipeline_stage`): a producer token of a
    /// [`CTA_PAIR_PEER_LOCALE`] pipeline, whose stage the caller has
    /// acquired and commits afterwards. One elected thread of the calling
    /// warp adds `tx_bytes` to the peer's full barrier and issues the
    /// `st.async` stores, which complete them. Panics for a token of a
    /// pipeline of another locale.
    #[track_caller]
    pub fn send_to_pipeline_stage(
        &self,
        stage_token: &StageToken,
        peer_rank: impl IntoValue<i32>,
        fields: &[(&str, AnyValue)],
    ) {
        let _at = builder::LocationScope::new();
        assert!(
            stage_token.locale() == Some(CTA_PAIR_PEER_LOCALE),
            "cute: experimental::ClusterMailbox::send_to_pipeline_stage: the stage token comes from {}, but a mailbox \
             sends through a pipeline of locale {CTA_PAIR_PEER_LOCALE} (CTA_PAIR_PEER_LOCALE), whose full barrier the \
             peer owns",
            stage_token.pipeline_type()
        );
        let peer = peer_rank.into_value();
        let words = self.codec.encode_fields(fields);
        let mbar = get_mbarrier(stage_token);
        let base = self.storage.data_ptr();
        let chunks = self.chunks();
        let tx = self.codec.tx_bytes() as i32;
        if_(elect_one(), || {
            mbarrier_expect_tx_with(
                &mbar,
                tx,
                MbarrierOptions { peer_cta_rank_in_cluster: Some(peer), ..Default::default() },
            );
            for (size, group) in &chunks {
                let ptr = base.offset(group[0]);
                let w = |k: usize| words[group[k]];
                match size {
                    4 => store_async_dsmem(&ptr, [w(0), w(1), w(2), w(3)], &mbar, peer),
                    2 => store_async_dsmem(&ptr, [w(0), w(1)], &mbar, peer),
                    _ => store_async_dsmem(&ptr, [w(0)], &mbar, peer),
                }
            }
        });
    }

    /// [`send_to_pipeline_stage`](Self::send_to_pipeline_stage) of a
    /// [`B32Payload`] value.
    #[track_caller]
    pub fn send_payload<P: B32Payload>(&self, stage_token: &StageToken, peer_rank: impl IntoValue<i32>, payload: &P) {
        let fields = payload.to_fields();
        let fields: Vec<(&str, AnyValue)> = fields.iter().map(|(n, v)| (*n, *v)).collect();
        self.send_to_pipeline_stage(stage_token, peer_rank, &fields);
    }

    /// The received payload, read from this CTA's storage (CuTe DSL's
    /// `recv`); call it after the consumer's wait on the stage.
    #[track_caller]
    pub fn recv(&self) -> DecodedPayload {
        self.codec.decode_from_b32_ptr(&self.storage.data_ptr())
    }

    /// [`recv`](Self::recv) as a [`B32Payload`] value.
    #[track_caller]
    pub fn recv_as<P: B32Payload>(&self) -> P {
        P::from_fields(&self.recv())
    }
}
