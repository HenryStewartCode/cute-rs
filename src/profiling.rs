//! Profiling scopes, the kernel symbol registry, and IKET in-kernel event tracing
//! (CuTe DSL's `cutlass.utils.profiling`, `cutlass.pipeline.profiling` and
//! `cute.experimental.iket`, reimplemented from their semantics).
//!
//! * **Scopes.** [`Scope`] names the instructions built inside it. Each instruction records
//!   its scope ([`Inst::scope`](crate::ir::Inst)); the MLIR emitter wraps its location in
//!   NameLocs (`loc("a"("b"("f.rs":1:1)))`, outermost first), IR dumps print the
//!   path, and simulator diagnostics name the scope the failing instruction was built in.
//! * **The symbol registry.** It lives in the builder (one per kernel, not a global) and
//!   comes back with the [`Kernel`](crate::Kernel): [`Kernel::symbols`](crate::Kernel::symbols),
//!   [`Kernel::dump_profiling_metadata`](crate::Kernel::dump_profiling_metadata).
//! * **IKET.** CuTe DSL lowers its event calls through NVIDIA's closed IKET pass; cute-rs has
//!   its own design. With [`KernelOptions::iket`](crate::KernelOptions::iket) on, each event
//!   is one 8-word record appended to a hidden module global ([`IKET_GLOBAL`]) by the lowest
//!   active lane of the warp, reserving its slot with one atomic add. Overflow sets a flag
//!   and drops the record. After a launch or a simulation the host reads the global and
//!   [`Kernel::decode_iket`](crate::Kernel::decode_iket) pairs starts with ends and pushes
//!   with pops, reports what is unmatched, and writes Chrome-trace (Perfetto) JSON. In the
//!   simulator the timestamp is the instruction counter, so traces are deterministic. With
//!   IKET off, the calls emit nothing.

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::builder;
use crate::control::{Carry, if_, if_else};
use crate::ir::ValueId;
use crate::tensor::Tensor;
use crate::value::{Boolean, Int32, Int64, Uint32, Value};

// ---------------------------------------------------------------------------------------
// Scopes
// ---------------------------------------------------------------------------------------

/// An index into [`KernelIr::scopes`](crate::ir::KernelIr).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ScopeId(pub u32);

/// One level of a scope path: a name and the enclosing level.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct NameScope {
    pub name: String,
    pub parent: Option<ScopeId>,
}

/// A value of a registry entry's metadata.
#[derive(Clone, PartialEq, Debug)]
pub enum MetaValue {
    Int(i64),
    Bool(bool),
    Str(String),
}

macro_rules! meta_from_int {
    ($($t:ty),*) => {$(impl From<$t> for MetaValue { fn from(v: $t) -> Self { MetaValue::Int(v as i64) } })*};
}
meta_from_int!(i32, i64, u32, u64, usize);
impl From<bool> for MetaValue {
    fn from(v: bool) -> Self {
        MetaValue::Bool(v)
    }
}
impl From<&str> for MetaValue {
    fn from(v: &str) -> Self {
        MetaValue::Str(v.to_string())
    }
}
impl From<String> for MetaValue {
    fn from(v: String) -> Self {
        MetaValue::Str(v)
    }
}

impl MetaValue {
    fn json(&self) -> String {
        match self {
            MetaValue::Int(v) => v.to_string(),
            MetaValue::Bool(v) => v.to_string(),
            MetaValue::Str(s) => json_str(s),
        }
    }
}

/// A JSON string literal.
fn json_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// One symbol of the registry: a name, its kind (`"warp"`, `"barrier"`, ...) and metadata.
#[derive(Clone, PartialEq, Debug)]
pub struct SymbolEntry {
    pub name: String,
    pub kind: Option<String>,
    pub metadata: Vec<(String, MetaValue)>,
}

impl SymbolEntry {
    /// A metadata value by key.
    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    fn int(&self, key: &str) -> Option<i64> {
        match self.get(key) {
            Some(MetaValue::Int(v)) => Some(*v),
            _ => None,
        }
    }

    fn text(&self, key: &str) -> Option<String> {
        match self.get(key) {
            Some(MetaValue::Str(v)) => Some(v.clone()),
            _ => None,
        }
    }
}

/// The symbols registered while a kernel builds, in registration order.
#[derive(Clone, PartialEq, Default, Debug)]
pub struct SymbolRegistry {
    pub entries: Vec<SymbolEntry>,
}

impl SymbolRegistry {
    fn register(&mut self, name: &str, kind: Option<&str>, metadata: &[(&str, MetaValue)]) {
        let at = match self.entries.iter().position(|e| e.name == name) {
            Some(at) => at,
            None => {
                self.entries.push(SymbolEntry { name: name.to_string(), kind: None, metadata: Vec::new() });
                self.entries.len() - 1
            }
        };
        let entry = &mut self.entries[at];
        if let Some(kind) = kind {
            entry.kind = Some(kind.to_string());
        }
        for (key, value) in metadata {
            match entry.metadata.iter_mut().find(|(k, _)| k == key) {
                Some(slot) => slot.1 = value.clone(),
                None => entry.metadata.push((key.to_string(), value.clone())),
            }
        }
    }
}

/// A warp range registered by a [`WarpScope`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WarpSymbol {
    pub name: String,
    pub warp_start: i64,
    pub warp_end: i64,
}

/// A named barrier registered by a pipeline's `create(name=...)`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BarrierSymbol {
    pub name: String,
    pub num_stages: i64,
    pub pipeline_type: String,
}

/// The registry grouped by kind (CuTe DSL's `dump_kernel_symbols`).
#[derive(Clone, PartialEq, Eq, Default, Debug)]
pub struct KernelSymbols {
    pub warps: Vec<WarpSymbol>,
    pub barriers: Vec<BarrierSymbol>,
    /// `name:stages, name:stages, ...` over the barriers.
    pub bar_alloc: String,
}

impl KernelSymbols {
    /// Group `registry`'s entries by kind.
    pub fn from_registry(registry: &SymbolRegistry) -> KernelSymbols {
        let mut out = KernelSymbols::default();
        for e in &registry.entries {
            match e.kind.as_deref() {
                Some("warp") => {
                    if let (Some(warp_start), Some(warp_end)) = (e.int("warp_start"), e.int("warp_end")) {
                        out.warps.push(WarpSymbol { name: e.name.clone(), warp_start, warp_end });
                    }
                }
                Some("barrier") => {
                    if let (Some(num_stages), Some(pipeline_type)) = (e.int("num_stages"), e.text("pipeline_type")) {
                        out.barriers.push(BarrierSymbol { name: e.name.clone(), num_stages, pipeline_type });
                    }
                }
                _ => {}
            }
        }
        out.bar_alloc =
            out.barriers.iter().map(|b| format!("{}:{}", b.name, b.num_stages)).collect::<Vec<_>>().join(", ");
        out
    }

    /// No warps and no barriers.
    pub fn is_empty(&self) -> bool {
        self.warps.is_empty() && self.barriers.is_empty()
    }

    /// `kernel_symbols.json`'s content, with `extra` merged in as top-level keys.
    pub fn to_json(&self, extra: &[(&str, MetaValue)]) -> String {
        let warps: Vec<String> = self
            .warps
            .iter()
            .map(|w| {
                format!("{}: {{\"warp_start\": {}, \"warp_end\": {}}}", json_str(&w.name), w.warp_start, w.warp_end)
            })
            .collect();
        let barriers: Vec<String> = self
            .barriers
            .iter()
            .map(|b| {
                format!(
                    "{}: {{\"num_stages\": {}, \"pipeline_type\": {}}}",
                    json_str(&b.name),
                    b.num_stages,
                    json_str(&b.pipeline_type)
                )
            })
            .collect();
        let mut parts = vec![
            format!("\"warps\": {{{}}}", warps.join(", ")),
            format!("\"barriers\": {{{}}}", barriers.join(", ")),
            format!("\"bar_alloc\": {}", json_str(&self.bar_alloc)),
        ];
        for (key, value) in extra {
            parts.push(format!("{}: {}", json_str(key), value.json()));
        }
        format!("{{{}}}\n", parts.join(", "))
    }

    /// Write `kernel_symbols.json` into `dir`; nothing when there are neither barriers nor warps.
    pub fn dump(
        &self,
        dir: &std::path::Path,
        extra: &[(&str, MetaValue)],
    ) -> std::io::Result<Option<std::path::PathBuf>> {
        if self.is_empty() {
            return Ok(None);
        }
        std::fs::create_dir_all(dir)?;
        let path = dir.join("kernel_symbols.json");
        std::fs::write(&path, self.to_json(extra))?;
        Ok(Some(path))
    }
}

/// Register `name` in the kernel's symbol registry, creating the entry if needed and merging
/// `metadata` into it. A `kind` of `Some` sets the entry's kind. Only inside
/// [`Kernel::build`](crate::Kernel::build).
#[track_caller]
pub fn register_symbol(name: &str, kind: Option<&str>, metadata: &[(&str, MetaValue)]) {
    builder::with(|b| b.profiling.registry.register(name, kind, metadata));
}

/// A copy of the registry as built so far.
#[track_caller]
pub fn get_symbol_registry() -> SymbolRegistry {
    builder::with(|b| b.profiling.registry.clone())
}

/// Clear the registry.
#[track_caller]
pub fn reset_symbol_registry() {
    builder::with(|b| b.profiling.registry = SymbolRegistry::default());
}

/// The registry grouped by kind, which it clears. Whatever is left when the build ends is
/// what [`Kernel::symbols`](crate::Kernel::symbols) returns.
#[track_caller]
pub fn dump_kernel_symbols() -> KernelSymbols {
    builder::with(|b| KernelSymbols::from_registry(&std::mem::take(&mut b.profiling.registry)))
}

/// Register a pipeline's barriers (every pipeline `create` with a name calls this): the
/// name, its stage count and the pipeline's type.
#[track_caller]
pub fn register_barrier(name: &str, num_stages: usize, pipeline_type: &str) {
    if name.is_empty() {
        return;
    }
    register_symbol(
        name,
        Some("barrier"),
        &[("num_stages", num_stages.into()), ("pipeline_type", pipeline_type.into())],
    );
}

/// A named scope over the instructions built while it is open; closed by
/// [`close`](Scope::close) or when dropped. Scopes opened inside an `if_`/`for_range`/`while`
/// body close in it.
#[must_use = "a Scope covers the instructions built while it is alive"]
pub struct Scope {
    base: usize,
    end: usize,
    iket: Option<RangeToken>,
    closed: bool,
}

impl Scope {
    /// Open a scope named `name`.
    #[track_caller]
    pub fn new(name: &str) -> Scope {
        Scope::open(&[name], false, &[])
    }

    /// Open nested scopes, outermost name first (CuTe DSL's tuple of names).
    #[track_caller]
    pub fn path(names: &[&str]) -> Scope {
        Scope::open(names, false, &[])
    }

    /// Open nested scopes with metadata (registered under the first name) and, with `iket`,
    /// an IKET range named `a/b`.
    #[track_caller]
    pub fn with(names: &[&str], iket: bool, metadata: &[(&str, MetaValue)]) -> Scope {
        Scope::open(names, iket, metadata)
    }

    #[track_caller]
    fn open(names: &[&str], iket: bool, metadata: &[(&str, MetaValue)]) -> Scope {
        assert!(!names.is_empty(), "cute: Scope needs at least one name");
        assert!(
            names.iter().all(|n| !n.is_empty() && !n.contains('/')),
            "cute: scope names must be non-empty and contain no '/': {names:?}"
        );
        if !metadata.is_empty() {
            register_symbol(names[0], None, metadata);
        }
        let (base, end) = builder::with(|b| b.profiling.push_names(names));
        let iket = iket.then(|| range_start(&names.join("/"), None));
        Scope { base, end, iket, closed: false }
    }

    /// Exit the current scope, then enter one named `name` (with `metadata`).
    #[track_caller]
    pub fn switch(&mut self, name: &str, metadata: &[(&str, MetaValue)]) {
        let iket = self.iket.is_some();
        self.shut();
        *self = Scope::open(&[name], iket, metadata);
    }

    /// Close the scope.
    #[track_caller]
    pub fn close(mut self) {
        self.shut();
    }

    #[track_caller]
    fn shut(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        if let Some(token) = self.iket.take() {
            range_end(token, None);
        }
        let (base, end) = (self.base, self.end);
        let ok = builder::with(|b| b.profiling.pop_to(base, end));
        assert!(ok, "cute: profiling scopes must close in the reverse order they opened");
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        if !self.closed && !std::thread::panicking() {
            self.shut();
        }
    }
}

/// Run `body` inside a scope named `name`.
#[track_caller]
pub fn scope<R>(name: &str, body: impl FnOnce() -> R) -> R {
    let s = Scope::new(name);
    let r = body();
    s.close();
    r
}

/// A [`Scope`] that registers a warp range (`kind = "warp"`, `warp_start`, `warp_end`).
#[must_use = "a WarpScope covers the instructions built while it is alive"]
pub struct WarpScope(Scope);

impl WarpScope {
    /// Open a scope named `name` for warps `warp_start..=warp_end`.
    #[track_caller]
    pub fn new(name: &str, warp_start: u32, warp_end: u32) -> WarpScope {
        WarpScope::with(name, warp_start, warp_end, false)
    }

    /// [`new`](Self::new), opening an IKET range named `name` with `iket`.
    #[track_caller]
    pub fn with(name: &str, warp_start: u32, warp_end: u32, iket: bool) -> WarpScope {
        register_symbol(name, Some("warp"), &[("warp_start", warp_start.into()), ("warp_end", warp_end.into())]);
        WarpScope(Scope::open(&[name], iket, &[]))
    }

    /// Close the scope.
    #[track_caller]
    pub fn close(self) {
        self.0.close();
    }
}

/// Scopes, the registry and IKET state of the kernel being built.
pub(crate) struct ProfilingState {
    pub(crate) scopes: Vec<NameScope>,
    pub(crate) stack: Vec<ScopeId>,
    pub(crate) registry: SymbolRegistry,
    iket: Option<IketBuild>,
}

struct IketBuild {
    options: IketOptions,
    events: Vec<String>,
    dags: Vec<IketDag>,
    /// The trace global, declared when the build starts.
    buffer: Option<Tensor<u32>>,
    /// Pushes not popped yet, in code order.
    open_pushes: Vec<String>,
}

impl ProfilingState {
    pub(crate) fn new(iket: Option<IketOptions>) -> ProfilingState {
        ProfilingState {
            scopes: Vec::new(),
            stack: Vec::new(),
            registry: SymbolRegistry::default(),
            iket: iket.map(|options| IketBuild {
                options,
                events: Vec::new(),
                dags: Vec::new(),
                buffer: None,
                open_pushes: Vec::new(),
            }),
        }
    }

    pub(crate) fn current_scope(&self) -> Option<ScopeId> {
        self.stack.last().copied()
    }

    /// Enter `names` nested; the stack depth before and after.
    fn push_names(&mut self, names: &[&str]) -> (usize, usize) {
        let base = self.stack.len();
        for name in names {
            let parent = self.current_scope();
            let id = match self.scopes.iter().position(|s| s.name == *name && s.parent == parent) {
                Some(i) => ScopeId(i as u32),
                None => {
                    self.scopes.push(NameScope { name: name.to_string(), parent });
                    ScopeId(self.scopes.len() as u32 - 1)
                }
            };
            self.stack.push(id);
        }
        (base, self.stack.len())
    }

    /// Leave the scopes opened between depths `base` and `end`, which must be the innermost.
    fn pop_to(&mut self, base: usize, end: usize) -> bool {
        if self.stack.len() != end {
            return false;
        }
        self.stack.truncate(base);
        true
    }

    /// The symbols and IKET decode info of the finished kernel; refuses unbalanced pushes.
    pub(crate) fn finish_build(&mut self, kernel: &str) -> (KernelSymbols, Option<IketInfo>) {
        let symbols = KernelSymbols::from_registry(&std::mem::take(&mut self.registry));
        let iket = self.iket.take().map(|b| {
            if let Some(name) = b.open_pushes.last() {
                panic!(
                    "cute: kernel `{kernel}` ends with {} range_push call(s) not popped (the last, `{name}`): every \
                     range_push needs a range_pop",
                    b.open_pushes.len()
                );
            }
            IketInfo { options: b.options, events: b.events, dags: b.dags }
        });
        (symbols, iket)
    }
}

// ---------------------------------------------------------------------------------------
// IKET: the device side
// ---------------------------------------------------------------------------------------

/// The name of the module global that holds the trace.
pub const IKET_GLOBAL: &str = "__cute_iket_trace";
/// Words of the trace global before its records.
const HEADER_WORDS: usize = 4;
/// Words per record.
const RECORD_WORDS: usize = 8;
/// The first event id of a named event; the ids below are reserved.
const FIRST_EVENT_ID: u32 = 32;
/// The event id `range_pop` records (the pop has no name of its own).
pub const POP_EVENT_ID: u32 = 31;

/// The clock that stamps events.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IketClock {
    /// `%globaltimer`, nanoseconds, comparable across SMs.
    Globaltimer,
    /// `%clock64`, cycles of one SM.
    Clock64,
}

/// What [`KernelOptions::iket`](crate::KernelOptions::iket) takes.
#[derive(Clone, Copy, Debug)]
pub struct IketOptions {
    /// Records the trace global holds (all warps together); later events set the overflow
    /// flag and are dropped.
    pub capacity: u32,
    pub clock: IketClock,
}

impl Default for IketOptions {
    fn default() -> Self {
        IketOptions { capacity: 4096, clock: IketClock::Globaltimer }
    }
}

/// A DAG of events, metadata only (CuTe DSL's `dag(name).edge(...)`).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct IketDag {
    pub name: String,
    pub edges: Vec<IketEdge>,
}

/// An edge of an [`IketDag`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct IketEdge {
    pub src: String,
    pub dst: String,
    pub label: Option<String>,
    pub via: Option<String>,
}

/// Adds edges to a DAG.
pub struct DagHandle {
    index: usize,
}

/// The DAG named `name`, created if needed. Metadata only: it appears in the trace file.
/// Does nothing when IKET is off.
#[track_caller]
pub fn dag(name: &str) -> DagHandle {
    let index = builder::with(|b| match b.profiling.iket.as_mut() {
        None => 0,
        Some(i) => match i.dags.iter().position(|d| d.name == name) {
            Some(at) => at,
            None => {
                i.dags.push(IketDag { name: name.to_string(), edges: Vec::new() });
                i.dags.len() - 1
            }
        },
    });
    DagHandle { index }
}

impl DagHandle {
    /// An edge from event `src` to event `dst`.
    #[track_caller]
    pub fn edge(&self, src: &str, dst: &str, label: Option<&str>, via: Option<&str>) -> &Self {
        builder::with(|b| {
            if let Some(i) = b.profiling.iket.as_mut() {
                i.dags[self.index].edges.push(IketEdge {
                    src: src.to_string(),
                    dst: dst.to_string(),
                    label: label.map(str::to_string),
                    via: via.map(str::to_string),
                });
            }
        });
        self
    }
}

/// An event's payload: an integer (its width and signedness recorded), a float or a pointer.
#[derive(Clone, Copy, Debug)]
pub enum Payload {
    I8(Value<i8>),
    U8(Value<u8>),
    I16(Value<i16>),
    U16(Value<u16>),
    I32(Int32),
    U32(Uint32),
    I64(Int64),
    U64(Value<u64>),
    F32(Value<f32>),
    /// A pointer, as its 64-bit address.
    Pointer(Int64),
}

macro_rules! payload_from {
    ($($t:ty => $v:ident),*) => {$(impl From<Value<$t>> for Payload { fn from(x: Value<$t>) -> Self { Payload::$v(x) } })*};
}
payload_from!(i8 => I8, u8 => U8, i16 => I16, u16 => U16, i32 => I32, u32 => U32, i64 => I64, u64 => U64, f32 => F32);

impl Payload {
    /// The type code stored in a record and by [`IketPayload::ty`].
    fn code(&self) -> u32 {
        match self {
            Payload::I8(_) => 1,
            Payload::U8(_) => 2,
            Payload::I16(_) => 3,
            Payload::U16(_) => 4,
            Payload::I32(_) => 5,
            Payload::U32(_) => 6,
            Payload::I64(_) => 7,
            Payload::U64(_) => 8,
            Payload::F32(_) => 9,
            Payload::Pointer(_) => 10,
        }
    }

    /// The payload as 64 bits: integers extended by their signedness, a float as its bits.
    #[track_caller]
    fn bits(&self) -> Int64 {
        match *self {
            Payload::I8(v) => v.to::<i64>(),
            Payload::U8(v) => v.to::<i64>(),
            Payload::I16(v) => v.to::<i64>(),
            Payload::U16(v) => v.to::<i64>(),
            Payload::I32(v) => v.to::<i64>(),
            Payload::U32(v) => v.to::<i64>(),
            Payload::I64(v) | Payload::Pointer(v) => v,
            Payload::U64(v) => v.bitcast::<i64>(),
            Payload::F32(v) => v.bitcast::<u32>().to::<i64>(),
        }
    }
}

fn payload_code(p: &Option<Payload>) -> u32 {
    p.as_ref().map_or(0, Payload::code)
}

fn payload_name(code: u32) -> &'static str {
    ["none", "i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "f32", "pointer"][code as usize % 11]
}

/// A sentinel token's payload code: the check at its end is skipped.
const ANY_PAYLOAD: u32 = u32::MAX;

const KIND_MARK: u32 = 0;
const KIND_START: u32 = 1;
const KIND_END: u32 = 2;
const KIND_PUSH: u32 = 3;
const KIND_POP: u32 = 4;

/// A started range: carry it through loops (it implements [`Carry`]) and end it with
/// [`range_end`]. A [`sentinel_token`] stands for "no range yet".
#[derive(Clone, Copy, Debug)]
pub struct RangeToken {
    event: Int32,
    is_none: Boolean,
    /// The kernel it belongs to.
    kernel: u32,
    /// The type code of the start's payload, which the end must match ([`ANY_PAYLOAD`] for a
    /// sentinel, whose carried copy cannot know it).
    payload: u32,
}

impl Carry for RangeToken {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        self.event.flatten(out);
        self.is_none.flatten(out);
    }
    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        RangeToken { event: self.event.rebuild(values), is_none: self.is_none.rebuild(values), ..*self }
    }
}

/// Declare the trace global when the kernel being built has IKET on.
pub(crate) fn init_iket() {
    let Some(capacity) = builder::with(|b| b.profiling.iket.as_ref().map(|i| i.options.capacity)) else {
        return;
    };
    assert!(capacity > 0, "cute: IketOptions::capacity must be positive");
    let words = HEADER_WORDS + RECORD_WORDS * capacity as usize;
    let buffer = crate::module_global::<u32>(Some(IKET_GLOBAL), words, crate::AddressSpace::Gmem, None, false);
    builder::with(|b| b.profiling.iket.as_mut().expect("iket is on").buffer = Some(buffer));
}

fn event_id(name: &str) -> u32 {
    builder::with(|b| {
        let events = &mut b.profiling.iket.as_mut().expect("iket is on").events;
        let at = match events.iter().position(|e| e == name) {
            Some(at) => at,
            None => {
                events.push(name.to_string());
                events.len() - 1
            }
        };
        FIRST_EVENT_ID + at as u32
    })
}

fn iket_on() -> bool {
    builder::with(|b| b.profiling.iket.is_some())
}

/// Append one record, from the lowest active lane of the warp.
#[track_caller]
fn emit(kind: u32, event: Int32, payload: Option<Payload>) {
    let (buffer, options) = builder::with(|b| {
        let i = b.profiling.iket.as_ref().expect("iket is on");
        (i.buffer.clone().expect("the trace global is declared at build start"), i.options)
    });
    let lead = (crate::lanemask_lt() & crate::activemask()).eq(0u32);
    if_(lead, || {
        let ts = match options.clock {
            IketClock::Globaltimer => crate::globaltimer(),
            IketClock::Clock64 => crate::clock64(),
        };
        let slot = crate::atomic_add(&buffer.ptr(), 1u32);
        let have_room = slot.lt(options.capacity);
        let (event, payload_bits) = (event, payload.map(|p| (p.code(), p.bits())));
        if_else(
            have_room,
            || {
                let at = slot.to::<i32>() * RECORD_WORDS as i32 + HEADER_WORDS as i32;
                let rec = buffer.ptr().offset(at);
                let word = |i: usize, v: Uint32| rec.offset(i as i32).store(v);
                let (ptype, bits) = payload_bits.unwrap_or_else(|| (0, Value::<i64>::constant(0)));
                word(0, ts.to::<u32>());
                word(1, (ts >> 32).to::<u32>());
                word(2, event.to::<u32>() | (kind << 16 | ptype << 20));
                word(3, crate::smid().to::<u32>());
                let (bi, gd, ti) = (crate::block_idx(), crate::grid_dim(), crate::linear_thread_idx());
                word(4, (bi.x + gd.x * (bi.y + gd.y * bi.z)).to::<u32>());
                word(5, (ti / 32).to::<u32>());
                word(6, bits.to::<u32>());
                word(7, (bits >> 32).to::<u32>());
            },
            || buffer.ptr().offset(1i32).store(1u32),
        );
    });
}

/// An instantaneous event named `event_name`. Nothing when IKET is off.
#[track_caller]
pub fn mark(event_name: &str, payload: Option<Payload>) {
    if !iket_on() {
        return;
    }
    let id = event_id(event_name);
    emit(KIND_MARK, Int32::constant(id as i32), payload);
}

/// Start a range named `event_name`; end it with [`range_end`] on the returned token.
#[track_caller]
pub fn range_start(event_name: &str, payload: Option<Payload>) -> RangeToken {
    let kernel = builder::with(|b| b.kernel);
    if !iket_on() {
        return RangeToken { event: Int32::constant(0), is_none: Boolean::constant(true), kernel, payload: 0 };
    }
    let id = event_id(event_name);
    let event = Int32::constant(id as i32);
    emit(KIND_START, event, payload);
    RangeToken { event, is_none: Boolean::constant(false), kernel, payload: payload_code(&payload) }
}

/// The token of a range that has not started, for a loop that ends the previous iteration's
/// range at its top (`range_end(token)`, then `token = range_start(..)` after the work).
/// [`range_end`] on it does nothing at run time.
#[track_caller]
pub fn sentinel_token(event_name: &str) -> RangeToken {
    let kernel = builder::with(|b| b.kernel);
    if !iket_on() {
        return RangeToken { event: Int32::constant(0), is_none: Boolean::constant(true), kernel, payload: 0 };
    }
    let id = event_id(event_name);
    RangeToken { event: Int32::constant(id as i32), is_none: Boolean::constant(true), kernel, payload: ANY_PAYLOAD }
}

/// End `token`'s range, with the same payload shape its start had. A no-op at run time for a
/// sentinel token. Panics for a token of another kernel or a payload of another type.
#[track_caller]
pub fn range_end(token: RangeToken, payload: Option<Payload>) {
    let kernel = builder::with(|b| b.kernel);
    assert!(token.kernel == kernel, "cute: range_end: the token was made while building another kernel");
    if !iket_on() {
        return;
    }
    assert!(
        token.payload == ANY_PAYLOAD || token.payload == payload_code(&payload),
        "cute: range_end: the range started with a {} payload but ends with a {} payload; both must have the same type",
        payload_name(token.payload),
        payload_name(payload_code(&payload))
    );
    if_(!token.is_none, || emit(KIND_END, token.event, payload));
}

/// Push a range named `event_name` on the warp's stack; [`range_pop`] ends the innermost one.
#[track_caller]
pub fn range_push(event_name: &str, payload: Option<Payload>) {
    if !iket_on() {
        return;
    }
    let id = event_id(event_name);
    builder::with(|b| b.profiling.iket.as_mut().expect("iket is on").open_pushes.push(event_name.to_string()));
    emit(KIND_PUSH, Int32::constant(id as i32), payload);
}

/// End the range pushed last (recorded with the reserved event id 31). Panics with no push open.
#[track_caller]
pub fn range_pop() {
    if !iket_on() {
        return;
    }
    let had = builder::with(|b| b.profiling.iket.as_mut().expect("iket is on").open_pushes.pop().is_some());
    assert!(had, "cute: range_pop without a matching range_push");
    emit(KIND_POP, Int32::constant(POP_EVENT_ID as i32), None);
}

// ---------------------------------------------------------------------------------------
// IKET: the host side
// ---------------------------------------------------------------------------------------

/// What decodes the trace of a kernel.
#[derive(Clone, Debug)]
pub struct IketInfo {
    pub options: IketOptions,
    /// Event names; event id `32 + i` is `events[i]`.
    pub events: Vec<String>,
    pub dags: Vec<IketDag>,
}

/// What an event is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IketKind {
    Mark,
    Start,
    End,
    Push,
    Pop,
}

/// A decoded payload.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct IketPayload {
    /// `i8`, `u8`, ..., `f32` or `pointer`.
    pub ty: &'static str,
    /// The value as 64 bits (a float's bits in the low word).
    pub bits: u64,
}

impl IketPayload {
    /// The value as a number: integers sign- or zero-extended by their type, floats decoded.
    pub fn as_f64(&self) -> f64 {
        match self.ty {
            "f32" => f32::from_bits(self.bits as u32) as f64,
            "u64" | "pointer" => self.bits as f64,
            _ => self.bits as i64 as f64,
        }
    }

    fn json(&self) -> String {
        match self.ty {
            "f32" => format!("{}", f32::from_bits(self.bits as u32)),
            "u64" | "pointer" => self.bits.to_string(),
            _ => (self.bits as i64).to_string(),
        }
    }
}

/// One record of the trace.
#[derive(Clone, PartialEq, Debug)]
pub struct IketRecord {
    /// The clock's value: nanoseconds, cycles, or the simulator's instruction counter.
    pub ts: u64,
    pub event: String,
    pub kind: IketKind,
    pub sm: u32,
    /// The linear block index.
    pub cta: u32,
    pub warp: u32,
    pub payload: Option<IketPayload>,
}

/// A matched start/end or push/pop.
#[derive(Clone, PartialEq, Debug)]
pub struct IketSpan {
    pub name: String,
    pub sm: u32,
    pub cta: u32,
    pub warp: u32,
    pub start: u64,
    pub end: u64,
    pub start_payload: Option<IketPayload>,
    pub end_payload: Option<IketPayload>,
}

/// A decoded trace.
#[derive(Clone, Debug)]
pub struct IketTrace {
    /// Every record, in the order the warps reserved their slots.
    pub records: Vec<IketRecord>,
    /// Matched ranges (ends of one event pair with starts first-in first-out; pops pair with
    /// pushes last-in first-out), in the order they ended.
    pub spans: Vec<IketSpan>,
    /// Events that found no partner: an end without a start, a start or push never closed,
    /// a pop without a push.
    pub unmatched: Vec<String>,
    /// Records dropped because the buffer was full.
    pub dropped: u32,
    pub overflowed: bool,
    clock: IketClock,
    symbols: KernelSymbols,
    dags: Vec<IketDag>,
}

impl IketTrace {
    /// Decode the words of the trace global.
    pub fn decode(info: &IketInfo, symbols: &KernelSymbols, words: &[u32]) -> Result<IketTrace, String> {
        let capacity = info.options.capacity as usize;
        let need = HEADER_WORDS + RECORD_WORDS * capacity;
        if words.len() < need {
            return Err(format!("cute: the trace has {} words, the buffer holds {need}", words.len()));
        }
        let count = words[0] as usize;
        let kept = count.min(capacity);
        let mut records = Vec::with_capacity(kept);
        for i in 0..kept {
            let w = &words[HEADER_WORDS + i * RECORD_WORDS..][..RECORD_WORDS];
            let (id, kind, ptype) = (w[2] & 0xffff, (w[2] >> 16) & 0xf, (w[2] >> 20) & 0xf);
            let kind = match kind {
                KIND_MARK => IketKind::Mark,
                KIND_START => IketKind::Start,
                KIND_END => IketKind::End,
                KIND_PUSH => IketKind::Push,
                KIND_POP => IketKind::Pop,
                k => return Err(format!("cute: trace record {i} has kind {k}")),
            };
            let event = if id == POP_EVENT_ID {
                "pop".to_string()
            } else {
                id.checked_sub(FIRST_EVENT_ID)
                    .and_then(|n| info.events.get(n as usize))
                    .cloned()
                    .ok_or_else(|| format!("cute: trace record {i} has unknown event id {id}"))?
            };
            records.push(IketRecord {
                ts: w[0] as u64 | (w[1] as u64) << 32,
                event,
                kind,
                sm: w[3],
                cta: w[4],
                warp: w[5],
                payload: (ptype != 0)
                    .then(|| IketPayload { ty: payload_name(ptype), bits: w[6] as u64 | (w[7] as u64) << 32 }),
            });
        }
        let mut trace = IketTrace {
            records,
            spans: Vec::new(),
            unmatched: Vec::new(),
            dropped: count.saturating_sub(capacity) as u32,
            overflowed: words[1] != 0,
            clock: info.options.clock,
            symbols: symbols.clone(),
            dags: info.dags.clone(),
        };
        trace.pair();
        Ok(trace)
    }

    fn pair(&mut self) {
        type Open = std::collections::VecDeque<usize>;
        let mut ranges: HashMap<(u32, u32, String), Open> = HashMap::new();
        let mut stacks: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        let span = |records: &[IketRecord], s: usize, e: usize| IketSpan {
            name: records[s].event.clone(),
            sm: records[s].sm,
            cta: records[s].cta,
            warp: records[s].warp,
            start: records[s].ts,
            end: records[e].ts,
            start_payload: records[s].payload,
            end_payload: records[e].payload,
        };
        let describe = |r: &IketRecord| format!("`{}` of CTA {} warp {}", r.event, r.cta, r.warp);
        for i in 0..self.records.len() {
            let r = &self.records[i];
            let (cta, warp) = (r.cta, r.warp);
            match r.kind {
                IketKind::Mark => {}
                IketKind::Start => ranges.entry((cta, warp, r.event.clone())).or_default().push_back(i),
                IketKind::End => match ranges.get_mut(&(cta, warp, r.event.clone())).and_then(Open::pop_front) {
                    Some(s) => self.spans.push(span(&self.records, s, i)),
                    None => self.unmatched.push(format!("range end without a start: {}", describe(r))),
                },
                IketKind::Push => stacks.entry((cta, warp)).or_default().push(i),
                IketKind::Pop => match stacks.get_mut(&(cta, warp)).and_then(Vec::pop) {
                    Some(s) => self.spans.push(span(&self.records, s, i)),
                    None => self.unmatched.push(format!("range_pop without a push: CTA {cta} warp {warp}")),
                },
            }
        }
        let mut open: Vec<usize> = ranges.into_values().flatten().chain(stacks.into_values().flatten()).collect();
        open.sort_unstable();
        for i in open {
            let kind = if self.records[i].kind == IketKind::Start { "range start" } else { "range_push" };
            self.unmatched.push(format!("{kind} never ended: {}", describe(&self.records[i])));
        }
    }

    /// The records of one warp, in program order.
    pub fn warp_records(&self, cta: u32, warp: u32) -> Vec<&IketRecord> {
        self.records.iter().filter(|r| r.cta == cta && r.warp == warp).collect()
    }

    /// Chrome-trace JSON (loads in Perfetto and `chrome://tracing`): spans are complete events,
    /// marks instants; pid is the CTA, tid the warp, args hold the SM and payloads. The kernel's
    /// symbols and the DAGs are metadata. Timestamps start at 0, in microseconds for the
    /// nanosecond clock and in raw ticks for the others.
    pub fn to_chrome_json(&self) -> String {
        let t0 = self.records.iter().map(|r| r.ts).min().unwrap_or(0);
        let unit = if self.clock == IketClock::Globaltimer { 1000.0 } else { 1.0 };
        let us = |ts: u64| (ts - t0) as f64 / unit;
        let args = |sm: u32, a: Option<IketPayload>, b: Option<IketPayload>| {
            let mut s = format!("{{\"sm\": {sm}");
            if let Some(p) = a {
                let _ = write!(s, ", \"payload\": {}, \"payload_type\": {}", p.json(), json_str(p.ty));
            }
            if let Some(p) = b {
                let _ = write!(s, ", \"end_payload\": {}", p.json());
            }
            s.push('}');
            s
        };
        let mut events = Vec::new();
        let mut warps: Vec<(u32, u32)> = self.records.iter().map(|r| (r.cta, r.warp)).collect();
        warps.sort_unstable();
        warps.dedup();
        for cta in warps.iter().map(|w| w.0).collect::<std::collections::BTreeSet<_>>() {
            events.push(format!(
                "{{\"ph\": \"M\", \"name\": \"process_name\", \"pid\": {cta}, \"args\": {{\"name\": \"CTA {cta}\"}}}}"
            ));
        }
        for (cta, warp) in &warps {
            events.push(format!(
                "{{\"ph\": \"M\", \"name\": \"thread_name\", \"pid\": {cta}, \"tid\": {warp}, \"args\": {{\"name\": \"warp {warp}\"}}}}"
            ));
        }
        for s in &self.spans {
            events.push(format!(
                "{{\"ph\": \"X\", \"name\": {}, \"cat\": \"iket\", \"pid\": {}, \"tid\": {}, \"ts\": {}, \"dur\": {}, \"args\": {}}}",
                json_str(&s.name),
                s.cta,
                s.warp,
                us(s.start),
                us(s.end) - us(s.start),
                args(s.sm, s.start_payload, s.end_payload)
            ));
        }
        for r in self.records.iter().filter(|r| r.kind == IketKind::Mark) {
            events.push(format!(
                "{{\"ph\": \"i\", \"s\": \"t\", \"name\": {}, \"cat\": \"iket\", \"pid\": {}, \"tid\": {}, \"ts\": {}, \"args\": {}}}",
                json_str(&r.event),
                r.cta,
                r.warp,
                us(r.ts),
                args(r.sm, r.payload, None)
            ));
        }
        let dags: Vec<String> = self
            .dags
            .iter()
            .map(|d| {
                let edges: Vec<String> = d
                    .edges
                    .iter()
                    .map(|e| {
                        format!(
                            "{{\"src\": {}, \"dst\": {}, \"label\": {}, \"via\": {}}}",
                            json_str(&e.src),
                            json_str(&e.dst),
                            e.label.as_deref().map_or("null".into(), json_str),
                            e.via.as_deref().map_or("null".into(), json_str)
                        )
                    })
                    .collect();
                format!("{}: [{}]", json_str(&d.name), edges.join(", "))
            })
            .collect();
        format!(
            "{{\"traceEvents\": [\n{}\n], \"displayTimeUnit\": \"ns\", \"metadata\": {{\"symbols\": {}, \"dags\": {{{}}}, \
             \"dropped\": {}}}}}\n",
            events.join(",\n"),
            self.symbols.to_json(&[]).trim_end(),
            dags.join(", "),
            self.dropped
        )
    }

    /// Write [`to_chrome_json`](Self::to_chrome_json) to `path`.
    pub fn write_chrome_trace(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_chrome_json())
    }
}
