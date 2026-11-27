// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Names and semantics follow CuTe DSL 4.8.1's
// cutlass.experimental.task_scheduling.resources (MemoryResource, StageInfo,
// TaskLocalVariable, producer_work / consumer_work; NVIDIA EULA); the code is
// written independently.

//! Resources: named memory with producer and consumer work methods,
//! optionally handed over through a pipeline.

use std::any::{Any, TypeId};
use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::control::{Carry, Var};
use crate::ir::{DType, ValueId};
use crate::pipeline::{
    Pipeline, PipelineAsync, PipelineAsyncOptions, PipelineAsyncUmma, PipelineBarriers, PipelineClcFetchAsync,
    PipelineClcFetchAsyncOptions, PipelineCpAsync, PipelineOp, PipelineState, PipelineTmaAsync,
    PipelineTmaAsyncOptions, PipelineTmaUmma, PipelineTmaUmmaOptions, PipelineUmmaAsync, PipelineUmmaOptions,
    PipelineUserType, make_pipeline_state,
};
use crate::tensor::Ptr;
use crate::utils::WorkTileInfo;
use crate::value::{Boolean, Int32, IntoValue, Numeric, Value};

use super::config::PipelineConfig;
use super::enums::{PipelineType, ScheduleStage, WorkAttr};
use super::pipeline::{TsPipeline, TsPipelineOptions, task_warp_leader};

// ---------------------------------------------------------------------------
// Task-local variables and the values work passes around
// ---------------------------------------------------------------------------

/// Types a [`TaskLocalVariable`] can hold: anything carried through loops
/// ([`Carry`]) that is cheap to clone. `SCALAR` marks the types a
/// `when_true` / `when_false` guard can test (numbers and booleans).
pub trait LocalValue: Carry + Clone + 'static {
    /// Whether a guard can test a value of this type for non-zero.
    const SCALAR: bool = false;
    /// A short name of the type for listings and diagnostics.
    fn type_label() -> String;

    /// The value as a guard: non-zero for numbers and booleans, `None` for
    /// types a guard cannot test.
    fn as_condition(&self) -> Option<Boolean> {
        None
    }
}

impl<T: Numeric> LocalValue for Value<T> {
    const SCALAR: bool = true;
    fn type_label() -> String {
        T::DTYPE.to_string()
    }

    #[track_caller]
    fn as_condition(&self) -> Option<Boolean> {
        Some(self.ne(Value::<T>::constant(T::from_bits(0))))
    }
}

impl<T: Numeric> LocalValue for crate::Vector<T> {
    fn type_label() -> String {
        format!("Vector[{}]", T::DTYPE)
    }
}

/// A compile-time integer argument of a work call (CuTe DSL's
/// `cutlass.Constexpr[int]` keyword arguments): declared with
/// [`Work::constexpr_input`], given in the schedule with
/// [`ArgSource::constexpr`](super::ArgSource::constexpr) and read in the
/// work with [`WorkArgs::constexpr`]. The value is part of the schedule, so
/// each call site can pass its own (a subtile index in an unrolled loop,
/// say).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Constexpr(pub i64);

impl Constexpr {
    /// The label of a constexpr input in listings and diagnostics.
    pub const TYPE_LABEL: &'static str = "Constexpr[int]";
}

impl LocalValue for WorkTileInfo {
    fn type_label() -> String {
        "WorkTileInfo".into()
    }
}

static NEXT_VAR_ID: AtomicU64 = AtomicU64::new(1);

/// Makes storage for a variable in a [`TaskLocals`] from its default;
/// false for a variable without one.
type DeclareFn = Rc<dyn Fn(&mut TaskLocals, &VarRef) -> bool>;

/// A [`TaskLocalVariable`] with its type erased: what schedules, routes and
/// diagnostics refer to. Two refs are equal when they name the same
/// variable.
#[derive(Clone)]
pub struct VarRef {
    id: u64,
    name: Rc<str>,
    type_label: Rc<str>,
    type_id: TypeId,
    scalar: bool,
    has_default: bool,
    declare: DeclareFn,
}

impl fmt::Debug for VarRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VarRef")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("type_label", &self.type_label)
            .field("scalar", &self.scalar)
            .finish()
    }
}

impl VarRef {
    /// The variable's name (its slot name).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Its type, as [`LocalValue::type_label`] names it.
    pub fn type_label(&self) -> &str {
        &self.type_label
    }

    /// Whether a guard can test it.
    pub fn is_scalar(&self) -> bool {
        self.scalar
    }

    /// Whether it has a default (storage can be made for it without a
    /// first value).
    pub fn has_default(&self) -> bool {
        self.has_default
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn type_id(&self) -> TypeId {
        self.type_id
    }
}

impl PartialEq for VarRef {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for VarRef {}

impl std::hash::Hash for VarRef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

/// A value that lives for a whole task, written by consumer work that
/// returns it and read by later work (CuTe DSL's `TaskLocalVariable(dtype,
/// default, docs)`). The handle is typed by the value it holds (an
/// [`Int16`](crate::Int16), a [`WorkTileInfo`], ...); the storage is not
/// part of it: [`TaskLocals`] (and the [`TaskManager`](super::TaskManager)) allocates one [`Var`] per variable, starting at the default.
///
/// The default is a closure, so that a variable can be declared outside a
/// kernel (its constants are only built when storage is made);
/// [`uninitialized`](Self::uninitialized) declares one without a default,
/// whose storage must be given a first value.
pub struct TaskLocalVariable<C: LocalValue> {
    r: VarRef,
    docs: Rc<str>,
    default: Option<Rc<dyn Fn() -> C>>,
}

impl<C: LocalValue> Clone for TaskLocalVariable<C> {
    fn clone(&self) -> Self {
        TaskLocalVariable { r: self.r.clone(), docs: self.docs.clone(), default: self.default.clone() }
    }
}

impl<C: LocalValue> fmt::Debug for TaskLocalVariable<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TaskLocalVariable({}: {})", self.r.name, self.r.type_label)
    }
}

impl<C: LocalValue> TaskLocalVariable<C> {
    fn make(name: &str, docs: &str, default: Option<Rc<dyn Fn() -> C>>) -> Self {
        let declare: DeclareFn = {
            let default = default.clone();
            Rc::new(move |locals: &mut TaskLocals, r: &VarRef| match &default {
                Some(d) => {
                    locals.declare_raw(r.clone(), Some(d.clone()), d());
                    true
                }
                None => false,
            })
        };
        let r = VarRef {
            id: NEXT_VAR_ID.fetch_add(1, Ordering::Relaxed),
            name: name.into(),
            type_label: C::type_label().into(),
            type_id: TypeId::of::<C>(),
            scalar: C::SCALAR,
            has_default: default.is_some(),
            declare,
        };
        TaskLocalVariable { r, docs: docs.into(), default }
    }

    /// A variable called `name` whose storage starts at `default()`.
    pub fn new(name: &str, default: impl Fn() -> C + 'static, docs: &str) -> Self {
        Self::make(name, docs, Some(Rc::new(default)))
    }

    /// A variable without a default (CuTe DSL's
    /// `TaskLocalVariable.uninitialized()`).
    pub fn uninitialized(name: &str, docs: &str) -> Self {
        Self::make(name, docs, None)
    }

    /// The variable's name.
    pub fn name(&self) -> &str {
        &self.r.name
    }

    /// Its documentation.
    pub fn docs(&self) -> &str {
        &self.docs
    }

    /// The type-erased reference.
    pub fn var_ref(&self) -> &VarRef {
        &self.r
    }

    /// The default value, built now (in a kernel), or `None` for an
    /// uninitialised variable.
    pub fn default_value(&self) -> Option<C> {
        self.default.as_ref().map(|f| f())
    }
}

impl<T: Numeric> TaskLocalVariable<Value<T>> {
    /// A number variable starting at the constant `default`.
    pub fn scalar(name: &str, default: T, docs: &str) -> Self {
        Self::new(name, move || Value::constant(default), docs)
    }
}

/// The named inputs of one work call: values keyed by the work's input
/// names (CuTe DSL passes them as keyword arguments).
#[derive(Clone, Default)]
pub struct WorkArgs {
    values: HashMap<String, Rc<dyn Any>>,
}

impl WorkArgs {
    /// No inputs.
    pub fn new() -> WorkArgs {
        WorkArgs::default()
    }

    /// These inputs plus `name = value`.
    pub fn with<C: LocalValue>(mut self, name: &str, value: C) -> WorkArgs {
        self.values.insert(name.to_string(), Rc::new(value));
        self
    }

    /// The input `name`, or `None` if it is missing or of another type.
    pub fn try_get<C: LocalValue>(&self, name: &str) -> Option<C> {
        self.values.get(name)?.downcast_ref::<C>().cloned()
    }

    /// The input `name`. Panics if it is missing or of another type.
    #[track_caller]
    pub fn get<C: LocalValue>(&self, name: &str) -> C {
        match self.values.get(name) {
            None => panic!("cute: task_scheduling: work input `{name}` was not passed"),
            Some(v) => v
                .downcast_ref::<C>()
                .cloned()
                .unwrap_or_else(|| panic!("cute: task_scheduling: work input `{name}` is not a {}", C::type_label())),
        }
    }

    /// These inputs plus the compile-time `name = value`.
    pub fn with_constexpr(mut self, name: &str, value: i64) -> WorkArgs {
        self.values.insert(name.to_string(), Rc::new(Constexpr(value)));
        self
    }

    /// The compile-time input `name` (declared with
    /// [`Work::constexpr_input`]). Panics if it is missing or not a
    /// constexpr.
    #[track_caller]
    pub fn constexpr(&self, name: &str) -> i64 {
        match self.values.get(name).map(|v| v.downcast_ref::<Constexpr>()) {
            None => panic!("cute: task_scheduling: work input `{name}` was not passed"),
            Some(None) => panic!("cute: task_scheduling: work input `{name}` is not a compile-time constant"),
            Some(Some(c)) => c.0,
        }
    }

    fn contains(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }

    /// These inputs plus `name = value`, the value type-erased.
    pub(crate) fn with_raw(mut self, name: &str, value: Rc<dyn Any>) -> WorkArgs {
        self.values.insert(name.to_string(), value);
        self
    }

    fn type_matches(&self, name: &str, type_id: TypeId) -> bool {
        self.values.get(name).is_some_and(|v| (**v).type_id() == type_id)
    }
}

type StoreFn = fn(&dyn Any, &dyn Any);

fn store_into<C: LocalValue>(var: &dyn Any, value: &dyn Any) {
    let var = var.downcast_ref::<Var<C>>().expect("task_scheduling: storage of another type");
    let value = value.downcast_ref::<C>().expect("task_scheduling: result of another type");
    var.set(value.clone());
}

/// What one consumer work call returns: a value for each of the
/// variables it declares (`returns=`).
#[derive(Clone, Default)]
pub struct WorkResults {
    values: Vec<(VarRef, Rc<dyn Any>, StoreFn)>,
}

impl WorkResults {
    /// No results (producer work, and consumer work that returns nothing).
    pub fn none() -> WorkResults {
        WorkResults::default()
    }

    /// The single result `var = value`.
    pub fn of<C: LocalValue>(var: &TaskLocalVariable<C>, value: C) -> WorkResults {
        WorkResults::none().and(var, value)
    }

    /// These results plus `var = value`.
    pub fn and<C: LocalValue>(mut self, var: &TaskLocalVariable<C>, value: C) -> WorkResults {
        self.values.retain(|(r, ..)| r != &var.r);
        self.values.push((var.r.clone(), Rc::new(value), store_into::<C>));
        self
    }

    /// The value returned for `var`, if any.
    pub fn try_get<C: LocalValue>(&self, var: &TaskLocalVariable<C>) -> Option<C> {
        let (_, v, _) = self.values.iter().find(|(r, ..)| r == &var.r)?;
        v.downcast_ref::<C>().cloned()
    }

    /// The value returned for `var`. Panics if the work returned none.
    #[track_caller]
    pub fn get<C: LocalValue>(&self, var: &TaskLocalVariable<C>) -> C {
        self.try_get(var).unwrap_or_else(|| panic!("cute: task_scheduling: the work returned no `{}`", var.name()))
    }

    /// The variables returned.
    pub fn vars(&self) -> impl Iterator<Item = &VarRef> {
        self.values.iter().map(|(r, ..)| r)
    }
}

/// Resets a variable's storage (a boxed [`Var`]) to its default.
type ResetFn = Box<dyn Fn(&dyn Any)>;

/// One variable's storage: the boxed [`Var`] and type-erased accessors.
struct Slot {
    storage: Box<dyn Any>,
    reset: ResetFn,
    load: fn(&dyn Any) -> Rc<dyn Any>,
    test: fn(&dyn Any) -> Option<Boolean>,
}

fn load_from<C: LocalValue>(storage: &dyn Any) -> Rc<dyn Any> {
    Rc::new(storage.downcast_ref::<Var<C>>().expect("task_scheduling: storage of another type").get())
}

fn test_of<C: LocalValue>(storage: &dyn Any) -> Option<Boolean> {
    storage.downcast_ref::<Var<C>>().expect("task_scheduling: storage of another type").get().as_condition()
}

/// Storage for task-local variables in a hand-written kernel: one [`Var`]
/// per declared variable, so that values written in one loop iteration or
/// branch are read in the next. The [`TaskManager`](super::TaskManager)
/// allocates the same storage for the variables its tasks use.
#[derive(Default)]
pub struct TaskLocals {
    vars: HashMap<u64, Slot>,
}

impl TaskLocals {
    /// No storage yet.
    pub fn new() -> TaskLocals {
        TaskLocals::default()
    }

    /// Storage for `var`, starting at its default (in a kernel). Panics
    /// for an uninitialised variable; use [`declare_with`](Self::declare_with).
    #[track_caller]
    pub fn declare<C: LocalValue>(&mut self, var: &TaskLocalVariable<C>) {
        let init = var.default_value().unwrap_or_else(|| {
            panic!("cute: task_scheduling: `{}` has no default; give it a first value with declare_with", var.name())
        });
        self.declare_with(var, init);
    }

    /// Storage for `var`, starting at `init`.
    #[track_caller]
    pub fn declare_with<C: LocalValue>(&mut self, var: &TaskLocalVariable<C>, init: C) {
        self.declare_raw(var.r.clone(), var.default.clone(), init);
    }

    #[track_caller]
    fn declare_raw<C: LocalValue>(&mut self, r: VarRef, default: Option<Rc<dyn Fn() -> C>>, init: C) {
        let reset: ResetFn = Box::new(move |storage| {
            if let (Some(d), Some(v)) = (&default, storage.downcast_ref::<Var<C>>()) {
                v.set(d());
            }
        });
        let slot = Slot { storage: Box::new(Var::new(init)), reset, load: load_from::<C>, test: test_of::<C> };
        self.vars.insert(r.id, slot);
    }

    /// Storage for the type-erased `var`, starting at its default; false
    /// (and nothing declared) for a variable without one.
    #[track_caller]
    pub(crate) fn declare_ref(&mut self, var: &VarRef) -> bool {
        let declare = var.declare.clone();
        declare(self, var)
    }

    #[track_caller]
    fn slot(&self, r: &VarRef) -> &Slot {
        self.vars
            .get(&r.id)
            .unwrap_or_else(|| panic!("cute: task_scheduling: `{}` has no storage; declare it first", r.name))
    }

    /// The current value of the type-erased `var`.
    #[track_caller]
    pub(crate) fn load_ref(&self, var: &VarRef) -> Rc<dyn Any> {
        let slot = self.slot(var);
        (slot.load)(slot.storage.as_ref())
    }

    /// The current value of the type-erased `var` as a guard (`None` for a
    /// type a guard cannot test).
    #[track_caller]
    pub(crate) fn condition_ref(&self, var: &VarRef) -> Option<Boolean> {
        let slot = self.slot(var);
        (slot.test)(slot.storage.as_ref())
    }

    #[track_caller]
    fn storage<C: LocalValue>(&self, var: &TaskLocalVariable<C>) -> &Var<C> {
        self.slot(&var.r).storage.downcast_ref::<Var<C>>().expect("task_scheduling: storage of another type")
    }

    /// The current value of `var`.
    #[track_caller]
    pub fn get<C: LocalValue>(&self, var: &TaskLocalVariable<C>) -> C {
        self.storage(var).get()
    }

    /// Write `value` to `var`.
    #[track_caller]
    pub fn set<C: LocalValue>(&self, var: &TaskLocalVariable<C>, value: C) {
        self.storage(var).set(value);
    }

    /// Write every result of a work call to its variable's storage.
    #[track_caller]
    pub fn store(&self, results: &WorkResults) {
        for (r, value, store) in &results.values {
            store(self.slot(r).storage.as_ref(), value.as_ref());
        }
    }

    /// Reset every variable with a default to it.
    #[track_caller]
    pub fn reset_to_defaults(&self) {
        for slot in self.vars.values() {
            (slot.reset)(slot.storage.as_ref());
        }
    }

    /// Whether `var` has storage.
    pub fn contains(&self, var: &VarRef) -> bool {
        self.vars.contains_key(&var.id)
    }
}

// ---------------------------------------------------------------------------
// StageInfo
// ---------------------------------------------------------------------------

/// Framework state shared by the resources of a kernel (CuTe DSL's
/// `ResourceContext`): the base of the unified shared-memory block of the
/// TaskManager's [`SmemAllocator`](super::SmemAllocator) and the
/// shared-memory word holding the address of its
/// [`TmemAllocator`](super::TmemAllocator)'s columns (`None` without them).
/// Every work call the TaskManager makes sees it in
/// [`StageInfo::context`].
#[derive(Clone, Debug, Default)]
pub struct ResourceContext {
    pub smem_base: Option<Ptr<u8>>,
    pub tmem_ptr_i32: Option<Ptr<i32>>,
}

/// What every work call is told about where it runs (CuTe DSL's
/// `StageInfo`): the domain loop's index (`loop_offset`) and bounds, the
/// pipeline stage (`stage_idx`, `phase` and its mbarrier,
/// [`barrier`](Self::barrier)) for a pipelined resource, the current work
/// tile in a work-tile loop, the framework [`context`](ResourceContext) and
/// the label of the schedule step.
///
/// Carried through loops ([`Carry`]): the values are flattened; the
/// barrier array, context, label and stage count are taken from the
/// template (the barrier of the stage is recomputed from `stage_idx`).
#[derive(Clone, Debug)]
pub struct StageInfo {
    /// The domain loop's index.
    pub loop_offset: Int32,
    /// The domain loop's bounds, `range(loop_start, loop_end, loop_step)`.
    pub loop_start: Int32,
    pub loop_end: Int32,
    pub loop_step: Int32,
    /// The pipeline stage, for a pipelined resource.
    pub stage_idx: Option<Int32>,
    /// The phase of that stage.
    pub phase: Option<Int32>,
    /// The work tile, in a work-tile loop.
    pub work_tile: Option<WorkTileInfo>,
    /// Framework state.
    pub context: ResourceContext,
    /// The label of the schedule step that called the work.
    pub label: Option<String>,
    /// Producer stages in flight (delayed-commit schedules; 0 otherwise).
    pub num_active_stages: usize,
    barrier_base: Option<Ptr<u64>>,
}

impl StageInfo {
    /// The info of iteration `loop_offset` of a one-iteration domain
    /// (`loop_start = loop_offset`, `loop_end = loop_offset + 1`, step 1),
    /// with no stage or tile.
    #[track_caller]
    pub fn new(loop_offset: impl IntoValue<i32>) -> StageInfo {
        let loop_offset = loop_offset.into_value();
        StageInfo {
            loop_offset,
            loop_start: loop_offset,
            loop_end: loop_offset + 1,
            loop_step: Int32::constant(1),
            stage_idx: None,
            phase: None,
            work_tile: None,
            context: ResourceContext::default(),
            label: None,
            num_active_stages: 0,
            barrier_base: None,
        }
    }

    /// With the domain loop's bounds.
    #[track_caller]
    pub fn with_bounds(
        mut self,
        start: impl IntoValue<i32>,
        end: impl IntoValue<i32>,
        step: impl IntoValue<i32>,
    ) -> StageInfo {
        self.loop_start = start.into_value();
        self.loop_end = end.into_value();
        self.loop_step = step.into_value();
        self
    }

    /// With the stage of `state` and the barrier array (indexed by stage)
    /// the work reports to.
    pub fn with_stage(mut self, state: &PipelineState, barrier_base: Option<Ptr<u64>>) -> StageInfo {
        self.stage_idx = Some(state.index());
        self.phase = Some(state.phase());
        self.barrier_base = barrier_base;
        self
    }

    /// With the current work tile.
    pub fn with_work_tile(mut self, work_tile: WorkTileInfo) -> StageInfo {
        self.work_tile = Some(work_tile);
        self
    }

    /// With the framework context.
    pub fn with_context(mut self, context: ResourceContext) -> StageInfo {
        self.context = context;
        self
    }

    /// With the step's label.
    pub fn with_label(mut self, label: &str) -> StageInfo {
        self.label = Some(label.to_string());
        self
    }

    /// The mbarrier of the current stage (the full barrier on the producer
    /// side, the empty one on the consumer side), for a pipelined resource.
    #[track_caller]
    pub fn barrier(&self) -> Option<Ptr<u64>> {
        match (&self.barrier_base, self.stage_idx) {
            (Some(base), Some(idx)) => Some(base.offset(idx)),
            _ => None,
        }
    }
}

impl Carry for StageInfo {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        (self.loop_offset, self.loop_start, self.loop_end, self.loop_step).flatten(out);
        for v in [self.stage_idx, self.phase].into_iter().flatten() {
            v.flatten(out);
        }
        if let Some(t) = &self.work_tile {
            t.flatten(out);
        }
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        let (loop_offset, loop_start, loop_end, loop_step) =
            (self.loop_offset, self.loop_start, self.loop_end, self.loop_step).rebuild(values);
        let stage_idx = self.stage_idx.map(|v| v.rebuild(values));
        let phase = self.phase.map(|v| v.rebuild(values));
        let work_tile = self.work_tile.as_ref().map(|t| t.rebuild(values));
        StageInfo { loop_offset, loop_start, loop_end, loop_step, stage_idx, phase, work_tile, ..self.clone() }
    }

    fn structure_mismatch(&self, other: &Self) -> Option<String> {
        let shape = |s: &StageInfo| (s.stage_idx.is_some(), s.phase.is_some(), s.work_tile.is_some());
        (shape(self) != shape(other)).then(|| {
            format!(
                "a StageInfo with (stage, phase, work tile) present {:?} became one with {:?}",
                shape(self),
                shape(other)
            )
        })
    }
}

// ---------------------------------------------------------------------------
// Work declarations
// ---------------------------------------------------------------------------

/// Which side of a resource a work method belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    Producer,
    Consumer,
}

/// A declared input of a work method.
#[derive(Clone, Debug)]
pub struct WorkInput {
    name: String,
    type_label: String,
    type_id: TypeId,
}

impl WorkInput {
    /// The input's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Its type.
    pub fn type_label(&self) -> &str {
        &self.type_label
    }

    pub(crate) fn type_id(&self) -> TypeId {
        self.type_id
    }
}

/// The body of a work method: the stage info and the named inputs in, the
/// declared results out.
pub type WorkFn = Rc<dyn Fn(&StageInfo, &WorkArgs) -> WorkResults>;

/// A work method of a resource (CuTe DSL's `@producer_work` and
/// `@consumer_work(returns=..., work_attrs=...)`), made with [`Work`].
#[derive(Clone)]
pub struct WorkDecl {
    name: String,
    side: Side,
    attrs: WorkAttr,
    inputs: Vec<WorkInput>,
    returns: Vec<VarRef>,
    body: WorkFn,
}

impl fmt::Debug for WorkDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkDecl")
            .field("name", &self.name)
            .field("side", &self.side)
            .field("attrs", &self.attrs)
            .field("inputs", &self.inputs.iter().map(|i| &i.name).collect::<Vec<_>>())
            .field("returns", &self.returns.iter().map(|r| r.name()).collect::<Vec<_>>())
            .finish()
    }
}

impl WorkDecl {
    /// The method's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Its side.
    pub fn side(&self) -> Side {
        self.side
    }

    /// Its attributes.
    pub fn attrs(&self) -> WorkAttr {
        self.attrs
    }

    /// The schedule stage a call of it is.
    pub fn stage(&self) -> ScheduleStage {
        match (self.side, self.attrs.is_auxiliary()) {
            (Side::Producer, false) => ScheduleStage::ProducerWork,
            (Side::Producer, true) => ScheduleStage::ProducerAuxWork,
            (Side::Consumer, false) => ScheduleStage::ConsumerWork,
            (Side::Consumer, true) => ScheduleStage::ConsumerAuxWork,
        }
    }

    /// Its declared inputs.
    pub fn inputs(&self) -> &[WorkInput] {
        &self.inputs
    }

    /// The variables it returns.
    pub fn returns(&self) -> &[VarRef] {
        &self.returns
    }
}

/// A builder of [`WorkDecl`]s.
///
/// ```
/// use cute::prelude::*;
/// use cute::task_scheduling::{TaskLocalVariable, Work, WorkResults};
///
/// let item = TaskLocalVariable::<Int16>::scalar("item", 0, "the element of this iteration");
/// let get_item = Work::consumer("get_item").returns(&item).run({
///     let item = item.clone();
///     move |si, _| WorkResults::of(&item, si.loop_offset.to::<i16>())
/// });
/// let set_item = Work::producer("set_item").input::<Int16>("data").run(|_, _| WorkResults::none());
/// assert_eq!(get_item.returns()[0].name(), "item");
/// assert_eq!(set_item.inputs()[0].name(), "data");
/// ```
pub struct Work {
    name: String,
    side: Side,
    attrs: WorkAttr,
    inputs: Vec<WorkInput>,
    returns: Vec<VarRef>,
}

impl Work {
    /// Producer work called `name` (CuTe DSL's `@producer_work`).
    pub fn producer(name: &str) -> Work {
        Work { name: name.to_string(), side: Side::Producer, attrs: WorkAttr::NONE, inputs: vec![], returns: vec![] }
    }

    /// Consumer work called `name` (CuTe DSL's `@consumer_work`).
    pub fn consumer(name: &str) -> Work {
        Work { side: Side::Consumer, ..Work::producer(name) }
    }

    /// Marked auxiliary (`work_attrs=WorkAttr.AUXILIARY`).
    pub fn auxiliary(mut self) -> Work {
        self.attrs = WorkAttr::AUXILIARY;
        self
    }

    /// With the attributes `attrs`.
    pub fn attrs(mut self, attrs: WorkAttr) -> Work {
        self.attrs = attrs;
        self
    }

    /// Taking the named input `name` of type `C`.
    pub fn input<C: LocalValue>(mut self, name: &str) -> Work {
        self.inputs.push(WorkInput { name: name.to_string(), type_label: C::type_label(), type_id: TypeId::of::<C>() });
        self
    }

    /// Taking the compile-time integer input `name` (a
    /// `cutlass.Constexpr[int]` keyword argument), which each call site of
    /// the schedule gives as a constant ([`Constexpr`]).
    pub fn constexpr_input(mut self, name: &str) -> Work {
        self.inputs.push(WorkInput {
            name: name.to_string(),
            type_label: Constexpr::TYPE_LABEL.to_string(),
            type_id: TypeId::of::<Constexpr>(),
        });
        self
    }

    /// Returning `var` (`returns=`).
    pub fn returns<C: LocalValue>(mut self, var: &TaskLocalVariable<C>) -> Work {
        self.returns.push(var.r.clone());
        self
    }

    /// With the body `f`.
    pub fn run(self, f: impl Fn(&StageInfo, &WorkArgs) -> WorkResults + 'static) -> WorkDecl {
        WorkDecl {
            name: self.name,
            side: self.side,
            attrs: self.attrs,
            inputs: self.inputs,
            returns: self.returns,
            body: Rc::new(f),
        }
    }
}

// ---------------------------------------------------------------------------
// Memory requirements (for the allocators)
// ---------------------------------------------------------------------------

/// A block of shared memory a resource asks the shared-memory allocator
/// for (CuTe DSL's `SmemAllocation`): `count` elements of `dtype` at an
/// `alignment`-byte boundary (a power of two).
///
/// Its byte offset in the kernel's unified shared-memory block is set by
/// [`SmemAllocator::compute_layout`](super::SmemAllocator::compute_layout)
/// and shared by every clone, so a resource keeps a clone and reads
/// [`offset`](Self::offset) (or [`ptr`](Self::ptr)) in its work.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SmemAllocation {
    pub name: String,
    pub dtype: DType,
    pub count: usize,
    pub alignment: usize,
    offset: Rc<Cell<Option<usize>>>,
}

impl SmemAllocation {
    /// `count` elements of `T`, `alignment`-byte aligned.
    pub fn new<T: Numeric>(name: &str, count: usize, alignment: usize) -> SmemAllocation {
        SmemAllocation { name: name.to_string(), dtype: T::DTYPE, count, alignment, offset: Rc::default() }
    }

    /// `size` bytes, 128-byte aligned (CuTe DSL's `SmemAllocation(name,
    /// size_bytes)`, whose default alignment suits TMA).
    pub fn bytes(name: &str, size: usize) -> SmemAllocation {
        SmemAllocation::new::<u8>(name, size, 128)
    }

    /// Its size in bytes.
    pub fn size_bytes(&self) -> usize {
        (self.count * self.dtype.bits()).div_ceil(8)
    }

    /// Its byte offset in the unified block, once laid out.
    pub fn try_offset(&self) -> Option<usize> {
        self.offset.get()
    }

    /// Its byte offset in the unified block. Panics before
    /// [`compute_layout`](super::SmemAllocator::compute_layout).
    #[track_caller]
    pub fn offset(&self) -> usize {
        self.offset.get().unwrap_or_else(|| {
            panic!("cute: task_scheduling: the shared memory of `{}` has no offset yet (compute_layout)", self.name)
        })
    }

    pub(crate) fn set_offset(&self, offset: usize) {
        self.offset.set(Some(offset));
    }

    /// Whether `other` is a clone of this allocation.
    pub fn same_allocation(&self, other: &SmemAllocation) -> bool {
        Rc::ptr_eq(&self.offset, &other.offset)
    }

    /// Its start as a pointer to `T`, from the unified block's base in
    /// `context` (in a kernel built with a TaskManager that has a shared
    /// memory allocator). Panics without one.
    #[track_caller]
    pub fn ptr<T: Numeric>(&self, context: &ResourceContext) -> Ptr<T> {
        let base = context.smem_base.as_ref().unwrap_or_else(|| {
            panic!("cute: task_scheduling: `{}` needs the TaskManager's shared memory allocator", self.name)
        });
        base.offset(self.offset() as i32).cast::<T>()
    }
}

/// Tensor-memory columns a resource asks the TMEM allocator for (CuTe
/// DSL's `TmemAllocation`). Its column offset is set by
/// [`TmemAllocator::compute_layout`](super::TmemAllocator::compute_layout)
/// and shared by every clone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TmemAllocation {
    pub name: String,
    pub num_columns: usize,
    offset: Rc<Cell<Option<usize>>>,
}

impl TmemAllocation {
    /// `num_columns` columns.
    pub fn new(name: &str, num_columns: usize) -> TmemAllocation {
        TmemAllocation { name: name.to_string(), num_columns, offset: Rc::default() }
    }

    /// Its column offset, once laid out.
    pub fn try_offset(&self) -> Option<usize> {
        self.offset.get()
    }

    /// Its column offset. Panics before
    /// [`compute_layout`](super::TmemAllocator::compute_layout).
    #[track_caller]
    pub fn offset(&self) -> usize {
        self.offset.get().unwrap_or_else(|| {
            panic!("cute: task_scheduling: the TMEM of `{}` has no offset yet (compute_layout)", self.name)
        })
    }

    pub(crate) fn set_offset(&self, offset: usize) {
        self.offset.set(Some(offset));
    }

    /// Whether `other` is a clone of this allocation.
    pub fn same_allocation(&self, other: &TmemAllocation) -> bool {
        Rc::ptr_eq(&self.offset, &other.offset)
    }

    /// Its first column as a TMEM pointer to `T` (one 32-bit element per
    /// column), read from the address the TaskManager's TMEM allocator
    /// stored (`context.tmem_ptr_i32`). Panics without one.
    #[track_caller]
    pub fn ptr<T: Numeric>(&self, context: &ResourceContext) -> Ptr<T> {
        let slot = context
            .tmem_ptr_i32
            .as_ref()
            .unwrap_or_else(|| panic!("cute: task_scheduling: `{}` needs the TaskManager's TMEM allocator", self.name));
        crate::retrieve_tmem_ptr::<f32>(16, slot).offset(self.offset() as i32).cast::<T>()
    }
}

// ---------------------------------------------------------------------------
// The materialised pipeline
// ---------------------------------------------------------------------------

/// The pipeline a resource's [`PipelineConfig`] materialises to.
#[derive(Clone, Debug)]
pub enum ResourcePipeline {
    Async(PipelineAsync),
    CpAsync(PipelineCpAsync),
    TmaAsync(PipelineTmaAsync),
    TmaUmma(PipelineTmaUmma),
    UmmaAsync(PipelineUmmaAsync),
    AsyncUmma(PipelineAsyncUmma),
    ClcFetchAsync(PipelineClcFetchAsync),
    /// One of Task Scheduling's own pipelines ([`TsPipeline`]): `UmmaUmma`,
    /// TMA pipelines with a task-warp leader or all-thread release, and the
    /// members of a [`PipelineGroup`](super::PipelineGroup).
    Ts(TsPipeline),
}

macro_rules! dispatch {
    ($self:expr, $p:ident => $e:expr) => {
        match $self {
            ResourcePipeline::Async($p) => $e,
            ResourcePipeline::CpAsync($p) => $e,
            ResourcePipeline::TmaAsync($p) => $e,
            ResourcePipeline::TmaUmma($p) => $e,
            ResourcePipeline::UmmaAsync($p) => $e,
            ResourcePipeline::AsyncUmma($p) => $e,
            ResourcePipeline::ClcFetchAsync($p) => $e,
            ResourcePipeline::Ts($p) => $e,
        }
    };
}

impl ResourcePipeline {
    /// Build the pipeline `cfg` describes (in a kernel). With `defer_sync`
    /// the barriers' initialisation is not published; the caller does it
    /// once for every pipeline. `UmmaUmma` and `AsyncUmma` make a
    /// [`TsPipeline`] (CuTe DSL's `TSPipelineUmmaUmma` and
    /// `TSPipelineAsyncUmma`, `cp.async` producers included). Panics
    /// for an invalid configuration.
    #[track_caller]
    pub fn create(cfg: &PipelineConfig, defer_sync: bool, name: &str) -> ResourcePipeline {
        ResourcePipeline::create_for(cfg, defer_sync, name, None, false)
    }

    /// [`create`](Self::create) with what the TaskManager derives from the
    /// tasks: the first warp of the producing task, whose leader alone arms
    /// a TMA stage when the producers signal as `TASK_WARP_LEADER`, and
    /// whether every consumer thread releases a `TmaAsync` stage (its
    /// consumer group counting the consuming task's threads). Either makes
    /// a `TmaAsync` resource a [`TsPipeline`].
    #[track_caller]
    pub fn create_for(
        cfg: &PipelineConfig,
        defer_sync: bool,
        name: &str,
        leader_warp: Option<u32>,
        all_thread_release: bool,
    ) -> ResourcePipeline {
        if let Err(e) = cfg.validate() {
            panic!("cute: task_scheduling: {e}");
        }
        let leader = leader_warp.filter(|_| cfg.producer_signaling_threads.has_task_warp_leader());
        let ts_tma = cfg.pipeline_type == PipelineType::TmaAsync && (leader.is_some() || all_thread_release);
        if ts_tma || matches!(cfg.pipeline_type, PipelineType::UmmaUmma | PipelineType::AsyncUmma) {
            let single = cfg.cta_layout_vmnk.as_ref().is_none_or(|l| l.size() == 1);
            assert!(
                single,
                "cute: task_scheduling: {name}: Task Scheduling's own pipelines take one CTA (no clusters yet)"
            );
            let options = TsPipelineOptions {
                barrier_storage: cfg.barrier_ptr.clone(),
                is_task_warp_leader: leader.map(task_warp_leader),
                all_thread_release,
                producer_op: Some(cfg.producer_op),
                defer_sync,
                name: name.to_string(),
            };
            return ResourcePipeline::Ts(TsPipeline::create(
                cfg.pipeline_type,
                cfg.num_stages,
                cfg.producer_group,
                cfg.consumer_group,
                cfg.num_bytes,
                options,
            ));
        }
        let (n, pg, cg) = (cfg.num_stages, cfg.producer_group, cfg.consumer_group);
        let barrier_storage = cfg.barrier_ptr.clone();
        let name = name.to_string();
        match cfg.pipeline_type {
            PipelineType::AsyncAsync => {
                let options = PipelineAsyncOptions { barrier_storage, defer_sync, name, ..Default::default() };
                match cfg.producer_op {
                    PipelineOp::AsyncLoad => ResourcePipeline::CpAsync(PipelineCpAsync::create(n, pg, cg, options)),
                    _ => ResourcePipeline::Async(PipelineAsync::create(n, pg, cg, options)),
                }
            }
            PipelineType::TmaAsync => {
                let options = PipelineTmaAsyncOptions {
                    barrier_storage,
                    cta_layout_vmnk: cfg.cta_layout_vmnk.clone(),
                    mcast_mode_mn: cfg.mcast_mode_mn,
                    defer_sync,
                    name,
                    ..Default::default()
                };
                ResourcePipeline::TmaAsync(PipelineTmaAsync::create(n, pg, cg, cfg.num_bytes, options))
            }
            PipelineType::TmaUmma => {
                let options = PipelineTmaUmmaOptions {
                    barrier_storage,
                    cta_layout_vmnk: cfg.cta_layout_vmnk.clone(),
                    mcast_mode_mn: cfg.mcast_mode_mn,
                    defer_sync,
                    name,
                    ..Default::default()
                };
                ResourcePipeline::TmaUmma(PipelineTmaUmma::create(n, pg, cg, cfg.num_bytes, options))
            }
            PipelineType::UmmaAsync => {
                let options = PipelineUmmaOptions {
                    barrier_storage,
                    cta_layout_vmnk: cfg.cta_layout_vmnk.clone(),
                    defer_sync,
                    name,
                };
                ResourcePipeline::UmmaAsync(PipelineUmmaAsync::create(n, pg, cg, options))
            }
            PipelineType::ClcFetchAsync => {
                let options = PipelineClcFetchAsyncOptions {
                    barrier_storage,
                    cta_layout_vmnk: cfg.cta_layout_vmnk.clone(),
                    defer_sync,
                    name,
                };
                ResourcePipeline::ClcFetchAsync(PipelineClcFetchAsync::create(n, pg, cg, cfg.num_bytes, options))
            }
            PipelineType::UmmaUmma | PipelineType::AsyncUmma => unreachable!("made a TsPipeline above"),
        }
    }

    /// The number of stages.
    pub fn num_stages(&self) -> usize {
        dispatch!(self, p => p.num_stages())
    }

    /// See [`Pipeline::producer_try_acquire`].
    #[track_caller]
    pub fn producer_try_acquire(&self, state: &PipelineState) -> Boolean {
        dispatch!(self, p => p.producer_try_acquire(state))
    }

    /// See [`Pipeline::producer_acquire`].
    #[track_caller]
    pub fn producer_acquire(&self, state: &PipelineState, token: Option<Boolean>) {
        dispatch!(self, p => p.producer_acquire(state, token))
    }

    /// See [`Pipeline::producer_commit`].
    #[track_caller]
    pub fn producer_commit(&self, state: &PipelineState) {
        dispatch!(self, p => p.producer_commit(state))
    }

    /// See [`Pipeline::consumer_try_wait`].
    #[track_caller]
    pub fn consumer_try_wait(&self, state: &PipelineState) -> Boolean {
        dispatch!(self, p => p.consumer_try_wait(state))
    }

    /// See [`Pipeline::consumer_wait`].
    #[track_caller]
    pub fn consumer_wait(&self, state: &PipelineState, token: Option<Boolean>) {
        dispatch!(self, p => p.consumer_wait(state, token))
    }

    /// See [`Pipeline::consumer_release`].
    #[track_caller]
    pub fn consumer_release(&self, state: &PipelineState) {
        dispatch!(self, p => p.consumer_release(state))
    }

    /// See [`Pipeline::producer_tail`].
    #[track_caller]
    pub fn producer_tail(&self, state: &mut PipelineState) {
        dispatch!(self, p => p.producer_tail(state))
    }

    /// The producers' barrier of stage `state.index()`.
    #[track_caller]
    pub fn producer_get_barrier(&self, state: &PipelineState) -> Ptr<u64> {
        dispatch!(self, p => p.producer_get_barrier(state))
    }

    /// The consumers' barrier of stage `state.index()`.
    #[track_caller]
    pub fn consumer_get_barrier(&self, state: &PipelineState) -> Ptr<u64> {
        dispatch!(self, p => p.consumer_get_barrier(state))
    }
}

// ---------------------------------------------------------------------------
// MemoryResource and the Resource trait
// ---------------------------------------------------------------------------

/// Run-time state a resource makes in [`Resource::create`]: its pipeline
/// and the status flags of its last probes.
struct Runtime {
    pipeline: Option<ResourcePipeline>,
    producer_status: Var<Int32>,
    consumer_status: Var<Int32>,
}

/// A named piece of memory with work methods (CuTe DSL's `MemoryResource`):
/// producer work fills it, consumer work reads it (returning task-local
/// variables), and with a [`PipelineConfig`] it is staged and handed over
/// through a pipeline, whose operations (`try_acquire`, `acquire`,
/// `commit`; `try_wait`, `wait`, `release`) schedules order around the work.
///
/// A resource is described once, outside or inside a kernel: its name,
/// work methods ([`Work`]), variables and memory requirements. Inside a
/// kernel, [`Resource::create`] builds its pipeline and status flags; then
/// a hand-written kernel calls [`run_work`](Self::run_work) and the
/// pipeline methods directly, and the [`TaskManager`](super::TaskManager)
/// calls them in the order a captured [`Schedule`](super::Schedule) gives.
///
/// The pipeline methods follow the default advance policy: a commit
/// advances the producer's state, a release the consumer's, by the role's
/// interleave stride. A probe stores its result in the side's status flag
/// and the next acquire or wait takes it as its token (and clears it), so
/// that `try_acquire` then `acquire` waits only if the probe failed. On a
/// resource without a pipeline the operations do nothing and the probes
/// return true.
pub struct MemoryResource {
    name: String,
    pipeline_config: Option<PipelineConfig>,
    variables: Vec<(VarRef, Rc<str>)>,
    works: Vec<WorkDecl>,
    smem: Vec<SmemAllocation>,
    tmem: Vec<TmemAllocation>,
    runtime: Option<Runtime>,
    leader_warp: Option<u32>,
    all_thread_release: bool,
}

impl fmt::Debug for MemoryResource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryResource")
            .field("name", &self.name)
            .field("pipeline", &self.pipeline_config.as_ref().map(|c| c.pipeline_type))
            .field("works", &self.works.iter().map(|w| &w.name).collect::<Vec<_>>())
            .field("variables", &self.variables.iter().map(|(v, _)| v.name()).collect::<Vec<_>>())
            .finish()
    }
}

impl MemoryResource {
    /// A resource called `name`, without work or a pipeline.
    pub fn new(name: &str) -> MemoryResource {
        MemoryResource {
            name: name.to_string(),
            pipeline_config: None,
            variables: vec![],
            works: vec![],
            smem: vec![],
            tmem: vec![],
            runtime: None,
            leader_warp: None,
            all_thread_release: false,
        }
    }

    /// Staged through the pipeline `cfg` describes.
    pub fn with_pipeline_config(mut self, cfg: PipelineConfig) -> Self {
        self.pipeline_config = Some(cfg);
        self
    }

    /// Owning the task-local variable `var` (variables a work method
    /// returns are added by [`work`](Self::work) as well).
    pub fn variable<C: LocalValue>(mut self, var: &TaskLocalVariable<C>) -> Self {
        self.add_variable(&var.r, &var.docs);
        self
    }

    fn add_variable(&mut self, var: &VarRef, docs: &Rc<str>) {
        if let Some((v, _)) = self.variables.iter().find(|(v, _)| v.name() == var.name()) {
            assert!(
                v == var,
                "cute: task_scheduling: {} already has a different variable called `{}`",
                self.name,
                var.name()
            );
        } else {
            self.variables.push((var.clone(), docs.clone()));
        }
    }

    /// With the work method `work`. Panics if its name is a pipeline
    /// operation's or another work method's.
    #[track_caller]
    pub fn work(mut self, work: WorkDecl) -> Self {
        self.add_work(work);
        self
    }

    /// Add the work method `work` (see [`work`](Self::work)).
    #[track_caller]
    pub fn add_work(&mut self, work: WorkDecl) {
        assert!(
            !ScheduleStage::OP_NAMES.contains(&work.name.as_str()),
            "cute: task_scheduling: {}: `{}` is the name of a pipeline operation and cannot name a work method",
            self.name,
            work.name
        );
        assert!(
            !self.works.iter().any(|w| w.name == work.name),
            "cute: task_scheduling: {} already has a work method called `{}`",
            self.name,
            work.name
        );
        for r in &work.returns {
            self.add_variable(r, &Rc::from(""));
        }
        self.works.push(work);
    }

    /// Asking the shared-memory allocator for `alloc`.
    pub fn smem_requirement(mut self, alloc: SmemAllocation) -> Self {
        self.smem.push(alloc);
        self
    }

    /// Asking the TMEM allocator for `alloc`.
    pub fn tmem_requirement(mut self, alloc: TmemAllocation) -> Self {
        self.tmem.push(alloc);
        self
    }

    /// The resource's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Its pipeline configuration.
    pub fn pipeline_config(&self) -> Option<&PipelineConfig> {
        self.pipeline_config.as_ref()
    }

    /// Its work methods.
    pub fn works(&self) -> &[WorkDecl] {
        &self.works
    }

    /// The work method `name`.
    pub fn find_work(&self, name: &str) -> Option<&WorkDecl> {
        self.works.iter().find(|w| w.name == name)
    }

    /// Its task-local variables.
    pub fn variables(&self) -> impl Iterator<Item = &VarRef> {
        self.variables.iter().map(|(v, _)| v)
    }

    /// The pipeline, once [`Resource::create`] has built it.
    pub fn pipeline(&self) -> Option<&ResourcePipeline> {
        self.runtime.as_ref()?.pipeline.as_ref()
    }

    /// Build the pipeline and status flags (in a kernel).
    #[track_caller]
    pub fn create_with(&mut self, defer_sync: bool) {
        let pipeline = self.pipeline_config.as_ref().map(|cfg| {
            ResourcePipeline::create_for(cfg, defer_sync, &self.name, self.leader_warp, self.all_thread_release)
        });
        self.create_with_pipeline(pipeline);
    }

    /// Build the status flags around a pipeline made elsewhere (by a
    /// [`PipelineGroup`](super::PipelineGroup) for its members).
    #[track_caller]
    pub fn create_with_pipeline(&mut self, pipeline: Option<ResourcePipeline>) {
        assert!(self.runtime.is_none(), "cute: task_scheduling: {} was created twice", self.name);
        self.runtime = Some(Runtime {
            pipeline,
            producer_status: Var::new(Int32::constant(0)),
            consumer_status: Var::new(Int32::constant(0)),
        });
    }

    /// Place the pipeline's barriers at `ptr` (the shared memory
    /// allocator's placement), unless the configuration has storage already.
    pub(crate) fn set_barrier_ptr(&mut self, ptr: Ptr<u64>) {
        if let Some(cfg) = self.pipeline_config.as_mut()
            && cfg.barrier_ptr.is_none()
        {
            cfg.barrier_ptr = Some(ptr);
        }
    }

    /// The TaskManager's derived pipeline hints (see
    /// [`ResourcePipeline::create_for`]).
    pub(crate) fn set_pipeline_hints(&mut self, leader_warp: Option<u32>, all_thread_release: bool) {
        self.leader_warp = leader_warp;
        self.all_thread_release = all_thread_release;
    }

    /// Advance `state` as a commit (producer) or release (consumer) does,
    /// without arriving (a group arrives for its members).
    pub(crate) fn advance_after_close(&self, state: &mut PipelineState, side: Side) {
        advance(state, self.stride(if side == Side::Producer { 1 } else { 3 }));
    }

    #[track_caller]
    fn runtime(&self) -> &Runtime {
        self.runtime
            .as_ref()
            .unwrap_or_else(|| panic!("cute: task_scheduling: {} is used before Resource::create", self.name))
    }

    fn stages(&self) -> usize {
        self.pipeline_config.as_ref().map_or(1, |c| c.num_stages)
    }

    fn stride(&self, role: usize) -> usize {
        self.pipeline_config.as_ref().map_or(1, |c| c.interleave_stride.0[role])
    }

    /// The producer's starting state (phase 1: the first pass over the
    /// ring does not wait). One notional stage without a pipeline.
    #[track_caller]
    pub fn producer_state(&self) -> PipelineState {
        make_pipeline_state(PipelineUserType::Producer, self.stages())
    }

    /// The consumer's starting state.
    #[track_caller]
    pub fn consumer_state(&self) -> PipelineState {
        make_pipeline_state(PipelineUserType::Consumer, self.stages())
    }

    /// Record the producer's probe result (CuTe DSL's `store_producer_status`).
    #[track_caller]
    pub fn store_producer_status(&self, value: Boolean) {
        self.runtime().producer_status.set(value.to::<i32>());
    }

    /// The producer's status flag.
    #[track_caller]
    pub fn load_producer_status(&self) -> Int32 {
        self.runtime().producer_status.get()
    }

    /// Record the consumer's probe result.
    #[track_caller]
    pub fn store_consumer_status(&self, value: Boolean) {
        self.runtime().consumer_status.set(value.to::<i32>());
    }

    /// The consumer's status flag.
    #[track_caller]
    pub fn load_consumer_status(&self) -> Int32 {
        self.runtime().consumer_status.get()
    }

    /// Probe whether the producer's stage is free; the result also goes to
    /// the status flag, for the next [`acquire`](Self::acquire).
    #[track_caller]
    pub fn try_acquire(&self, state: &PipelineState) -> Boolean {
        let Some(p) = self.runtime().pipeline.as_ref() else { return Boolean::constant(true) };
        let ok = p.producer_try_acquire(state);
        self.store_producer_status(ok);
        ok
    }

    /// Wait until the producer's stage is free, unless the last probe said
    /// so; clears the status flag.
    #[track_caller]
    pub fn acquire(&self, state: &PipelineState) {
        let rt = self.runtime();
        let Some(p) = rt.pipeline.as_ref() else { return };
        p.producer_acquire(state, Some(rt.producer_status.get().ne(0)));
        rt.producer_status.set(Int32::constant(0));
    }

    /// Signal that the producer's stage is filled, and advance `state`.
    #[track_caller]
    pub fn commit(&self, state: &mut PipelineState) {
        if let Some(p) = self.runtime().pipeline.as_ref() {
            p.producer_commit(state);
            advance(state, self.stride(1));
        }
    }

    /// Probe whether the consumer's stage is full; the result also goes to
    /// the status flag, for the next [`wait`](Self::wait).
    #[track_caller]
    pub fn try_wait(&self, state: &PipelineState) -> Boolean {
        let Some(p) = self.runtime().pipeline.as_ref() else { return Boolean::constant(true) };
        let ok = p.consumer_try_wait(state);
        self.store_consumer_status(ok);
        ok
    }

    /// Wait until the consumer's stage is full, unless the last probe said
    /// so; clears the status flag.
    #[track_caller]
    pub fn wait(&self, state: &PipelineState) {
        let rt = self.runtime();
        let Some(p) = rt.pipeline.as_ref() else { return };
        p.consumer_wait(state, Some(rt.consumer_status.get().ne(0)));
        rt.consumer_status.set(Int32::constant(0));
    }

    /// Signal that the consumer is done with its stage, and advance `state`.
    #[track_caller]
    pub fn release(&self, state: &mut PipelineState) {
        if let Some(p) = self.runtime().pipeline.as_ref() {
            p.consumer_release(state);
            advance(state, self.stride(3));
        }
    }

    /// Before the producer exits: wait until every stage is released.
    #[track_caller]
    pub fn producer_tail(&self, state: &mut PipelineState) {
        if let Some(p) = self.runtime().pipeline.as_ref() {
            p.producer_tail(state);
        }
    }

    /// `info` with the producer's stage of `state` and its full barrier.
    #[track_caller]
    pub fn producer_stage(&self, state: &PipelineState, info: StageInfo) -> StageInfo {
        match self.runtime().pipeline.as_ref() {
            Some(p) => info.with_stage(state, Some(p.producer_get_barrier(&base_state(p)))),
            None => info,
        }
    }

    /// `info` with the consumer's stage of `state` and its empty barrier.
    #[track_caller]
    pub fn consumer_stage(&self, state: &PipelineState, info: StageInfo) -> StageInfo {
        match self.runtime().pipeline.as_ref() {
            Some(p) => info.with_stage(state, Some(p.consumer_get_barrier(&base_state(p)))),
            None => info,
        }
    }

    /// Call the work method `name` with `info` and `args`. Panics if there
    /// is no such method, an input is missing or of the wrong type, or the
    /// results are not the declared ones.
    #[track_caller]
    pub fn run_work(&self, name: &str, info: &StageInfo, args: &WorkArgs) -> WorkResults {
        let work = self.find_work(name).unwrap_or_else(|| {
            let names: Vec<&str> = self.works.iter().map(|w| w.name.as_str()).collect();
            panic!("cute: task_scheduling: {} has no work method `{name}` (it has: {})", self.name, names.join(", "))
        });
        for input in &work.inputs {
            assert!(
                args.contains(&input.name),
                "cute: task_scheduling: {}.{name} needs the input `{}`",
                self.name,
                input.name
            );
            assert!(
                args.type_matches(&input.name, input.type_id),
                "cute: task_scheduling: {}.{name}: the input `{}` must be a {}",
                self.name,
                input.name,
                input.type_label
            );
        }
        let info = if info.label.is_none() { info.clone().with_label(name) } else { info.clone() };
        let results = (work.body)(&info, args);
        let returned: Vec<&str> = results.vars().map(|v| v.name()).collect();
        let declared: Vec<&str> = work.returns.iter().map(|v| v.name()).collect();
        assert!(
            work.returns.iter().all(|r| results.vars().any(|v| v == r)) && results.vars().count() == work.returns.len(),
            "cute: task_scheduling: {}.{name} declares the results {declared:?} but returned {returned:?}",
            self.name
        );
        results
    }
}

/// A state at stage 0, for the base of a barrier array.
fn base_state(p: &ResourcePipeline) -> PipelineState {
    PipelineState::new(p.num_stages(), 0, 0, 0)
}

fn advance(state: &mut PipelineState, stride: usize) {
    if stride == 1 {
        state.advance();
    } else {
        state.advance_by(stride);
    }
}

/// What every resource offers (CuTe DSL's `MemoryResource` interface): a
/// name, an optional pipeline, memory requirements for the allocators,
/// and creation in a kernel. [`MemoryResource`] and [`WorkQueue`](super::WorkQueue)
/// implement it; a resource with its own logic wraps a `MemoryResource`
/// and overrides what it needs (the requirement hooks, say).
pub trait Resource {
    /// The underlying memory resource.
    fn memory(&self) -> &MemoryResource;

    /// The underlying memory resource, mutably.
    fn memory_mut(&mut self) -> &mut MemoryResource;

    /// The resource's name.
    fn name(&self) -> &str {
        self.memory().name()
    }

    /// Its pipeline configuration.
    fn pipeline_config(&self) -> Option<&PipelineConfig> {
        self.memory().pipeline_config()
    }

    /// The shared memory it needs (CuTe DSL's `get_smem_requirements`).
    fn get_smem_requirements(&self) -> Vec<SmemAllocation> {
        self.memory().smem.clone()
    }

    /// The TMEM it needs (CuTe DSL's `get_tmem_requirements`).
    fn get_tmem_requirements(&self) -> Vec<TmemAllocation> {
        self.memory().tmem.clone()
    }

    /// Whether it is a [`WorkQueue`](super::WorkQueue).
    fn is_work_queue(&self) -> bool {
        false
    }

    /// The pipeline group, if it is one ([`PipelineGroup`](super::PipelineGroup)).
    fn as_group(&self) -> Option<&super::PipelineGroup> {
        None
    }

    /// The pipeline group, mutably.
    fn as_group_mut(&mut self) -> Option<&mut super::PipelineGroup> {
        None
    }

    /// The work queue, if it is one (the TaskManager drives its persistent
    /// loop and skip predicate).
    fn as_work_queue(&self) -> Option<&super::WorkQueue> {
        None
    }

    /// Which programmatic-dependent-launch barrier it is, if it is one
    /// ([`PdlWaitBarrier`](super::PdlWaitBarrier),
    /// [`PdlLaunchBarrier`](super::PdlLaunchBarrier)).
    fn pdl_role(&self) -> Option<super::PdlRole> {
        None
    }

    /// Whether it is a barrier-only resource (CuTe DSL's `is_barrier`):
    /// it carries no data, so graph edges into or out of it only order.
    fn is_barrier(&self) -> bool {
        self.pdl_role().is_some()
    }

    /// Build its run-time state in a kernel, leaving the pipeline's
    /// initialisation unpublished with `defer_sync`.
    #[track_caller]
    fn create_with(&mut self, defer_sync: bool) {
        self.memory_mut().create_with(defer_sync);
    }

    /// Build its run-time state and publish the pipeline's barriers (every
    /// thread of the block must reach this).
    #[track_caller]
    fn create(&mut self) {
        self.create_with(false);
    }
}

impl Resource for MemoryResource {
    fn memory(&self) -> &MemoryResource {
        self
    }

    fn memory_mut(&mut self) -> &mut MemoryResource {
        self
    }
}
