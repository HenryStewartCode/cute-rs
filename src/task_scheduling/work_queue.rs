// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Names and semantics follow CuTe DSL 4.8.1's WorkQueue
// (cutlass.experimental.task_scheduling.resources, NVIDIA EULA); the code is
// written independently.

//! The work queue: a resource wrapping a persistent tile scheduler.

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use crate::arch::{block_idx, block_idx_in_cluster, elect_one, grid_dim, issue_clc_query, make_warp_uniform};
use crate::control::{Var, if_};
use crate::tensor::Ptr;
use crate::utils::{ClcDynamicPersistentTileScheduler, StaticPersistentTileScheduler, WorkTileInfo};
use crate::value::{Boolean, Int32};

use super::config::{PipelineConfig, TileSchedulerConfig};
use super::resource::{MemoryResource, Resource, StageInfo, TaskLocalVariable, Work, WorkResults};

/// A materialised tile scheduler, held in a [`Var`] so that advancing it
/// in one loop iteration is seen by the next.
enum SchedulerState {
    Static(Var<StaticPersistentTileScheduler>),
    Clc { scheduler: Var<ClcDynamicPersistentTileScheduler>, response: Ptr<i32>, stages: usize },
}

type Shared = Rc<RefCell<Option<SchedulerState>>>;
type SkipFn = Rc<dyn Fn(&WorkTileInfo) -> Boolean>;

/// A resource that hands out work tiles (CuTe DSL's `WorkQueue`), over
/// m9's [`StaticPersistentTileScheduler`] or m10's
/// [`ClcDynamicPersistentTileScheduler`].
///
/// Its consumer variables are `work_tile` (the current [`WorkTileInfo`])
/// and `skip_work_tile`; its work methods are
///
/// * `init_work_tile` (consumer, auxiliary): the scheduler's first tile,
///   and `false` for `skip_work_tile`;
/// * `get_and_advance_work_tile` (consumer): the next tile. The static
///   scheduler advances here, on the consumer side, since it has no
///   producer; with cluster launch control the scheduler warp has fetched
///   it, and this decodes the response of the waited stage;
/// * `fetch_work_tile` (producer): with cluster launch control, CTA 0 of
///   the cluster asks for the next tile, reporting to the stage's barrier;
///   nothing for the static scheduler.
///
/// So every task follows the same pattern at the end of a tile (`try_wait`,
/// `wait`, `get_and_advance_work_tile`, `release`) whichever scheduler is
/// in use; without a pipeline (the static scheduler needs none) the
/// pipeline operations do nothing.
///
/// [`skip_work_tile_if`](Self::skip_work_tile_if) decides which tiles a
/// `work_tile_loop(skip_if)` leaves the skippable work out for; set it with
/// [`with_skip_if`](Self::with_skip_if) (CuTe DSL overrides the method in a
/// subclass).
pub struct WorkQueue {
    mem: MemoryResource,
    config: TileSchedulerConfig,
    state: Shared,
    skip_if: Option<SkipFn>,
    work_tile: TaskLocalVariable<WorkTileInfo>,
    skip_work_tile: TaskLocalVariable<Boolean>,
}

impl fmt::Debug for WorkQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkQueue")
            .field("name", &self.mem.name())
            .field("scheduler", &self.config.tile_scheduler_type())
            .field("skip_if", &self.skip_if.is_some())
            .finish()
    }
}

fn invalid_tile() -> WorkTileInfo {
    WorkTileInfo::new([Int32::constant(0); 3], Boolean::constant(false))
}

/// The response buffer of `stage` (16 bytes, four i32, per stage).
fn stage_response(response: &Ptr<i32>, stages: usize, info: &StageInfo) -> Ptr<i32> {
    match (stages, info.stage_idx) {
        (1, _) | (_, None) => response.clone(),
        (_, Some(idx)) => response.offset(idx * 4),
    }
}

impl WorkQueue {
    /// A work queue called `name` over the scheduler `config` describes.
    pub fn new(name: &str, config: TileSchedulerConfig) -> WorkQueue {
        let work_tile = TaskLocalVariable::new("work_tile", invalid_tile, "the work tile of the persistent loop");
        let skip_work_tile =
            TaskLocalVariable::scalar("skip_work_tile", false, "whether the skippable work is left out for this tile");
        let state: Shared = Rc::new(RefCell::new(None));
        let mut mem = MemoryResource::new(name).variable(&work_tile).variable(&skip_work_tile);

        let init = Work::consumer("init_work_tile").auxiliary().returns(&work_tile).returns(&skip_work_tile).run({
            let (state, wt, skip) = (state.clone(), work_tile.clone(), skip_work_tile.clone());
            move |_, _| {
                let tile = match &*state.borrow() {
                    Some(SchedulerState::Static(s)) => s.get().initial_work_tile_info(),
                    Some(SchedulerState::Clc { scheduler, .. }) => scheduler.get().initial_work_tile_info(),
                    None => invalid_tile(),
                };
                WorkResults::of(&wt, tile).and(&skip, Boolean::constant(false))
            }
        });
        let next = Work::consumer("get_and_advance_work_tile").returns(&work_tile).run({
            let (state, wt) = (state.clone(), work_tile.clone());
            move |info, _| {
                let tile = match &*state.borrow() {
                    Some(SchedulerState::Static(s)) => {
                        let mut sched = s.get();
                        sched.advance_to_next_work();
                        s.set(sched.clone());
                        sched.get_current_work()
                    }
                    Some(SchedulerState::Clc { scheduler, response, stages }) => {
                        scheduler.get().work_tile_info_from_clc_response(&stage_response(response, *stages, info))
                    }
                    None => panic!("cute: task_scheduling: the work queue is used before Resource::create"),
                };
                WorkResults::of(&wt, tile)
            }
        });
        let fetch = Work::producer("fetch_work_tile").run({
            let state = state.clone();
            move |info, _| {
                if let Some(SchedulerState::Clc { scheduler, response, stages }) = &*state.borrow() {
                    let barrier = info
                        .barrier()
                        .expect("cute: task_scheduling: fetch_work_tile needs the work queue's pipeline stage");
                    let response = stage_response(response, *stages, info);
                    if_(make_warp_uniform(block_idx_in_cluster()).eq(0), || {
                        if_(elect_one(), || issue_clc_query(&barrier, &response, true));
                    });
                    let mut sched = scheduler.get();
                    sched.count_tile_executed();
                    scheduler.set(sched);
                }
                WorkResults::none()
            }
        });
        mem.add_work(init);
        mem.add_work(next);
        mem.add_work(fetch);
        WorkQueue { mem, config, state, skip_if: None, work_tile, skip_work_tile }
    }

    /// Staged through the pipeline `cfg` describes (a `ClcFetchAsync`
    /// pipeline for the cluster launch control scheduler).
    pub fn with_pipeline_config(mut self, cfg: PipelineConfig) -> Self {
        self.mem = self.mem.with_pipeline_config(cfg);
        self
    }

    /// Skipping the tiles for which `f` holds (in a
    /// `work_tile_loop(skip_if)`).
    pub fn with_skip_if(mut self, f: impl Fn(&WorkTileInfo) -> Boolean + 'static) -> Self {
        self.skip_if = Some(Rc::new(f));
        self
    }

    /// The scheduler configuration.
    pub fn tile_scheduler_config(&self) -> &TileSchedulerConfig {
        &self.config
    }

    /// The `work_tile` variable.
    pub fn work_tile(&self) -> &TaskLocalVariable<WorkTileInfo> {
        &self.work_tile
    }

    /// The `skip_work_tile` variable.
    pub fn skip_work_tile(&self) -> &TaskLocalVariable<Boolean> {
        &self.skip_work_tile
    }

    /// Whether a skip predicate was set.
    pub fn has_skip_predicate(&self) -> bool {
        self.skip_if.is_some()
    }

    /// Build the scheduler the configuration describes, for this CTA (in a
    /// kernel; CuTe DSL's `create_tile_scheduler`).
    #[track_caller]
    pub fn create_tile_scheduler(&self) -> TileScheduler {
        match &self.config {
            TileSchedulerConfig::StaticPersistent(params) => {
                TileScheduler::Static(StaticPersistentTileScheduler::create(params, block_idx(), grid_dim()))
            }
            TileSchedulerConfig::ClcDynamicPersistent { params, response_ptr } => TileScheduler::Clc(
                ClcDynamicPersistentTileScheduler::create(params, block_idx(), grid_dim(), response_ptr, true),
            ),
        }
    }

    /// The scheduler's first tile for this CTA.
    #[track_caller]
    pub fn initial_work_tile_info(&self) -> WorkTileInfo {
        match &*self.state.borrow() {
            Some(SchedulerState::Static(s)) => s.get().initial_work_tile_info(),
            Some(SchedulerState::Clc { scheduler, .. }) => scheduler.get().initial_work_tile_info(),
            None => panic!("cute: task_scheduling: {} is used before Resource::create", self.mem.name()),
        }
    }

    /// Whether the skippable work is left out for `work_tile`: the
    /// predicate of [`with_skip_if`](Self::with_skip_if), or false.
    #[track_caller]
    pub fn skip_work_tile_if(&self, work_tile: &WorkTileInfo) -> Boolean {
        match &self.skip_if {
            Some(f) => f(work_tile),
            None => Boolean::constant(false),
        }
    }

    /// Run `init_work_tile` (in a hand-written kernel).
    #[track_caller]
    pub fn init_work_tile(&self, info: &StageInfo) -> (WorkTileInfo, Boolean) {
        let r = self.mem.run_work("init_work_tile", info, &Default::default());
        (r.get(&self.work_tile), r.get(&self.skip_work_tile))
    }

    /// Run `get_and_advance_work_tile`.
    #[track_caller]
    pub fn get_and_advance_work_tile(&self, info: &StageInfo) -> WorkTileInfo {
        self.mem.run_work("get_and_advance_work_tile", info, &Default::default()).get(&self.work_tile)
    }

    /// Run `fetch_work_tile`.
    #[track_caller]
    pub fn fetch_work_tile(&self, info: &StageInfo) {
        self.mem.run_work("fetch_work_tile", info, &Default::default());
    }
}

/// The scheduler a [`WorkQueue`] creates.
#[derive(Clone, Debug)]
pub enum TileScheduler {
    Static(StaticPersistentTileScheduler),
    Clc(ClcDynamicPersistentTileScheduler),
}

impl Resource for WorkQueue {
    fn memory(&self) -> &MemoryResource {
        &self.mem
    }

    fn memory_mut(&mut self) -> &mut MemoryResource {
        &mut self.mem
    }

    fn is_work_queue(&self) -> bool {
        true
    }

    fn as_work_queue(&self) -> Option<&WorkQueue> {
        Some(self)
    }

    #[track_caller]
    fn create_with(&mut self, defer_sync: bool) {
        self.mem.create_with(defer_sync);
        let state = match self.create_tile_scheduler() {
            TileScheduler::Static(s) => SchedulerState::Static(Var::new(s)),
            TileScheduler::Clc(s) => {
                let TileSchedulerConfig::ClcDynamicPersistent { response_ptr, .. } = &self.config else {
                    unreachable!()
                };
                let stages = self.mem.pipeline_config().map_or(1, |c| c.num_stages);
                SchedulerState::Clc { scheduler: Var::new(s), response: response_ptr.clone(), stages }
            }
        };
        *self.state.borrow_mut() = Some(state);
    }
}

impl std::ops::Deref for WorkQueue {
    type Target = MemoryResource;
    fn deref(&self) -> &MemoryResource {
        &self.mem
    }
}
