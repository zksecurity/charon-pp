//! Region outlives requirements of closures.
//!
//! The borrow checker of a closure body may need relations between the regions the closure
//! receives from its creator (e.g. the regions of its signature that rustc erases). It
//! propagates these relations to the creator instead of stating them on the closure, and
//! keeps them internal. We recover them with the borrowck consumer API whenever a typeck root
//! gets borrow-checked, and state them as predicates on the closure.
//!
//! Inside a closure body, region variable 0 is `'static` and region variables `1..=n` are the
//! free regions of the closure's identity args (`type_of(closure)`), in the order of
//! `fold_regions`. We record requirements between these, indexed by their position in
//! that order. Requirements on the late-bound regions of the creator can't be stated on the
//! closure and are not recorded.
use rustc_borrowck::consumers::{ConsumerOptions, get_bodies_with_borrowck_facts};
use rustc_hir::def::DefKind;
use rustc_middle::ty::{self, RegionVid, TyCtxt, Upcast};
use rustc_middle::util::Providers;
use rustc_span::def_id::LocalDefId;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// A region of a closure's identity args: its position among their free regions, or `'static`.
#[derive(Debug, Clone, Copy)]
pub enum ClosureRegion {
    Static,
    FreeRegion(usize),
}

/// `(longer, shorter)`: the region `longer` must outlive the free region `shorter`.
pub type ClosureRequirement = (usize, ClosureRegion);

static REQUIREMENTS: LazyLock<Mutex<HashMap<LocalDefId, Vec<ClosureRequirement>>>> =
    LazyLock::new(Default::default);

/// Wrap the `mir_borrowck` query to record the requirements of the closures it checks.
pub fn record_with_borrowck(providers: &mut Providers) {
    providers.queries.mir_borrowck = |tcx, def_id| {
        record_requirements(tcx, def_id);
        (rustc_interface::DEFAULT_QUERY_PROVIDERS
            .queries
            .mir_borrowck)(tcx, def_id)
    };
}

/// The regions of `args` that are not bound inside them, in the order rustc numbers them.
fn free_regions<'tcx>(tcx: TyCtxt<'tcx>, args: ty::GenericArgsRef<'tcx>) -> Vec<ty::Region<'tcx>> {
    let mut regions = vec![];
    ty::fold_regions(tcx, args, |r, _| {
        regions.push(r);
        r
    });
    regions
}

/// The identity args of a closure, or `None` for other closure-like bodies.
fn closure_args<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> Option<ty::GenericArgsRef<'tcx>> {
    let ty = tcx
        .type_of(def_id)
        .instantiate_identity()
        .skip_normalization();
    match *ty.kind() {
        ty::Closure(_, args) => Some(args),
        _ => None,
    }
}

fn record_requirements(tcx: TyCtxt<'_>, root: LocalDefId) {
    let has_closures = tcx
        .nested_bodies_within(root)
        .iter()
        .any(|body| tcx.def_kind(body) == DefKind::Closure);
    if !has_closures {
        return;
    }
    let bodies = get_bodies_with_borrowck_facts(tcx, root, ConsumerOptions::RegionInferenceContext);
    let mut requirements = REQUIREMENTS.lock().unwrap();
    for (def_id, body) in bodies {
        let Some(args) = closure_args(tcx, def_id) else {
            continue;
        };
        let regioncx = &body.region_inference_context;
        let num_free = free_regions(tcx, args).len();
        let vid = |i: usize| RegionVid::from_usize(i + 1);
        let mut reqs = vec![];
        for longer in 0..num_free {
            if regioncx.eval_outlives(vid(longer), RegionVid::from_usize(0)) {
                reqs.push((longer, ClosureRegion::Static));
                continue;
            }
            for shorter in 0..num_free {
                if longer != shorter && regioncx.eval_outlives(vid(longer), vid(shorter)) {
                    reqs.push((longer, ClosureRegion::FreeRegion(shorter)));
                }
            }
        }
        requirements.insert(def_id, reqs);
    }
}

/// The outlives requirements of a local closure, as clauses over its identity args. Empty if
/// its typeck root wasn't borrow-checked.
pub fn closure_requirement_clauses<'tcx>(
    tcx: TyCtxt<'tcx>,
    closure: LocalDefId,
    identity_args: ty::GenericArgsRef<'tcx>,
) -> Vec<ty::Clause<'tcx>> {
    let Some(reqs) = REQUIREMENTS.lock().unwrap().get(&closure).cloned() else {
        return vec![];
    };
    let regions = free_regions(tcx, identity_args);
    assert_eq!(
        regions.len(),
        free_regions(tcx, closure_args(tcx, closure).unwrap()).len()
    );
    reqs.into_iter()
        .map(|(longer, shorter)| {
            let shorter = match shorter {
                ClosureRegion::Static => tcx.lifetimes.re_static,
                ClosureRegion::FreeRegion(i) => regions[i],
            };
            let pred = ty::OutlivesClause(regions[longer], shorter);
            ty::ClauseKind::RegionOutlives(pred).upcast(tcx)
        })
        .collect()
}
