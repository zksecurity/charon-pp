//! Compute the mutability of the lifetime parameters of closure state types.
//!
//! The lifetime mutability of ADTs is computed during translation from the rustc field types;
//! closure states are not ADTs for rustc, so we compute theirs here from the translated fields. A
//! lifetime of a closure state is mutable if a field mentions it as the lifetime of a `&mut`, or
//! passes it to an ADT (closure states included) in a mutable lifetime position.
use std::collections::HashMap;

use crate::ast::*;
use crate::transform::{TransformCtx, ctx::TransformPass};

/// Collects the item-level lifetimes used mutably in the visited types.
#[derive(Visitor)]
struct MutableRegionsCollector<'a> {
    /// The mutable lifetimes of each ADT, by index of region parameter.
    mutable_regions: &'a HashMap<TypeDeclId, Vec<bool>>,
    /// The mutable lifetimes found so far.
    found: Vec<bool>,
    binder_depth: DeBruijnId,
}

impl MutableRegionsCollector<'_> {
    fn mark(&mut self, r: &Region) {
        if let Region::Var(var) = r
            && let Some(id) = var.bound_at_depth(self.binder_depth)
        {
            self.found[id.index()] = true;
        }
    }
}

impl VisitorWithBinderDepth for MutableRegionsCollector<'_> {
    fn binder_depth_mut(&mut self) -> &mut DeBruijnId {
        &mut self.binder_depth
    }
}

impl VisitAst for MutableRegionsCollector<'_> {
    fn visit<T: AstVisitable>(&mut self, x: &T) -> ControlFlow<Self::Break> {
        VisitWithBinderDepth::new(self).visit(x)
    }

    fn enter_ty(&mut self, ty: &Ty) {
        match ty.kind() {
            TyKind::Ref(r, _, RefKind::Mut) => self.mark(r),
            TyKind::Adt(tref) => {
                if let Some(mutable) = self.mutable_regions.get(&tref.id) {
                    for (r, &is_mut) in tref.generics.regions.iter().zip(mutable) {
                        if is_mut {
                            self.mark(r);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

pub struct Transform;
impl TransformPass for Transform {
    fn transform_ctx(&self, ctx: &mut TransformCtx) {
        let mut mutable_regions: HashMap<TypeDeclId, Vec<bool>> = ctx
            .translated
            .type_decls
            .iter()
            .map(|tdecl| {
                let mutable = if tdecl.src.is_closure() {
                    vec![false; tdecl.generics.regions.len()]
                } else {
                    tdecl
                        .generics
                        .regions
                        .iter()
                        .map(|r| r.mutability.is_mutable())
                        .collect()
                };
                (tdecl.def_id, mutable)
            })
            .collect();

        // Closures may capture other closures: iterate to a fixpoint. Each round can only turn
        // lifetimes from shared to mutable, so this terminates.
        loop {
            let mut changed = false;
            for tdecl in ctx.translated.type_decls.iter() {
                let TypeSource::Closure { .. } = tdecl.src else {
                    continue;
                };
                let TypeDeclKind::Struct(fields) = &tdecl.kind else {
                    continue;
                };
                let mut collector = MutableRegionsCollector {
                    mutable_regions: &mutable_regions,
                    found: mutable_regions[&tdecl.def_id].clone(),
                    binder_depth: DeBruijnId::zero(),
                };
                let _ = collector.visit(fields);
                let found = collector.found;
                if found != mutable_regions[&tdecl.def_id] {
                    mutable_regions.insert(tdecl.def_id, found);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        for tdecl in ctx.translated.type_decls.iter_mut() {
            if !tdecl.src.is_closure() {
                continue;
            }
            let mutable = &mutable_regions[&tdecl.def_id];
            for (param, &is_mut) in tdecl.generics.regions.iter_mut().zip(mutable) {
                param.mutability = if is_mut {
                    LifetimeMutability::Mutable
                } else {
                    LifetimeMutability::Shared
                };
            }
        }
    }
}
