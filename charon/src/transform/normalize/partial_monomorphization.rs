//! This module implements partial monomorphization, which allows specializing generic items on
//! some specific instanciation patterns. This is used by Aeneas to avoid nested mutable borrows:
//! we transform `Iter<'a, &'b mut T>` to `{Iter::<_, &mut U>}<'a, 'b, T>`, where
//! ```ignore
//! struct {Iter::<'a, &'b mut U>}<'a, 'b, U> {
//!   // the field of `Iter` but instantiated with `T -> &'b mut U`.
//! }
//! ```
//!
//! Note: We may need to partial-mono the same item multiple times: `Foo::<&mut A, B>`, `Foo::<A,
//! &mut B>`. Note also that partial-mono is infectious: `Foo<Bar<&mut A>>` generates `Bar::<&mut
//! A>` then `Foo::<Bar::<&mut A>>``.
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Display;
use std::mem;
use std::ops::Range;

use derive_generic_visitor::Visitor;
use index_vec::Idx;

use crate::ast::from_rustc::LangItem;
use crate::ast::visitor::{VisitWithBinderDepth, VisitorWithBinderDepth};
use crate::formatter::IntoFormatter;
use crate::options::MonomorphizeMut;
use crate::pretty::FmtWithCtx;
use crate::register_error;
use crate::transform::ctx::TransformPass;
use crate::transform::normalize::skip_trait_refs_when_known::impl_method_call;
use crate::{transform::TransformCtx, ullbc_ast::*};

type MutabilityShape = Binder<GenericArgs>;

/// See the docs of `MutabilityShapeBuilder::compute_shape`.
#[derive(Visitor)]
struct MutabilityShapeBuilder<'pm, 'ctx> {
    pm: &'pm PartialMonomorphizer<'ctx>,
    /// The parameters that will constitute the final binder.
    params: GenericParams,
    /// The arguments to pass to the final binder to recover the input arguments.
    extracted: GenericArgs,
    /// Current depth under which we're visiting.
    binder_depth: DeBruijnId,
    /// Whether we're visiting values determined by a baked trait proof (see `bake_proof`): the
    /// proof itself and the type arguments its predicate mentions. We keep their structure and
    /// only abstract variables and regions, reusing the variables of equal extracted values.
    concrete: bool,
    /// Set when a proof we're baking can't be expressed in terms of the shape params.
    proof_failed: bool,
    /// For each type argument, the extracted regions and types that come from it.
    extracted_from_type: IndexVec<TypeVarId, (Range<usize>, Range<usize>)>,
    /// While baking a proof, the extracted regions and types that come from its `Self` type. When
    /// reusing extracted values, these take precedence so the proof agrees with the type it's
    /// about.
    preferred: (Range<usize>, Range<usize>),
}

impl<'pm, 'ctx> MutabilityShapeBuilder<'pm, 'ctx> {
    /// Compute the mutability "shape" of a set of generic arguments by factoring out the minimal
    /// amount of information that still allows reconstructing the original arguments while keeping the
    /// "shape arguments" free of mutable borrows.
    ///
    /// For example, for input:
    ///   <u32, &'a mut &'b A, Option::<&'a mut bool>>
    /// we want to build:
    ///   binder<'a, 'b, A, B, C> <A, &'a mut B, Option::<&'b mut C>>
    /// from which we can recover the original arguments by instantiating it with:
    ///   <'a, 'a, u32, &'b A, bool>
    ///
    /// Formally, given `let (shape, shape_args) = get_mutability_shape(args);`, we have the following:
    /// - `shape.substitute(shape_args) == args`;
    /// - `shape_args` contains no infected types;
    /// - `shape` is as shallow as possible (i.e. takes just enough to get all the infected types
    ///   and not more).
    ///
    /// Note: the input arguments are assumed to have been already partially monomorphized, in the
    /// sense that we won't recurse inside ADT args because we assume any ADT applied to infected
    /// args to have been replaced with a fresh infected ADT.
    fn compute_shape(
        pm: &'pm PartialMonomorphizer<'ctx>,
        target_params: &GenericParams,
        args: &GenericArgs,
    ) -> (MutabilityShape, GenericArgs) {
        // We start with the implicit parameters from the original item. We'll need to substitute
        // them once we've figured out the mapping of explicit parameters, but we'll also be adding
        // new trait clauses potentially so we can't leave the vector empty (the ids would be
        // wrong).
        let mut shape_contents = args.clone();
        let trait_args = mem::take(&mut shape_contents.trait_refs);
        let mut builder = Self {
            pm,
            params: GenericParams {
                regions: IndexVec::new(),
                types: IndexVec::new(),
                const_generics: IndexVec::new(),
                ..target_params.clone()
            },
            extracted: GenericArgs {
                regions: IndexVec::new(),
                types: IndexVec::new(),
                const_generics: IndexVec::new(),
                trait_refs: trait_args.clone(),
            },
            binder_depth: DeBruijnId::zero(),
            concrete: false,
            proof_failed: false,
            extracted_from_type: IndexVec::new(),
            preferred: (0..0, 0..0),
        };

        // The clauses whose proof we'll try to bake in, and the type parameters their predicates
        // mention: a concrete proof determines these types so we must keep them.
        let bakeable: HashSet<TraitClauseId> = trait_args
            .iter_enumerated()
            .filter(|(_, tref)| pm.is_bakeable(tref))
            .map(|(i, _)| i)
            .collect();
        let mut fixed_types: HashSet<TypeVarId> = HashSet::new();
        for &i in &bakeable {
            struct CollectTypeVars<'a>(&'a mut HashSet<TypeVarId>);
            impl VarsVisitor for CollectTypeVars<'_> {
                fn visit_type_var(&mut self, v: TypeDbVar) -> Option<Ty> {
                    if let Some(id) = v.bound_at_depth(DeBruijnId::zero()) {
                        self.0.insert(id);
                    }
                    None
                }
            }
            target_params.trait_clauses[i]
                .trait_
                .clone()
                .visit_vars(&mut CollectTypeVars(&mut fixed_types));
        }

        // Traverse the generics and replace any non-infected type, region or const generic with a
        // fresh variable.
        let _ = VisitWithBinderDepth::new(&mut builder).visit(&mut shape_contents.regions);
        for (id, ty) in shape_contents.types.iter_mut_enumerated() {
            let regions_start = builder.extracted.regions.len();
            let types_start = builder.extracted.types.len();
            builder.concrete = fixed_types.contains(&id);
            let _ = VisitWithBinderDepth::new(&mut builder).visit(ty);
            builder.concrete = false;
            builder.extracted_from_type.push((
                regions_start..builder.extracted.regions.len(),
                types_start..builder.extracted.types.len(),
            ));
        }
        let _ = VisitWithBinderDepth::new(&mut builder).visit(&mut shape_contents.const_generics);

        // Bake the concrete proofs about infected types into the shape.
        let baked: IndexVec<TraitClauseId, Option<PolyTraitRef>> = trait_args
            .iter_enumerated()
            .map(|(i, tref)| {
                bakeable
                    .contains(&i)
                    .then(|| builder.bake_proof(&target_params.trait_clauses[i], tref))?
            })
            .collect();

        let shape_params = {
            let mut shape_params = builder.params;
            // Now the explicit params in `shape_params` are correct, and the implicit params are a mix
            // of the old params and new trait clauses. The old params may refer to the old explicit
            // params which is wrong and must be fixed up.
            shape_params.trait_clauses = shape_params.trait_clauses.map_indexed(|i, x| {
                if i.index() < target_params.trait_clauses.len() {
                    x.substitute_explicits(&shape_contents)
                } else {
                    x
                }
            });
            shape_params.trait_type_constraints =
                shape_params.trait_type_constraints.map_indexed(|i, x| {
                    if i.index() < target_params.trait_type_constraints.len() {
                        x.substitute_explicits(&shape_contents)
                    } else {
                        x
                    }
                });
            shape_params.outlives = shape_params
                .outlives
                .into_iter()
                .enumerate()
                .map(|(i, x)| {
                    if i < target_params.outlives.len() {
                        x.substitute_explicits(&shape_contents)
                    } else {
                        x
                    }
                })
                .collect();
            shape_params
        };

        // The first half of the trait params correspond to the original item clauses so we can
        // pass them unmodified.
        shape_contents.trait_refs = shape_params.identity_args().trait_refs;
        shape_contents
            .trait_refs
            .truncate(target_params.trait_clauses.len());

        let mut shape_args = builder.extracted;
        let (shape_params, shape_contents) = if baked.iter().any(Option::is_some) {
            // Remove the clauses whose proof we baked in, and replace their uses with the baked
            // proofs.
            let is_baked = |i: TraitClauseId| baked.get(i).is_some_and(Option::is_some);
            let mut new_params = shape_params.clone();
            new_params.trait_clauses = shape_params
                .trait_clauses
                .iter_enumerated()
                .filter(|(i, _)| !is_baked(*i))
                .map(|(_, clause)| clause.clone())
                .collect();
            for (i, clause) in new_params.trait_clauses.iter_mut_enumerated() {
                clause.clause_id = i;
            }
            // Baked proofs may mention the clauses we added for their nested proofs, so we first
            // renumber these, then use the result as replacement for the baked clauses.
            let old_identity = shape_params.identity_args();
            let mut kept_clauses = new_params.identity_args().trait_refs.into_iter();
            let renumber_kept = GenericArgs {
                trait_refs: old_identity.trait_refs.map_ref_indexed(|i, identity| {
                    if is_baked(i) {
                        identity.clone()
                    } else {
                        kept_clauses.next().unwrap()
                    }
                }),
                ..old_identity.clone()
            };
            let renumber = GenericArgs {
                trait_refs: renumber_kept.trait_refs.map_ref_indexed(|i, tref| {
                    match baked.get(i) {
                        Some(Some(proof)) => proof
                            .clone()
                            .substitute_with_self(&renumber_kept, &TraitRefKind::SelfId),
                        _ => tref.clone(),
                    }
                }),
                ..old_identity
            };
            new_params.trait_clauses = new_params
                .trait_clauses
                .map(|x| x.substitute_with_self(&renumber, &TraitRefKind::SelfId));
            new_params.trait_type_constraints = new_params
                .trait_type_constraints
                .map(|x| x.substitute_with_self(&renumber, &TraitRefKind::SelfId));
            new_params.outlives = new_params
                .outlives
                .into_iter()
                .map(|x| x.substitute_with_self(&renumber, &TraitRefKind::SelfId))
                .collect();
            shape_args.trait_refs = shape_args
                .trait_refs
                .into_iter_enumerated()
                .filter(|(i, _)| !is_baked(*i))
                .map(|(_, tref)| tref)
                .collect();
            (
                new_params,
                shape_contents.substitute_with_self(&renumber, &TraitRefKind::SelfId),
            )
        } else {
            (shape_params, shape_contents)
        };

        let shape = Binder::new(BinderKind::Other, shape_params, shape_contents);
        (shape, shape_args)
    }

    /// If `tref` is a concrete impl proving a predicate about an infected type, express it in
    /// terms of the shape params so the specialized item can use it directly instead of taking it
    /// as a trait clause. This is what makes e.g. `fn f<F: FnMut()>(f: F)` called with a closure
    /// that captures a `&mut` specialize into a function that calls the closure's method directly.
    ///
    /// Inside the proof, types keep their structure, values already extracted are mapped to the
    /// same variable, and nested proofs that aren't concrete become fresh trait clauses.
    fn bake_proof(&mut self, clause: &TraitParam, tref: &PolyTraitRef) -> Option<PolyTraitRef> {
        // The `Self` type of the clause, if it's one of the item's type parameters.
        self.preferred = clause
            .trait_
            .skip_binder
            .generics
            .types
            .first()
            .and_then(|ty| ty.clone().move_from_under_binder())
            .and_then(|ty| ty.as_type_var()?.bound_at_depth(DeBruijnId::zero()))
            .and_then(|id| self.extracted_from_type.get(id).cloned())
            .unwrap_or((0..0, 0..0));
        let params = self.params.clone();
        let extracted = self.extracted.clone();
        let mut tref = tref.clone();
        self.concrete = true;
        self.proof_failed = false;
        let _ = VisitWithBinderDepth::new(self).visit(&mut tref);
        self.concrete = false;
        if self.proof_failed {
            self.params = params;
            self.extracted = extracted;
            return None;
        }
        Some(tref)
    }

    /// Visit a proof nested inside a proof we're baking: keep it if it's concrete, otherwise
    /// replace it with a fresh clause.
    fn visit_nested_proof(&mut self, tref: &mut TraitRef) {
        if matches!(
            tref.kind,
            TraitRefKind::TraitImpl(_) | TraitRefKind::BuiltinOrAuto { .. }
        ) {
            tref.with_contents_mut(|contents| {
                let _ = VisitWithBinderDepth::new(self).visit(contents);
            });
            return;
        }
        let Some(shifted) = tref.clone().move_from_under_binders(self.binder_depth) else {
            self.proof_failed = true;
            return;
        };
        let mut pred = shifted.trait_decl_ref.clone();
        let depth = mem::replace(&mut self.binder_depth, DeBruijnId::zero());
        let _ = VisitWithBinderDepth::new(self).visit(&mut pred);
        self.binder_depth = depth;
        let clause_id = self.params.trait_clauses.next_idx();
        self.params.trait_clauses.push(TraitParam {
            clause_id,
            span: None,
            origin: PredicateOrigin::WhereClauseOnFn,
            trait_: RegionBinder::empty(pred.clone()),
        });
        self.extracted.trait_refs.push(PolyTraitRef::empty(shifted));
        *tref = TraitRef::new(
            TraitRefKind::Clause(
                DeBruijnVar::bound(self.binder_depth, clause_id),
                RegionArgs::empty(),
            ),
            pred.move_under_binders(self.binder_depth),
        );
    }

    /// Replace this value with a fresh variable, and record that we did so.
    fn replace_with_fresh_var<Id, Param, Arg>(
        &mut self,
        val: &mut Arg,
        reuse_existing: bool,
        mk_param: impl FnOnce(Id) -> Param,
        mk_value: impl FnOnce(DeBruijnVar<Id>) -> Arg,
    ) where
        Id: PreferredRange + Display,
        Arg: TyVisitable + Clone + PartialEq,
        GenericParams: HasIdxVecOf<Id, Output = Param>,
        GenericArgs: HasIdxVecOf<Id, Output = Arg>,
    {
        let Some(shifted_val) = val.clone().move_from_under_binders(self.binder_depth) else {
            // Give up on this value.
            return;
        };
        if reuse_existing
            && let Some(id) = {
                let existing = self.extracted.get_idx_vec();
                let preferred = Id::preferred_range(&self.preferred);
                preferred
                    .clone()
                    .chain(0..existing.len())
                    .map(Id::from_usize)
                    .find(|&id| existing.get(id) == Some(&shifted_val))
            }
        {
            *val = mk_value(DeBruijnVar::bound(self.binder_depth, id));
            return;
        }
        // Record the mapping in the output `GenericArgs`.
        self.extracted.get_idx_vec_mut().push(shifted_val);
        // Put a fresh param in place of `val`.
        let id = self.params.get_idx_vec_mut().push_with(mk_param);
        *val = mk_value(DeBruijnVar::bound(self.binder_depth, id));
    }
}

/// Picks the relevant range in `MutabilityShapeBuilder::preferred`.
trait PreferredRange: Idx {
    fn preferred_range(preferred: &(Range<usize>, Range<usize>)) -> Range<usize>;
}
impl PreferredRange for RegionId {
    fn preferred_range(preferred: &(Range<usize>, Range<usize>)) -> Range<usize> {
        preferred.0.clone()
    }
}
impl PreferredRange for TypeVarId {
    fn preferred_range(preferred: &(Range<usize>, Range<usize>)) -> Range<usize> {
        preferred.1.clone()
    }
}
impl PreferredRange for ConstGenericVarId {
    fn preferred_range(_: &(Range<usize>, Range<usize>)) -> Range<usize> {
        0..0
    }
}

impl<'pm, 'ctx> VisitorWithBinderDepth for MutabilityShapeBuilder<'pm, 'ctx> {
    fn binder_depth_mut(&mut self) -> &mut DeBruijnId {
        &mut self.binder_depth
    }
}

impl<'pm, 'ctx> VisitAstMut for MutabilityShapeBuilder<'pm, 'ctx> {
    fn visit<T: AstVisitable>(&mut self, x: &mut T) -> ControlFlow<Self::Break> {
        VisitWithBinderDepth::new(self).visit(x)
    }

    fn enter_ty(&mut self, ty: &mut Ty) {
        if self.concrete {
            if let TyKind::TypeVar(_) = ty.kind() {
                self.replace_with_fresh_var(
                    ty,
                    true,
                    |id| TypeParam::new(id, format!("T{id}"), Variance::Unknown),
                    |v| v.into(),
                );
            }
        } else if !self.pm.is_infected(ty) {
            self.replace_with_fresh_var(
                ty,
                false,
                |id| TypeParam::new(id, format!("T{id}"), Variance::Unknown),
                |v| v.into(),
            );
        }
    }
    fn exit_ty_kind(&mut self, kind: &mut TyKind) {
        if let TyKind::Adt(TypeDeclRef { id, generics, .. }) = kind {
            // Since the type was not replaced with a type var, it's an infected type. We've
            // traversed it so we have its final explicit arguments. Now we need to satisfy its
            // predicates. For that we add all its predicates to the new item, and pass those new
            // trait clauses to it.
            let Some(target_params) = self.pm.generic_params.get(&(*id).into()) else {
                return;
            };
            let Some(shifted_generics) =
                generics.clone().move_from_under_binders(self.binder_depth)
            else {
                // Give up on this value.
                return;
            };

            // Add the target predicates (properly substituted) to the new item params.
            let num_clauses_before_merge = self.params.trait_clauses.len();
            self.params.merge_predicates_from(
                target_params
                    .clone()
                    .substitute_explicits(&shifted_generics),
            );

            // Record the trait arguments in the output `GenericArgs`.
            self.extracted
                .trait_refs
                .extend(shifted_generics.trait_refs);

            // Replace each trait ref with a clause var.
            for (target_clause_id, tref) in generics.trait_refs.iter_mut_enumerated() {
                let clause_id = target_clause_id + num_clauses_before_merge;
                *tref =
                    self.params.trait_clauses[clause_id].identity_tref_at_depth(self.binder_depth);
            }
        }
    }
    fn enter_region(&mut self, r: &mut Region) {
        self.replace_with_fresh_var(
            r,
            self.concrete && !r.is_erased(),
            |id| RegionParam::new(id, None, Variance::Unknown),
            |v| v.into(),
        );
    }
    // TODO: we're missing type info for this
    // fn enter_const_generic(&mut self, cg: &mut ConstGeneric) {
    //     self.replace_with_fresh_var(cg, |id| {
    //         ConstGenericParam::new(id, format!("N{id}"), cg.ty().clone())
    //     });
    // }
    fn visit_trait_ref(&mut self, tref: &mut TraitRef) -> ControlFlow<Self::Break> {
        // Outside of baked proofs, we don't touch trait refs or we'd risk adding duplicated extra
        // params. Instead, we fix them up in `exit_ty_kind` and `compute_shape`.
        if self.concrete {
            self.visit_nested_proof(tref);
        }
        ControlFlow::Continue(())
    }

    fn visit_constant_expr(
        &mut self,
        _: &mut ConstantExpr,
    ) -> ::std::ops::ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
}

#[derive(Visitor)]
struct PartialMonomorphizer<'a> {
    ctx: &'a mut TransformCtx,
    /// Tracks the closest span to emit useful errors.
    span: Span,
    /// Whether we partially-monomorphize type declarations.
    specialize_adts: bool,
    /// Types that contain mutable references.
    infected_types: HashSet<TypeDeclId>,
    /// The closure state types.
    closure_types: HashSet<TypeDeclId>,
    /// The `Fn`, `FnMut` and `FnOnce` traits.
    fn_traits: HashSet<TraitDeclId>,
    /// Whether we're visiting the `Fn*` impl of a closure or one of its methods.
    in_closure_fn_impl: bool,
    /// Map of generic params for each item. We can't use `ctx.translated` because while iterating
    /// over items the current item isn't available anymore, which would break recursive types.
    /// This also makes it possible to record the generics of our to-be-added items without adding
    /// them.
    generic_params: HashMap<ItemId, GenericParams>,
    /// Map of partial monomorphizations. The source item applied with the generic params gives the
    /// target item. The resulting partially-monomorphized item will have the binder params as
    /// generic params.
    partial_mono_shapes: SeqHashMap<(ItemId, MutabilityShape), ItemId>,
    /// Reverse of `partial_mono_shapes`.
    reverse_shape_map: HashMap<ItemId, (ItemId, MutabilityShape)>,
    /// Items that need to be processed.
    to_process: VecDeque<ItemId>,
}

impl<'a> PartialMonomorphizer<'a> {
    pub fn new(ctx: &'a mut TransformCtx, specialize_adts: bool) -> Self {
        // Compute the types that contain `&mut` (even indirectly). We actually can ignore
        // `&'static mut`, so we simply rely on our "lifetime mutability" computation.
        let infected_types: HashSet<_> = ctx
            .translated
            .type_decls
            .iter()
            .filter(|tdecl| {
                tdecl
                    .generics
                    .regions
                    .iter()
                    .any(|r| r.mutability.is_mutable())
            })
            .map(|tdecl| tdecl.def_id)
            .collect();

        let closure_types = ctx
            .translated
            .type_decls
            .iter()
            .filter(|tdecl| tdecl.src.is_closure())
            .map(|tdecl| tdecl.def_id)
            .collect();
        let fn_traits = ctx
            .translated
            .trait_decls
            .iter()
            .filter(|tdecl| {
                matches!(
                    tdecl.item_meta.lang_item,
                    Some(LangItem::Fn | LangItem::FnMut | LangItem::FnOnce)
                )
            })
            .map(|tdecl| tdecl.def_id)
            .collect();

        // Record the generic params of all items.
        let generic_params: HashMap<ItemId, GenericParams> = ctx
            .translated
            .all_items()
            .map(|item| (item.id(), item.generic_params().clone()))
            .collect();

        // Enqueue all items to be processed.
        let to_process = ctx.translated.all_ids().collect();
        PartialMonomorphizer {
            ctx,
            span: Span::dummy(),
            specialize_adts,
            infected_types,
            closure_types,
            fn_traits,
            in_closure_fn_impl: false,
            generic_params,
            to_process,
            partial_mono_shapes: SeqHashMap::default(),
            reverse_shape_map: Default::default(),
        }
    }

    /// Whether this proof is a concrete impl for an infected type. Specializing an item for such a
    /// proof bakes it in; see `MutabilityShapeBuilder::bake_proof`.
    fn is_bakeable(&self, tref: &PolyTraitRef) -> bool {
        let tref = &tref.0.skip_binder;
        matches!(tref.kind, TraitRefKind::TraitImpl(_))
            && tref
                .trait_decl_ref
                .generics
                .types
                .first()
                .is_some_and(|ty| self.is_infected(ty))
    }

    /// Whether this type is or contains a `&mut`. This assumes that we've already visited this
    /// type and partially monomorphized any ADT references.
    fn is_infected(&self, ty: &Ty) -> bool {
        match ty.kind() {
            TyKind::Ref(_, _, RefKind::Mut) => true,
            TyKind::Ref(_, ty, _)
            | TyKind::RawPtr(ty, _)
            | TyKind::Array(ty, ..)
            | TyKind::Pattern(ty, _)
            | TyKind::Slice(ty, _) => self.is_infected(ty),
            TyKind::Adt(tref) => {
                let ty_infected = self.infected_types.contains(&tref.id);
                let args_infected = if self.specialize_adts {
                    // Since we make sure to only call the method on a processed type, any type
                    // with infected arguments would have been replaced with a fresh instantiated
                    // (and infected type). Hence we don't need to check the arguments here, only
                    // the type id.
                    false
                } else {
                    tref.generics.types.iter().any(|ty| self.is_infected(ty))
                };
                ty_infected || args_infected
            }
            // A function pointer/item by itself doesn't carry any mutable reference, even if it
            // uses some in its signature. Compare with closures: a closure without captures
            // doesn't trigger partial mono regardless of its signature.
            TyKind::FnDef(..) | TyKind::FnPtr(..) => false,
            TyKind::DynTrait(_) => {
                register_error!(
                    self.ctx,
                    self.span,
                    "`dyn Trait` is unsupported with `--monomorphize-mut`"
                );
                false
            }
            TyKind::TypeVar(..)
            | TyKind::Scalar(..)
            | TyKind::Never
            | TyKind::TraitType(..)
            | TyKind::PtrMetadata(..)
            | TyKind::Error(_) => false,
        }
    }

    /// Whether this is a closure state type, or a partial monomorphization of one.
    fn is_closure_type(&self, ty: &Ty) -> bool {
        let TyKind::Adt(tref) = ty.kind() else {
            return false;
        };
        let id = match self.reverse_shape_map.get(&tref.id.into()) {
            Some((ItemId::Type(base_id), _)) => *base_id,
            _ => tref.id,
        };
        self.closure_types.contains(&id)
    }

    /// Whether this is a `Fn*` predicate about a closure.
    fn is_closure_fn_trait(&self, x: &TraitDeclRef) -> bool {
        self.fn_traits.contains(&x.id)
            && x.generics
                .types
                .first()
                .is_some_and(|ty| self.is_closure_type(ty))
    }

    /// Resolve parent clauses of known impls to the corresponding impl proofs. This is what
    /// exposes the concrete impls behind e.g. the `FnOnce` parent clause of a baked `FnMut` proof.
    fn resolve_parent_clauses(&self, tref: &TraitRef) -> Option<TraitRef> {
        const MAX_STEPS: usize = 100;
        let mut tref = tref.clone();
        let mut changed = false;
        for _ in 0..MAX_STEPS {
            let TraitRefKind::ParentClause(parent, clause_id, clause_args) = &tref.kind else {
                break;
            };
            let parent = self
                .resolve_parent_clauses(parent)
                .unwrap_or(parent.clone());
            let TraitRefKind::TraitImpl(impl_ref) = &parent.kind else {
                break;
            };
            let Some(proof) = self
                .ctx
                .translated
                .trait_impls
                .get(impl_ref.id)
                .and_then(|timpl| timpl.implied_trait_refs.get(*clause_id))
            else {
                break;
            };
            tref = ItemBinder::new(impl_ref.id, proof.clone())
                .substitute(ItemBinder::new(CurrentItem, &impl_ref.generics))
                .under_current_binder()
                .apply(clause_args.clone());
            changed = true;
        }
        changed.then_some(tref)
    }

    /// Given that `generics` apply to item `id`, if any of the generics is infected we generate a
    /// reference to a new item obtained by partially instantiating item `id`. (That new item isn't
    /// added immediately but is added to the `to_process` queue to be created later).
    fn process_generics(&mut self, id: ItemId, generics: &GenericArgs) -> Option<DeclRef<ItemId>> {
        if !generics.types.iter().any(|ty| self.is_infected(ty)) {
            return None;
        }

        // If the type is already an instantiation, transform this reference into a reference to
        // the original type so we don't instantiate the instantiation.
        let mut new_generics;
        let (id, generics) = if let Some(&(base_id, ref shape)) = self.reverse_shape_map.get(&id) {
            new_generics = shape.clone().apply(generics);
            let _ = self.visit(&mut new_generics); // New instantiation may require cleanup.
            (base_id, &new_generics)
        } else {
            (id, generics)
        };

        // Split the args between the infected part and the non-infected part.
        let item_params = self.generic_params.get(&id)?;
        let (shape, shape_args) =
            MutabilityShapeBuilder::compute_shape(self, item_params, generics);

        // Create a new type id.
        let new_params = shape.params.clone();
        let key: (ItemId, MutabilityShape) = (id, shape);
        let new_id = *self
            .partial_mono_shapes
            .entry(key.clone())
            .or_insert_with(|| {
                let new_id = match id {
                    ItemId::Type(_) => {
                        let new_id = self.ctx.translated.type_decls.reserve_slot();
                        self.infected_types.insert(new_id);
                        new_id.into()
                    }
                    ItemId::Fun(_) => self.ctx.translated.fun_decls.reserve_slot().into(),
                    ItemId::Global(_) => self.ctx.translated.global_decls.reserve_slot().into(),
                    ItemId::TraitDecl(_) => self.ctx.translated.trait_decls.reserve_slot().into(),
                    ItemId::TraitImpl(_) => self.ctx.translated.trait_impls.reserve_slot().into(),
                };
                self.generic_params.insert(new_id, new_params);
                self.reverse_shape_map.insert(new_id, key);
                self.to_process.push_back(new_id);
                new_id
            });

        let fmt_ctx = self.ctx.into_fmt();
        trace!(
            "processing {}{}\n output: {}{}",
            id.with_ctx(&fmt_ctx),
            generics.with_ctx(&fmt_ctx),
            new_id.with_ctx(&fmt_ctx),
            shape_args.with_ctx(&fmt_ctx),
        );
        Some(DeclRef {
            id: new_id,
            generics: Box::new(shape_args),
            trait_ref: None,
        })
    }

    /// Traverse the item, replacing any type instantiations we don't want with references to
    /// soon-to-be-created partially-monomorphized types. This does not access the items in
    /// `self.translated`, which may be missing since we took `item` out for processing.
    pub fn process_item(&mut self, item: &mut ItemRefMut<'_>) {
        let _ = item.drive_mut(self);
    }

    /// Creates the item corresponding to this id by instantiating the item it is based on.
    ///
    /// This accesses the items in `self.translated`, which must therefore all be there.
    /// That's why items are created outside of `process_item`.
    pub fn create_pending_instantiation(&mut self, new_id: ItemId) -> ItemByVal {
        let (orig_id, shape) = &self.reverse_shape_map[&new_id];
        let mut decl = self
            .ctx
            .translated
            .get_item(*orig_id)
            .unwrap()
            .to_owned()
            .substitute_with_self(&shape.skip_binder, &TraitRefKind::SelfId);

        let mut decl_mut = decl.as_mut();
        decl_mut.set_id(new_id);
        *decl_mut.generic_params() = shape.params.clone();

        let name_ref = &mut decl_mut.item_meta().name;
        *name_ref = mem::take::<crate::ast::Name>(name_ref).instantiate(shape.clone());
        self.ctx
            .translated
            .item_names
            .insert(new_id, decl.as_ref().item_meta().name.clone());
        if let (ItemId::TraitDecl(orig_trait_id), ItemId::TraitDecl(new_trait_id)) =
            (*orig_id, new_id)
        {
            let names = self.ctx.translated.assoc_item_names[orig_trait_id].clone();
            self.ctx
                .translated
                .assoc_item_names
                .insert(new_trait_id, names);
        }

        decl
    }
}

impl VisitorWithSpan for PartialMonomorphizer<'_> {
    fn current_span(&mut self) -> &mut Span {
        &mut self.span
    }
}
impl VisitAstMut for PartialMonomorphizer<'_> {
    fn visit<T: AstVisitable>(&mut self, x: &mut T) -> ControlFlow<Self::Break> {
        // Track a useful enclosing span, for error messages.
        VisitWithSpan::new(self).visit(x)
    }

    fn exit_type_decl_ref(&mut self, x: &mut TypeDeclRef) {
        if x.is_tuple() && self.ctx.options.no_gen_tuple_structs {
            return;
        }
        if self.specialize_adts
            && let Some(new_decl_ref) = self.process_generics(x.id.into(), &x.generics)
        {
            x.id = new_decl_ref.id.try_into().unwrap();
            x.generics = new_decl_ref.generics;
        }
    }
    fn visit_trait_ref(&mut self, x: &mut TraitRef) -> ControlFlow<Self::Break> {
        if let Some(resolved) = self.resolve_parent_clauses(x) {
            *x = resolved;
        }
        // A closure's `Fn*` impls keep implementing the generic `Fn*` traits: specialized items
        // call their methods directly. The other `Fn*` predicates about closures, e.g. clauses
        // whose proof we couldn't bake in, are handled like any other.
        if let TraitRefKind::TraitImpl(_) = x.kind
            && self.is_closure_fn_trait(&x.trait_decl_ref)
        {
            x.with_contents_mut(|contents| self.visit(&mut contents.kind))
        } else {
            self.visit_inner(x)
        }
    }
    fn enter_fn_ptr(&mut self, x: &mut FnPtr) {
        // Call methods of known impls directly. Partial monomorphization substitutes concrete
        // impls for clauses, so this exposes direct calls that may themselves need
        // specialization; `exit_fn_ptr` takes care of that.
        if let FnPtrKind::Trait(tref, method_id) = x.kind.as_ref() {
            let tref = self
                .resolve_parent_clauses(tref)
                .unwrap_or_else(|| tref.clone());
            if let TraitRefKind::TraitImpl(impl_ref) = &tref.kind
                && let Some(timpl) = self.ctx.translated.trait_impls.get(impl_ref.id)
                && let Some(fn_ptr) = impl_method_call(timpl, impl_ref, *method_id, &x.generics)
            {
                *x = fn_ptr;
            }
        }
    }
    fn exit_fn_ptr(&mut self, x: &mut FnPtr) {
        // TODO: methods on clauses. any `Trait::method<&mut A>` requires monomorphizing all the
        // instances of that method just in case :>>>
        if let FnPtrKind::Fun(id) = *x.kind
            && let Some(new_decl_ref) = self.process_generics(id.into(), &x.generics)
        {
            *x = new_decl_ref.try_into().unwrap()
        }
    }
    fn exit_fun_decl_ref(&mut self, x: &mut FunDeclRef) {
        if let Some(new_decl_ref) = self.process_generics(x.id.into(), &x.generics) {
            *x = new_decl_ref.try_into().unwrap()
        }
    }
    fn exit_global_decl_ref(&mut self, x: &mut GlobalDeclRef) {
        if let Some(new_decl_ref) = self.process_generics(x.id.into(), &x.generics) {
            *x = new_decl_ref.try_into().unwrap()
        }
    }
    fn visit_trait_impl(&mut self, x: &mut TraitImpl) -> ControlFlow<Self::Break> {
        // The `Fn*` impls of a closure implement the generic traits (see `visit_trait_ref`), so
        // the proofs of the trait's parent clauses must stay about the generic traits too.
        if self.is_closure_fn_trait(&x.impl_trait) {
            let implied_trait_refs = mem::take(&mut x.implied_trait_refs);
            self.in_closure_fn_impl = true;
            let res = self.visit_inner(x);
            self.in_closure_fn_impl = false;
            x.implied_trait_refs = implied_trait_refs;
            res
        } else {
            self.visit_inner(x)
        }
    }
    fn visit_fun_source(&mut self, x: &mut FunSource) -> ControlFlow<Self::Break> {
        // The methods of the `Fn*` impls of a closure, see `visit_trait_impl`.
        if let FunSource::TraitImpl { trait_ref, .. } = x
            && self.is_closure_fn_trait(trait_ref)
        {
            self.in_closure_fn_impl = true;
            let res = self.visit_inner(x);
            self.in_closure_fn_impl = false;
            res
        } else {
            self.visit_inner(x)
        }
    }
    fn exit_trait_decl_ref(&mut self, x: &mut TraitDeclRef) {
        if self.in_closure_fn_impl && self.is_closure_fn_trait(x) {
            return;
        }
        if let Some(new_decl_ref) = self.process_generics(x.id.into(), &x.generics) {
            *x = new_decl_ref.try_into().unwrap()
        }
    }
    fn exit_trait_impl_ref(&mut self, x: &mut TraitImplRef) {
        if let Some(new_decl_ref) = self.process_generics(x.id.into(), &x.generics) {
            *x = new_decl_ref.try_into().unwrap()
        }
    }
}

pub struct Transform;
impl TransformPass for Transform {
    fn transform_ctx(&self, ctx: &mut TransformCtx) {
        let Some(include_types) = ctx.options.monomorphize_mut else {
            return;
        };
        // TODO: test name matcher, also with methods
        let mut visitor =
            PartialMonomorphizer::new(ctx, matches!(include_types, MonomorphizeMut::All));
        while let Some(id) = visitor.to_process.pop_front() {
            // Get the item corresponding to this id, either by creating it or by getting an
            // existing one.
            let mut decl = if visitor.reverse_shape_map.contains_key(&id) {
                // Create the required item by instantiating the item it's based on.
                visitor.create_pending_instantiation(id)
            } else {
                // Take the item out so we can modify it. Warning: don't look up other items in the
                // meantime as this would break in recursive cases.
                match visitor.ctx.translated.remove_item_temporarily(id) {
                    Some(decl) => decl,
                    None => continue,
                }
            };
            // Visit the item, replacing type instantiations with references to soon-to-be-created
            // partially-monomorphized types.
            visitor.process_item(&mut decl.as_mut());
            // Put the item back.
            visitor.ctx.translated.put_item_back(id, decl);
        }
        // An instantiated trait declaration gets all the methods of the declaration it
        // instantiates, including the ones only its other instantiations use.
        if !ctx.options.translate_all_methods {
            crate::transform::simplify_output::remove_unmentioned_methods::remove_unmentioned_methods(
                &mut ctx.translated,
            );
        }
    }
}
