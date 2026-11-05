//! Remove the default trait methods that aren't mentioned anywhere.
use petgraph::prelude::DiGraphMap;
use petgraph::visit::{Dfs, Walker};
use std::collections::{HashMap, HashSet};

use crate::ast::*;

/// Emulate the behavior of our lazy method translation scheme by removing default trait methods
/// that aren't usefully mentioned anywhere.
pub fn remove_unmentioned_methods(krate: &mut TranslatedCrate) {
    type MethodKey = (TraitDeclId, TraitMethodId);

    use ReachabilityNode::*;
    #[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    enum ReachabilityNode {
        Root,
        Method(MethodKey),
        Fun(FunDeclId),
    }

    #[derive(Visitor)]
    struct MentionedFunVisitor<F>(F);

    impl<F> VisitAst for MentionedFunVisitor<F>
    where
        F: FnMut(ReachabilityNode),
    {
        fn enter_fun_decl_id(&mut self, id: &FunDeclId) {
            (self.0)(Fun(*id));
        }

        fn enter_fn_ptr(&mut self, fn_ptr: &FnPtr) {
            if let FnPtrKind::Trait(trait_ref, method_id) = fn_ptr.kind.as_ref() {
                (self.0)(Method((trait_ref.trait_id(), *method_id)));
            }
        }
    }

    // Build a graph where the items we want to keep are reachable from the root. To start with
    // that's all the `FunDeclId`s that aren't a method (or a target-dispatch target coming from a
    // method), as well as all the methods without default. We end up with a graph where methods
    // with a default may end up not reachable.
    let graph = {
        let mut graph: DiGraphMap<ReachabilityNode, ()> = DiGraphMap::new();
        graph.add_node(Root);

        for (fun_id, fun) in krate.fun_decls.iter_indexed() {
            let fun_node = Fun(fun_id);
            graph.add_node(fun_node);

            if let FunSource::TraitDefault {
                trait_ref, item_id, ..
            }
            | FunSource::TraitImpl {
                trait_ref, item_id, ..
            } = &fun.src
            {
                let method_key = (trait_ref.id, *item_id);
                // The method node is reachable iff any of the corresponding function nodes is.
                graph.add_edge(Method(method_key), fun_node, ());
                graph.add_edge(fun_node, Method(method_key), ());
            }

            match &fun.src {
                FunSource::TraitDefault { .. }
                | FunSource::TraitImpl { .. }
                | FunSource::TargetDependent { .. } => {}
                // Functions that aren't any of the above are reachable. target-dependent functions
                // will be reachable if their dispatcher is.
                _ => {
                    graph.add_edge(Root, fun_node, ());
                }
            }

            let _ = fun.body.drive(&mut MentionedFunVisitor(|n| {
                graph.add_edge(fun_node, n, ());
            }));
        }

        for trait_decl in krate.trait_decls.iter() {
            for (method_id, method) in trait_decl.methods.iter_enumerated() {
                if method.skip_binder.default.is_none() {
                    graph.add_edge(Root, Method((trait_decl.def_id, method_id)), ());
                }
            }
        }

        graph
    };

    let reachable_nodes: HashSet<_> = Dfs::new(&graph, Root).iter(&graph).collect();

    let mut unused_methods: HashMap<TraitDeclId, HashSet<TraitMethodId>> = HashMap::new();
    // Iterate over unreachable nodes.
    for n in graph.nodes().filter(|n| !reachable_nodes.contains(n)) {
        match n {
            Root => {}
            Method((trait_id, method_id)) => {
                unused_methods
                    .entry(trait_id)
                    .or_default()
                    .insert(method_id);
            }
            Fun(fun_id) => {
                // Remove unreachable functions.
                krate.remove_item(ItemId::Fun(fun_id));
            }
        }
    }
    if unused_methods.is_empty() {
        return;
    }

    // Remove unreachable methods from both decls and impls.
    for trait_impl in krate.trait_impls.iter_mut() {
        let trait_id = trait_impl.impl_trait.id;
        if let Some(unused_methods) = unused_methods.get(&trait_id) {
            trait_impl
                .methods
                .retain(|method_id, _| !unused_methods.contains(&method_id));
        }
    }
    for (trait_id, unused_methods) in unused_methods {
        if let Some(trait_decl) = krate.trait_decls.get_mut(trait_id) {
            trait_decl
                .methods
                .retain(|method_id, _| !unused_methods.contains(&method_id));
        }
    }
}
