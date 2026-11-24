//! Experimental native structural planner. No Python objects in the search state.
use pyo3::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

type Id = usize;
type Set = BTreeSet<Id>;
type Step = (Id, Id, Vec<Id>); // operation index, output, qualified child inputs
type RawRow = (Id, Vec<Id>, Option<Id>, Option<Id>); // local index, ports, transform output, child context

type RouteKey = (Id, Id, usize, Vec<Id>); // context, target, depth, available
type PlanReport = (Option<Vec<Step>>, BTreeMap<String, usize>);

struct CachedRoute {
    inputs: Option<Vec<Id>>,
    cost: usize,
}

// A cache belongs to one top-level plan, never to a model or a numerical
// evaluation. Retain the logical search budget even when replaying a result.
struct Search {
    budget: usize,
    cache_enabled: bool,
    routes: HashMap<RouteKey, CachedRoute>,
    observed: HashMap<RouteKey, u8>,
    queries: usize,
    successes: usize,
    success_hits: usize,
    failure_hits: usize,
    budget_skips: usize,
}

impl Search {
    fn new(cache_enabled: bool) -> Self {
        Self {
            budget: 4096,
            cache_enabled,
            routes: HashMap::new(),
            observed: HashMap::new(),
            queries: 0,
            successes: 0,
            success_hits: 0,
            failure_hits: 0,
            budget_skips: 0,
        }
    }

    fn record(&mut self, key: RouteKey, success: bool) {
        let flag = if success { 1 } else { 2 };
        *self.observed.entry(key).or_default() |= flag;
        self.successes += usize::from(success);
    }

    fn stats(&self) -> BTreeMap<String, usize> {
        [
            ("child_route_queries", self.queries),
            ("child_route_successes", self.successes),
            ("child_route_failures", self.queries - self.successes),
            ("child_route_unique", self.observed.len()),
            (
                "child_route_unique_success",
                self.observed
                    .values()
                    .filter(|&&flag| flag & 1 != 0)
                    .count(),
            ),
            (
                "child_route_unique_failure",
                self.observed
                    .values()
                    .filter(|&&flag| flag & 2 != 0)
                    .count(),
            ),
            ("child_route_success_hits", self.success_hits),
            ("child_route_failure_hits", self.failure_hits),
            ("child_route_budget_skips", self.budget_skips),
            ("child_route_cache_entries", self.routes.len()),
            ("logical_search_budget_used", 4096 - self.budget),
        ]
        .into_iter()
        .map(|(name, count)| (name.to_owned(), count))
        .collect()
    }
}

#[derive(Clone)]
struct Row {
    operation: Id,
    ports: Vec<Id>,
    output: Option<Id>,
    child: Option<Id>,
}

struct Context {
    rows: Vec<Row>,
    adjacent: BTreeMap<Id, Vec<Id>>,
}

/// Immutable, qualified integer-ID model, shared across requests.
#[pyclass]
pub struct NativeStructuralModel {
    contexts: Vec<Context>,
}

fn augment(row: Id, edges: &[Vec<Id>], seen: &mut Set, mates: &mut BTreeMap<Id, Id>) -> bool {
    for &port in &edges[row] {
        if !seen.insert(port) {
            continue;
        }
        let previous = mates.get(&port).copied();
        if previous.is_none_or(|other| augment(other, edges, seen, mates)) {
            mates.insert(port, row);
            return true;
        }
    }
    false
}

fn matching(edges: &[Vec<Id>]) -> BTreeMap<Id, Id> {
    let mut mates = BTreeMap::new();
    for row in 0..edges.len() {
        augment(row, edges, &mut Set::new(), &mut mates);
    }
    mates
}

// A complete request-scoped expansion. Each open context sees qualified ports;
// a child opens only when one of its unknown boundary ports is encountered.
impl NativeStructuralModel {
    fn scope(&self, root: Id, available: &Set, target: Id) -> (Set, Vec<(Id, Id)>) {
        let mut attributes = Set::from([target]);
        let mut contexts = Set::from([root]);
        let mut selected = BTreeSet::new();
        let mut seen = BTreeSet::new();
        let mut pending = vec![(root, target)];
        while let Some((context, port)) = pending.pop() {
            if available.contains(&port) || !seen.insert((context, port)) {
                continue;
            }
            if let Some(rows) = self.contexts[context].adjacent.get(&port) {
                for &index in rows {
                    let row = &self.contexts[context].rows[index];
                    if let Some(child) = row.child {
                        if contexts.insert(child) {
                            pending.extend(attributes.iter().map(|&p| (child, p)));
                        }
                        continue;
                    }
                    if row.output.is_some_and(|output| output != port)
                        || !selected.insert((context, index))
                    {
                        continue;
                    }
                    for &p in &row.ports {
                        if attributes.insert(p) {
                            pending.extend(contexts.iter().map(|&c| (c, p)));
                        }
                    }
                }
            }
        }
        (attributes, selected.into_iter().collect())
    }

    // DM classification without constructing a Python graph. Overdetermination
    // anywhere in the relevant scope, or underdetermination of the target,
    // causes a reference-engine fallback (which owns diagnostic messages).
    fn validated_scope(&self, root: Id, available: &Set, target: Id) -> Option<Vec<(Id, Id)>> {
        if available.contains(&target) {
            return Some(Vec::new());
        }
        let (attributes, selected) = self.scope(root, available, target);
        let unknown: Set = attributes.difference(available).copied().collect();
        let edges: Vec<Vec<Id>> = selected
            .iter()
            .map(|&(c, r)| {
                self.contexts[c].rows[r]
                    .ports
                    .iter()
                    .copied()
                    .filter(|p| !available.contains(p))
                    .collect()
            })
            .collect();
        let mates = matching(&edges);
        // Any unmatched operation is in the overdetermined region.
        if mates.len() != edges.len() {
            return None;
        }
        // Follow alternating edges from every unmatched attribute.
        let mut pending: Vec<_> = unknown
            .iter()
            .copied()
            .filter(|p| !mates.contains_key(p))
            .collect();
        let mut reached: Set = pending.iter().copied().collect();
        let row_mates: BTreeMap<_, _> = mates.iter().map(|(&p, &r)| (r, p)).collect();
        while let Some(port) = pending.pop() {
            if port == target {
                return None;
            }
            for (row, ports) in edges.iter().enumerate() {
                if ports.contains(&port) && mates.get(&port) != Some(&row) {
                    if let Some(&next) = row_mates.get(&row) {
                        if reached.insert(next) {
                            pending.push(next);
                        }
                    }
                }
            }
        }
        Some(selected)
    }

    // Forward producers are a preference only; every other candidate edge is
    // retained for retries. The prototype uses a deterministic bipartite
    // augmenting-path matcher rather than rustworkx's weighted blossom.
    fn preferred(&self, scope: &[(Id, Id)], available: &Set) -> BTreeMap<Id, (Id, Id)> {
        let mut reached = available.clone();
        let mut producers = BTreeMap::new();
        loop {
            let before = reached.len();
            for &(context, index) in scope {
                let row = &self.contexts[context].rows[index];
                let mut unknown = row.ports.iter().filter(|p| !reached.contains(*p));
                if let Some(&port) = unknown.next() {
                    if unknown.next().is_none() && row.output.is_none_or(|output| output == port) {
                        reached.insert(port);
                        producers.insert(port, (context, index));
                    }
                }
            }
            if reached.len() == before {
                return producers;
            }
        }
    }

    fn route_inputs(
        &self,
        child: Id,
        available: &Set,
        port: Id,
        search: &mut Search,
        depth: usize,
    ) -> Option<Vec<Id>> {
        search.queries += 1;
        let key = (child, port, depth, available.iter().copied().collect());
        if search.cache_enabled {
            if let Some(cached) = search.routes.get(&key) {
                // A result computed with spare budget can be replayed only
                // when the caller has enough budget to reproduce that search.
                if search.budget > cached.cost {
                    let result = cached.inputs.clone();
                    search.budget -= cached.cost;
                    if result.is_some() {
                        search.success_hits += 1;
                    } else {
                        search.failure_hits += 1;
                    }
                    search.record(key, result.is_some());
                    return result;
                }
            }
        }
        let before = search.budget;
        let result = self.uncached_route(child, available, port, search, depth);
        search.record(key.clone(), result.is_some());
        // Budget-exhausted searches are not evidence of an impossible route.
        if search.budget == 0 {
            search.budget_skips += 1;
        } else if search.cache_enabled {
            search.routes.insert(
                key,
                CachedRoute {
                    inputs: result.clone(),
                    cost: before - search.budget,
                },
            );
        }
        result
    }

    fn uncached_route(
        &self,
        child: Id,
        available: &Set,
        port: Id,
        search: &mut Search,
        depth: usize,
    ) -> Option<Vec<Id>> {
        let plan = self.plan_inner(child, available, &[port], &BTreeSet::new(), search, depth)?;
        let mut required = Set::new();
        for (operation, output, inputs) in plan {
            let source = self.contexts[child]
                .rows
                .iter()
                .find(|r| r.operation == operation)?;
            if source.child.is_some() {
                required.extend(inputs.into_iter().filter(|p| available.contains(p)));
            } else {
                required.extend(
                    source
                        .ports
                        .iter()
                        .copied()
                        .filter(|&p| p != output && available.contains(&p)),
                );
            }
        }
        Some(required.into_iter().collect())
    }

    fn plan_inner(
        &self,
        context: Id,
        available: &Set,
        targets: &[Id],
        blocked: &BTreeSet<(Id, Id)>,
        search: &mut Search,
        depth: usize,
    ) -> Option<Vec<Step>> {
        if depth > 64 || search.budget == 0 {
            return None;
        }
        search.budget -= 1;
        let mut scope = BTreeSet::new();
        for &target in targets {
            scope.extend(self.validated_scope(context, available, target)?);
        }
        let preferred = self.preferred(&scope.into_iter().collect::<Vec<_>>(), available);
        let rows = &self.contexts[context].rows;
        let mut edges = Vec::new();
        let mut owners = Vec::new();
        for (index, row) in rows.iter().enumerate() {
            let mut ports: Vec<_> = row
                .ports
                .iter()
                .copied()
                .filter(|p| {
                    !available.contains(p)
                        && !blocked.contains(&(row.operation, *p))
                        && row.output.is_none_or(|output| output == *p)
                })
                .collect();
            // Prefer locally reachable directions without excluding alternatives.
            ports.sort_by_key(|p| {
                let direct = row.ports.iter().all(|q| q == p || available.contains(q));
                (preferred.get(p) != Some(&(context, index)), !direct, *p)
            });
            if row.child.is_some() {
                for port in ports {
                    edges.push(vec![port]);
                    owners.push(index);
                }
            } else {
                edges.push(ports);
                owners.push(index);
            }
        }
        let mut pending = vec![edges];
        let mut visited = HashSet::new();
        // Bound search so unsupported or difficult models return to Python.
        for _ in 0..24 {
            let Some(candidate) = pending.pop() else {
                break;
            };
            if !visited.insert(candidate.clone()) {
                continue;
            }
            if search.budget == 0 {
                break;
            }
            search.budget -= 1;
            let matched = matching(&candidate);
            let assignments: BTreeMap<_, _> =
                matched.iter().map(|(&p, &r)| (p, owners[r])).collect();
            let mut settlement = Settlement {
                model: self,
                context,
                assignments: &assignments,
                fixed: available,
                established: available.clone(),
                active: Set::new(),
                steps: Vec::new(),
                search,
                depth,
            };
            if targets.iter().all(|&port| settlement.resolve(port)) {
                // Settlement emits producers first; discard exploratory sibling
                // steps not in the final union of requested dependencies.
                let mut needed: Set = targets.iter().copied().collect();
                let mut steps = Vec::new();
                for step in settlement.steps.into_iter().rev() {
                    if !needed.contains(&step.1) {
                        continue;
                    }
                    let row = &rows[assignments[&step.1]];
                    if row.child.is_some() {
                        needed.extend(step.2.iter().copied());
                    } else {
                        needed.extend(row.ports.iter().copied().filter(|&p| p != step.1));
                    }
                    steps.push(step);
                }
                steps.reverse();
                return Some(steps);
            }
            // Alternative matchings exclude one selected edge, preserving all
            // other candidates. This is bounded experimental search, not a
            // guarantee of finding every reference-engine route.
            for (&port, &row) in &matched {
                let mut alternative = candidate.clone();
                alternative[row].retain(|&p| p != port);
                pending.push(alternative);
            }
        }
        None
    }
}

struct Settlement<'a> {
    model: &'a NativeStructuralModel,
    context: Id,
    assignments: &'a BTreeMap<Id, Id>,
    fixed: &'a Set,
    established: Set,
    active: Set,
    steps: Vec<Step>,
    search: &'a mut Search,
    depth: usize,
}

impl Settlement<'_> {
    fn resolve(&mut self, port: Id) -> bool {
        if self.established.contains(&port) {
            return true;
        }
        if self.active.len() > 512 || !self.active.insert(port) {
            return false;
        }
        let result = self.resolve_inner(port);
        self.active.remove(&port);
        result
    }

    fn resolve_inner(&mut self, port: Id) -> bool {
        let Some(&index) = self.assignments.get(&port) else {
            return false;
        };
        let row = self.model.contexts[self.context].rows[index].clone();
        let inputs = if let Some(child) = row.child {
            // First settle non-child producers, then try the child with the
            // available boundary. Same-child derived outputs are not fixed.
            for &p in &row.ports {
                if p != port
                    && self
                        .assignments
                        .get(&p)
                        .is_some_and(|&r| self.model.contexts[self.context].rows[r].child.is_none())
                {
                    self.resolve(p);
                }
            }
            let mut route = self.child_route(&row, child, port, false);
            if route.is_none() {
                for &p in &row.ports {
                    if p != port && !self.established.contains(&p) {
                        self.resolve(p);
                        route = self.child_route(&row, child, port, false);
                        if route.is_some() {
                            break;
                        }
                    }
                }
            }
            let Some(inputs) = route.or_else(|| self.child_route(&row, child, port, true)) else {
                return false;
            };
            inputs
        } else {
            for &p in &row.ports {
                if p != port && !self.resolve(p) {
                    return false;
                }
            }
            Vec::new()
        };
        self.established.insert(port);
        self.steps.push((row.operation, port, inputs));
        true
    }

    fn child_route(&mut self, row: &Row, child: Id, port: Id, reduced: bool) -> Option<Vec<Id>> {
        let available: Set = row
            .ports
            .iter()
            .copied()
            .filter(|p| {
                *p != port
                    && self.established.contains(p)
                    && (!reduced
                        || self.fixed.contains(p)
                        || self.assignments.get(p).is_none_or(|&r| {
                            self.model.contexts[self.context].rows[r].child != Some(child)
                        }))
            })
            .collect();
        self.model
            .route_inputs(child, &available, port, self.search, self.depth + 1)
    }
}

#[pymethods]
impl NativeStructuralModel {
    #[new]
    fn new(contexts: Vec<Vec<RawRow>>) -> PyResult<Self> {
        let count = contexts.len();
        if count == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "missing root context",
            ));
        }
        let mut result = Vec::new();
        for (context, raw) in contexts.into_iter().enumerate() {
            let mut rows = Vec::new();
            let mut adjacent: BTreeMap<Id, Vec<Id>> = BTreeMap::new();
            for (operation, ports, output, child) in raw {
                if child.is_some_and(|c| c <= context || c >= count) {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "invalid child context",
                    ));
                }
                let index = rows.len();
                for &p in &ports {
                    adjacent.entry(p).or_default().push(index);
                }
                rows.push(Row {
                    operation,
                    ports,
                    output,
                    child,
                });
            }
            result.push(Context { rows, adjacent });
        }
        Ok(Self { contexts: result })
    }

    /// Return an acyclic candidate schedule or None for reference fallback.
    fn plan(
        &self,
        available: Vec<Id>,
        targets: Vec<Id>,
        blocked: Vec<(Id, Id)>,
    ) -> Option<Vec<Step>> {
        self.plan_with_stats(available, targets, blocked, true).0
    }

    /// Compare cache-on and cache-off without changing the search budget.
    fn plan_with_stats(
        &self,
        available: Vec<Id>,
        targets: Vec<Id>,
        blocked: Vec<(Id, Id)>,
        cache_enabled: bool,
    ) -> PlanReport {
        let mut search = Search::new(cache_enabled);
        let result = self.plan_inner(
            0,
            &available.into_iter().collect(),
            &targets,
            &blocked.into_iter().collect(),
            &mut search,
            0,
        );
        (result, search.stats())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_and_joint_targets() {
        let model = NativeStructuralModel::new(vec![vec![
            (0, vec![0, 1], None, None),
            (1, vec![1, 2], None, None),
        ]])
        .unwrap();
        assert_eq!(
            model.plan(vec![0], vec![1, 2], vec![]),
            Some(vec![(0, 1, vec![]), (1, 2, vec![])])
        );
        assert!(model.plan(vec![], vec![2], vec![]).is_none());
        assert!(model.plan(vec![0, 1], vec![2], vec![]).is_some());
    }

    #[test]
    fn nested_child_keeps_required_boundary() {
        let model = NativeStructuralModel::new(vec![
            vec![(0, vec![0, 1], None, Some(1)), (1, vec![1, 2], None, None)],
            vec![(0, vec![0, 1], None, None)],
        ])
        .unwrap();
        assert_eq!(
            model.plan(vec![0], vec![2], vec![]),
            Some(vec![(0, 1, vec![0]), (1, 2, vec![])])
        );
    }

    #[test]
    fn cache_replays_success_and_failure_with_same_logical_budget() {
        let model = NativeStructuralModel::new(vec![vec![(0, vec![0, 1], None, None)]]).unwrap();
        let mut cached = Search::new(true);
        let mut control = Search::new(false);
        for available in [Set::from([0]), Set::new(), Set::from([0]), Set::new()] {
            assert_eq!(
                model.route_inputs(0, &available, 1, &mut cached, 1),
                model.route_inputs(0, &available, 1, &mut control, 1)
            );
            assert_eq!(cached.budget, control.budget);
        }
        assert_eq!(cached.success_hits, 1);
        assert_eq!(cached.failure_hits, 1);
        assert_eq!(cached.observed.len(), 2);
    }

    #[test]
    fn exhausted_budget_does_not_poison_cache() {
        let model = NativeStructuralModel::new(vec![vec![(0, vec![0, 1], None, None)]]).unwrap();
        let mut search = Search::new(true);
        search.budget = 1;
        assert!(model
            .route_inputs(0, &Set::from([0]), 1, &mut search, 1)
            .is_none());
        assert!(search.routes.is_empty());
        search.budget = 4096;
        assert_eq!(
            model.route_inputs(0, &Set::from([0]), 1, &mut search, 1),
            Some(vec![0])
        );
        search.budget = 1;
        assert!(model
            .route_inputs(0, &Set::from([0]), 1, &mut search, 1)
            .is_none());
        assert_eq!(search.success_hits, 0);
    }

    #[test]
    fn nested_plan_and_logical_budget_are_identical_with_cache() {
        let model = NativeStructuralModel::new(vec![
            vec![(0, vec![0, 1], None, Some(1)), (1, vec![1, 2], None, None)],
            vec![(0, vec![0, 1], None, None)],
        ])
        .unwrap();
        let (control, control_stats) = model.plan_with_stats(vec![0], vec![2], vec![], false);
        let (cached, cached_stats) = model.plan_with_stats(vec![0], vec![2], vec![], true);
        assert_eq!(control, cached);
        assert_eq!(
            control_stats["logical_search_budget_used"],
            cached_stats["logical_search_budget_used"]
        );
    }

    #[test]
    fn duplicate_constraints_and_transform_direction() {
        let model = NativeStructuralModel::new(vec![vec![
            (0, vec![0, 1], None, None),
            (1, vec![0, 1], None, None),
        ]])
        .unwrap();
        assert!(model.plan(vec![0], vec![1], vec![]).is_none());
        let transform =
            NativeStructuralModel::new(vec![vec![(0, vec![0, 1], Some(1), None)]]).unwrap();
        assert!(transform.plan(vec![1], vec![0], vec![]).is_none());
    }
}
