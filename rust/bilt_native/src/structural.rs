//! Experimental native structural planner. No Python objects in the search state.
use pyo3::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashSet};

type Id = usize;
type Set = BTreeSet<Id>;
type Step = (Id, Id, Vec<Id>); // operation index, output, qualified child inputs
type RawRow = (Id, Vec<Id>, Option<Id>, Option<Id>); // local index, ports, transform output, child context

// A child route is a dependency query, not a new request with all candidate
// ports fixed. Validate its actual consumed boundary before accepting it.
enum PlanTargets<'a> {
    Strict(&'a [Id]),
    Validated(Id, Vec<(Id, Id)>),
}

#[derive(Clone, Copy)]
enum ScopeFailure {
    Overdetermined,
    Underdetermined,
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
    fn validated_scope(
        &self,
        root: Id,
        available: &Set,
        target: Id,
    ) -> Result<Vec<(Id, Id)>, ScopeFailure> {
        if available.contains(&target) {
            return Ok(Vec::new());
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
            return Err(ScopeFailure::Overdetermined);
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
                return Err(ScopeFailure::Underdetermined);
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
        Ok(selected)
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

    // An overdetermined candidate boundary may still contain a valid
    // smaller boundary. Probe only forward-reachable primitive directions:
    // no recursive settlement or alternative matching search in this probe.
    fn discover_boundary(&self, context: Id, available: &Set, target: Id) -> Option<Vec<Id>> {
        let (_, scope) = self.scope(context, available, target);
        let producers = self.preferred(&scope, available);
        let mut pending = vec![target];
        let mut seen = Set::new();
        let mut required = Set::new();
        while let Some(port) = pending.pop() {
            if !seen.insert(port) {
                continue;
            }
            if available.contains(&port) {
                required.insert(port);
                continue;
            }
            let &(owner, index) = producers.get(&port)?;
            let row = &self.contexts[owner].rows[index];
            pending.extend(row.ports.iter().copied().filter(|&p| p != port));
        }
        self.validated_scope(context, &required, target).ok()?;
        Some(required.into_iter().collect())
    }

    /// Preserve the baseline search for valid boundaries. Only probe a
    /// smaller boundary when supplied candidates overdetermine the target.
    fn route_inputs(
        &self,
        context: Id,
        available: &Set,
        target: Id,
        budget: &mut usize,
        depth: usize,
    ) -> Option<Vec<Id>> {
        if depth > 64 || *budget == 0 {
            return None;
        }
        let scope = match self.validated_scope(context, available, target) {
            Ok(scope) => scope,
            Err(ScopeFailure::Underdetermined) => return None,
            Err(ScopeFailure::Overdetermined) => {
                return self.discover_boundary(context, available, target);
            }
        };
        let plan = self.plan_inner(
            context,
            available,
            PlanTargets::Validated(target, scope),
            &BTreeSet::new(),
            budget,
            depth,
        )?;
        let mut required = Set::new();
        for (operation, output, inputs) in plan {
            let source = self.contexts[context]
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
        request: PlanTargets<'_>,
        blocked: &BTreeSet<(Id, Id)>,
        budget: &mut usize,
        depth: usize,
    ) -> Option<Vec<Step>> {
        if depth > 64 || *budget == 0 {
            return None;
        }
        *budget -= 1;
        let targets = match &request {
            PlanTargets::Strict(targets) => *targets,
            PlanTargets::Validated(target, _) => std::slice::from_ref(target),
        };
        let mut scope = BTreeSet::new();
        match &request {
            PlanTargets::Strict(_) => {
                for &target in targets {
                    scope.extend(self.validated_scope(context, available, target).ok()?);
                }
            }
            PlanTargets::Validated(_, rows) => scope.extend(rows.iter().copied()),
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
            if *budget == 0 {
                break;
            }
            *budget -= 1;
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
                budget,
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
    budget: &'a mut usize,
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
            .route_inputs(child, &available, port, self.budget, self.depth + 1)
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
        let mut budget = 4096;
        self.plan_inner(
            0,
            &available.into_iter().collect(),
            PlanTargets::Strict(&targets),
            &blocked.into_iter().collect(),
            &mut budget,
            0,
        )
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
    fn discovery_validates_consumed_boundary_not_all_candidate_ports() {
        let model = NativeStructuralModel::new(vec![vec![
            (0, vec![0, 2], None, None),
            (1, vec![1, 2], None, None),
        ]])
        .unwrap();
        let available = Set::from([0, 1]);
        // Both candidates fixed makes the original request overdetermined.
        assert!(model.plan(vec![0, 1], vec![2], vec![]).is_none());
        let inputs = model
            .route_inputs(0, &available, 2, &mut 4096, 0)
            .unwrap();
        assert_eq!(inputs.len(), 1);
        assert!(model.plan(inputs, vec![2], vec![]).is_some());
    }

    #[test]
    fn discovery_cannot_accept_overdetermined_consumed_boundary() {
        let model = NativeStructuralModel::new(vec![vec![
            (0, vec![0, 1], None, None),
            (1, vec![0, 1], None, None),
        ]])
        .unwrap();
        assert!(model
            .route_inputs(0, &Set::from([0]), 1, &mut 4096, 0)
            .is_none());
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
