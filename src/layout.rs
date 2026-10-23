//! Layered graph placement for one ONNX graph.
//!
//! Initializers stay off the canvas unless requested. A real model carries one
//! weight tensor per parameter; drawing each of them hides the operator graph.
//! Their dtype and shape still show up on the operator that reads them.

use crate::onnx::{Model, TensorInfo, ValueInfo};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Input,
    Initializer,
    Operator,
    Output,
    Unresolved,
}

pub struct Scene {
    pub nodes: Vec<SceneNode>,
    pub edges: Vec<(usize, usize)>,
    pub width: f64,
    pub height: f64,
}

pub struct SceneNode {
    pub id: String,
    pub kind: Kind,
    pub op_type: String,
    pub title: String,
    pub lines: Vec<String>,
    pub detail: Vec<String>,
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

const NODE_W: f64 = 196.0;
const GAP_X: f64 = 28.0;
const GAP_Y: f64 = 72.0;
const MARGIN: f64 = 28.0;

pub fn layout(model: &Model, show_weights: bool) -> Scene {
    let graph = &model.graph;
    let initializers: HashMap<&str, &TensorInfo> = graph
        .initializers
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect();
    let types = type_index(graph);

    let mut nodes: Vec<Draft> = Vec::new();
    let mut producer: HashMap<String, usize> = HashMap::new();

    for input in &graph.inputs {
        if input.name.is_empty() || initializers.contains_key(input.name.as_str()) {
            continue;
        }
        let idx = nodes.len();
        producer.insert(input.name.clone(), idx);
        nodes.push(Draft::value(
            Kind::Input,
            &input.name,
            type_of(&types, input),
        ));
    }

    if show_weights {
        for tensor in &graph.initializers {
            if tensor.name.is_empty() {
                continue;
            }
            let idx = nodes.len();
            producer.insert(tensor.name.clone(), idx);
            let mut draft = Draft::value(Kind::Initializer, &tensor.name, tensor.type_text.clone());
            if tensor.external {
                draft.detail.push("数据在外部文件".into());
            }
            nodes.push(draft);
        }
    }

    let op_start = nodes.len();
    for (ordinal, node) in graph.nodes.iter().enumerate() {
        let title = if node.op_type.is_empty() {
            format!("op {ordinal}")
        } else {
            node.op_type.clone()
        };
        let mut lines = Vec::new();
        if !node.name.is_empty() {
            lines.push(node.name.clone());
        }
        let mut weights = Vec::new();
        for input in &node.inputs {
            if let Some(tensor) = initializers.get(input.as_str()) {
                let mut label = format!("{} {}", tensor.name, tensor.type_text);
                if tensor.external {
                    label.push_str(" external");
                }
                weights.push(label);
            }
        }
        if !show_weights && !weights.is_empty() {
            lines.push(format!("权重 × {}", weights.len()));
        }
        for attribute in node.attributes.iter().take(3) {
            if attribute.value.is_empty() {
                lines.push(attribute.name.clone());
            } else {
                lines.push(format!("{} = {}", attribute.name, attribute.value));
            }
        }
        if node.attributes.len() > 3 {
            lines.push(format!("… 另有 {} 个属性", node.attributes.len() - 3));
        }

        let mut detail = Vec::new();
        if !node.name.is_empty() {
            detail.push(format!("name: {}", node.name));
        }
        if !node.domain.is_empty() {
            detail.push(format!("domain: {}", node.domain));
        }
        for input in &node.inputs {
            if input.is_empty() {
                detail.push("input: (optional)".into());
            } else {
                detail.push(match types.get(input.as_str()) {
                    Some(ty) => format!("input: {input} {ty}"),
                    None => format!("input: {input}"),
                });
            }
        }
        for output in &node.outputs {
            detail.push(match types.get(output.as_str()) {
                Some(ty) => format!("output: {output} {ty}"),
                None => format!("output: {output}"),
            });
        }
        for attribute in &node.attributes {
            if attribute.value.is_empty() {
                detail.push(format!("attr: {}", attribute.name));
            } else {
                detail.push(format!("attr: {} = {}", attribute.name, attribute.value));
            }
        }
        for weight in &weights {
            detail.push(format!("weight: {weight}"));
        }

        let idx = nodes.len();
        for output in &node.outputs {
            if !output.is_empty() {
                producer.entry(output.clone()).or_insert(idx);
            }
        }
        nodes.push(Draft {
            id: format!("op-{ordinal}"),
            kind: Kind::Operator,
            op_type: node.op_type.clone(),
            title,
            lines,
            detail,
        });
    }

    let mut edges: HashSet<(usize, usize)> = HashSet::new();
    for (ordinal, node) in graph.nodes.iter().enumerate() {
        let to = op_start + ordinal;
        for input in &node.inputs {
            if input.is_empty() {
                continue;
            }
            if !show_weights && initializers.contains_key(input.as_str()) {
                continue;
            }
            let from = match producer.get(input.as_str()) {
                Some(index) => *index,
                None => {
                    let idx = nodes.len();
                    producer.insert(input.clone(), idx);
                    nodes.push(Draft::value(Kind::Unresolved, input, String::new()));
                    idx
                }
            };
            if from != to {
                edges.insert((from, to));
            }
        }
    }

    for output in &graph.outputs {
        if output.name.is_empty() {
            continue;
        }
        let to = nodes.len();
        nodes.push(Draft::value(
            Kind::Output,
            &output.name,
            output.type_text.clone(),
        ));
        if let Some(from) = producer.get(output.name.as_str()) {
            if *from != to {
                edges.insert((*from, to));
            }
        }
    }

    place(nodes, edges)
}

fn type_index(graph: &crate::onnx::Graph) -> HashMap<String, String> {
    let mut types = HashMap::new();
    let remember = |types: &mut HashMap<String, String>, info: &ValueInfo| {
        if !info.name.is_empty() && !info.type_text.is_empty() {
            types.insert(info.name.clone(), info.type_text.clone());
        }
    };
    for info in graph
        .inputs
        .iter()
        .chain(&graph.outputs)
        .chain(&graph.value_info)
    {
        remember(&mut types, info);
    }
    for tensor in &graph.initializers {
        if !tensor.name.is_empty() {
            types.insert(tensor.name.clone(), tensor.type_text.clone());
        }
    }
    types
}

fn type_of(types: &HashMap<String, String>, info: &ValueInfo) -> String {
    if info.type_text.is_empty() {
        types.get(&info.name).cloned().unwrap_or_default()
    } else {
        info.type_text.clone()
    }
}

struct Draft {
    id: String,
    kind: Kind,
    op_type: String,
    title: String,
    lines: Vec<String>,
    detail: Vec<String>,
}

impl Draft {
    fn value(kind: Kind, name: &str, type_text: String) -> Self {
        let title = match kind {
            Kind::Input => "input",
            Kind::Initializer => "initializer",
            Kind::Output => "output",
            Kind::Unresolved => "unresolved",
            Kind::Operator => "op",
        };
        let mut lines = vec![name.to_string()];
        let mut detail = vec![format!("name: {name}")];
        if !type_text.is_empty() {
            lines.push(type_text.clone());
            detail.push(format!("type: {type_text}"));
        }
        Self {
            id: format!("{title}:{name}"),
            kind,
            op_type: String::new(),
            title: title.into(),
            lines,
            detail,
        }
    }
}

fn place(drafts: Vec<Draft>, edges: HashSet<(usize, usize)>) -> Scene {
    let n = drafts.len();
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (from, to) in &edges {
        succs[*from].push(*to);
        preds[*to].push(*from);
    }

    let mut indegree: Vec<usize> = preds.iter().map(|items| items.len()).collect();
    let mut layer = vec![0usize; n];
    let mut queue: VecDeque<usize> = (0..n).filter(|idx| indegree[*idx] == 0).collect();
    let mut placed = vec![false; n];
    while let Some(node) = queue.pop_front() {
        placed[node] = true;
        for succ in &succs[node] {
            layer[*succ] = layer[*succ].max(layer[node] + 1);
            indegree[*succ] -= 1;
            if indegree[*succ] == 0 {
                queue.push_back(*succ);
            }
        }
    }
    let cycle_layer = layer.iter().copied().max().unwrap_or(0) + 1;
    for (idx, done) in placed.iter().enumerate() {
        if !done {
            layer[idx] = cycle_layer;
        }
    }
    let operators: Vec<bool> = drafts
        .iter()
        .map(|draft| draft.kind == Kind::Operator)
        .collect();
    pull_floating_ops(&mut layer, &preds, &succs, &operators);

    let layer_count = layer
        .iter()
        .copied()
        .max()
        .map(|value| value + 1)
        .unwrap_or(0);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); layer_count];
    for (idx, level) in layer.iter().copied().enumerate() {
        groups[level].push(idx);
    }
    order_groups(&mut groups, &preds, &succs);

    let mut heights = vec![0.0; n];
    for (idx, draft) in drafts.iter().enumerate() {
        heights[idx] = 34.0 + draft.lines.len() as f64 * 16.0;
    }

    let mut positions = vec![(0.0, 0.0); n];
    let mut max_x = 0.0f64;
    let mut y = MARGIN;
    for group in &groups {
        let row_width = group.len() as f64 * NODE_W + group.len().saturating_sub(1) as f64 * GAP_X;
        let mut x = MARGIN;
        let row_height = group.iter().map(|idx| heights[*idx]).fold(0.0, f64::max);
        for idx in group {
            positions[*idx] = (x, y);
            x += NODE_W + GAP_X;
        }
        max_x = max_x.max(row_width + MARGIN);
        y += row_height + GAP_Y;
    }

    let nodes = drafts
        .into_iter()
        .enumerate()
        .map(|(idx, draft)| {
            let (x, y) = positions[idx];
            SceneNode {
                id: draft.id,
                kind: draft.kind,
                op_type: draft.op_type,
                title: draft.title,
                lines: draft.lines,
                detail: draft.detail,
                x,
                y,
                w: NODE_W,
                h: heights[idx],
            }
        })
        .collect();

    Scene {
        nodes,
        edges: edges.into_iter().collect(),
        width: max_x + MARGIN + NODE_W,
        height: y + MARGIN,
    }
}

/// Operators whose visible inputs are only hidden initializers would otherwise
/// sit on the first row, with a long edge down to the node that reads them.
fn pull_floating_ops(
    layer: &mut [usize],
    preds: &[Vec<usize>],
    succs: &[Vec<usize>],
    operators: &[bool],
) {
    for node in 0..layer.len() {
        if !operators[node] || !preds[node].is_empty() || succs[node].is_empty() {
            continue;
        }
        let earliest = succs[node].iter().map(|succ| layer[*succ]).min().unwrap();
        if earliest > 0 && layer[node] + 1 < earliest {
            layer[node] = earliest - 1;
        }
    }
}

fn order_groups(groups: &mut [Vec<usize>], preds: &[Vec<usize>], succs: &[Vec<usize>]) {
    fn barycenter(
        group_index: &HashMap<usize, usize>,
        neighbors: &[usize],
        fallback: usize,
    ) -> f64 {
        let found: Vec<usize> = neighbors
            .iter()
            .filter_map(|node| group_index.get(node).copied())
            .collect();
        if found.is_empty() {
            fallback as f64
        } else {
            found.iter().sum::<usize>() as f64 / found.len() as f64
        }
    }

    for _ in 0..4 {
        let mut index = HashMap::new();
        for group in groups.iter() {
            for (pos, node) in group.iter().enumerate() {
                index.insert(*node, pos);
            }
        }
        for level in 1..groups.len() {
            let current = index.clone();
            groups[level].sort_by(|a, b| {
                let left = barycenter(&current, &preds[*a], current[a]);
                let right = barycenter(&current, &preds[*b], current[b]);
                left.total_cmp(&right).then(a.cmp(b))
            });
        }
        let mut index = HashMap::new();
        for group in groups.iter() {
            for (pos, node) in group.iter().enumerate() {
                index.insert(*node, pos);
            }
        }
        for level in (0..groups.len().saturating_sub(1)).rev() {
            let current = index.clone();
            groups[level].sort_by(|a, b| {
                let left = barycenter(&current, &succs[*a], current[a]);
                let right = barycenter(&current, &succs[*b], current[b]);
                left.total_cmp(&right).then(a.cmp(b))
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::{parse_model, tiny_model_bytes};

    #[test]
    fn relu_sits_above_add_and_weights_stay_off_canvas() {
        let model = parse_model(&tiny_model_bytes()).unwrap();
        let scene = layout(&model, false);
        let relu = scene
            .nodes
            .iter()
            .find(|node| node.op_type == "Relu")
            .unwrap();
        let add = scene
            .nodes
            .iter()
            .find(|node| node.op_type == "Add")
            .unwrap();
        assert!(relu.y < add.y);
        assert!(scene
            .nodes
            .iter()
            .all(|node| node.kind != Kind::Initializer));
        assert!(scene.nodes.iter().any(|node| node.kind == Kind::Input));
        assert!(scene.nodes.iter().any(|node| node.kind == Kind::Output));
    }

    #[test]
    fn weight_only_op_sits_beside_its_consumer() {
        let model = parse_model(&reshape_into_add_bytes()).unwrap();
        let scene = layout(&model, false);
        let relu = scene
            .nodes
            .iter()
            .find(|node| node.op_type == "Relu")
            .unwrap();
        let reshape = scene
            .nodes
            .iter()
            .find(|node| node.op_type == "Reshape")
            .unwrap();
        let add = scene
            .nodes
            .iter()
            .find(|node| node.op_type == "Add")
            .unwrap();
        assert_eq!(reshape.y, relu.y);
        assert!(reshape.y < add.y);
    }

    #[test]
    fn cycle_gets_one_extra_layer() {
        let bytes = cycle_bytes();
        let model = parse_model(&bytes).unwrap();
        let scene = layout(&model, false);
        let ys: Vec<f64> = scene
            .nodes
            .iter()
            .filter(|node| node.kind == Kind::Operator)
            .map(|node| node.y)
            .collect();
        assert_eq!(ys.len(), 2);
        assert_eq!(ys[0], ys[1]);
    }

    fn reshape_into_add_bytes() -> Vec<u8> {
        fn field_bytes(field: u32, bytes: &[u8]) -> Vec<u8> {
            let mut out = vec![((field << 3) | 2) as u8, bytes.len() as u8];
            out.extend(bytes);
            out
        }
        fn field_str(field: u32, text: &str) -> Vec<u8> {
            field_bytes(field, text.as_bytes())
        }
        fn field_varint(field: u32, value: u64) -> Vec<u8> {
            vec![((field << 3) | 0) as u8, value as u8]
        }
        let node = |op: &str, inputs: &[&str], output: &str| {
            let mut body = Vec::new();
            for input in inputs {
                body.extend(field_str(1, input));
            }
            body.extend(field_str(2, output));
            body.extend(field_str(4, op));
            body
        };
        let mut weight = field_varint(2, 7);
        weight.extend(field_str(8, "s"));
        let mut graph = field_bytes(1, &node("Relu", &["x"], "h"));
        graph.extend(field_bytes(1, &node("Reshape", &["s"], "r")));
        graph.extend(field_bytes(1, &node("Add", &["h", "r"], "y")));
        graph.extend(field_bytes(5, &weight));
        graph.extend(field_bytes(11, &field_str(1, "x")));
        graph.extend(field_bytes(12, &field_str(1, "y")));
        field_bytes(7, &graph)
    }

    fn cycle_bytes() -> Vec<u8> {
        fn field_bytes(field: u32, bytes: &[u8]) -> Vec<u8> {
            let mut out = vec![((field << 3) | 2) as u8, bytes.len() as u8];
            out.extend(bytes);
            out
        }
        fn field_str(field: u32, text: &str) -> Vec<u8> {
            field_bytes(field, text.as_bytes())
        }
        let node = |op: &str, input: &str, output: &str| {
            let mut body = field_str(1, input);
            body.extend(field_str(2, output));
            body.extend(field_str(4, op));
            body
        };
        let mut graph = field_bytes(1, &node("A", "b", "a"));
        graph.extend(field_bytes(1, &node("B", "a", "b")));
        field_bytes(7, &graph)
    }
}
