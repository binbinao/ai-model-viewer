//! Hierarchy synthesis for pure weight / checkpoint models (Safetensors, GGUF, state_dict, HDF5).
//!
//! Groups flat tensor names like `model.layers.0.self_attn.q_proj.weight` into
//! topological module nodes and connects sequential blocks into a readable graph.

use crate::onnx::{Attribute, Graph, Node, TensorInfo, ValueInfo};

/// Synthesize an operator graph from a list of named weight tensors by grouping
/// them along hierarchical prefixes and linking sequential blocks.
pub fn build_hierarchy_graph(
    graph_name: String,
    tensors: Vec<TensorInfo>,
    extra_metadata: Vec<Attribute>,
) -> Graph {
    if tensors.is_empty() {
        return Graph {
            name: graph_name,
            doc_string: String::new(),
            nodes: vec![Node {
                name: "empty".into(),
                op_type: "Checkpoint".into(),
                domain: String::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                attributes: extra_metadata,
            }],
            inputs: Vec::new(),
            outputs: Vec::new(),
            initializers: Vec::new(),
            value_info: Vec::new(),
        };
    }

    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    let mut group_index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    for (idx, tensor) in tensors.iter().enumerate() {
        let prefix = module_prefix(&tensor.name);
        if let Some(&g_idx) = group_index.get(&prefix) {
            groups[g_idx].1.push(idx);
        } else {
            let g_idx = groups.len();
            group_index.insert(prefix.clone(), g_idx);
            groups.push((prefix, vec![idx]));
        }
    }

    // Sort groups naturally so layers.0, layers.1, ..., layers.10 stay in order
    groups.sort_by(|(a, _), (b, _)| natural_cmp(a, b));

    let mut nodes = Vec::with_capacity(groups.len());
    let mut value_info = Vec::new();
    let mut prev_activation: Option<String> = None;

    for (ordinal, (prefix, member_indices)) in groups.iter().enumerate() {
        let op_type = infer_module_type(prefix, member_indices, &tensors);
        let mut inputs = Vec::new();
        if let Some(prev) = &prev_activation {
            inputs.push(prev.clone());
        }
        for &t_idx in member_indices {
            inputs.push(tensors[t_idx].name.clone());
        }

        let out_name = format!("{prefix}:out");
        value_info.push(ValueInfo {
            name: out_name.clone(),
            type_text: summarize_group_shapes(member_indices, &tensors),
        });
        prev_activation = Some(out_name.clone());

        let mut attributes = Vec::new();
        if ordinal == 0 {
            attributes.extend(extra_metadata.clone());
        }
        attributes.push(Attribute {
            name: "tensors".into(),
            value: member_indices.len().to_string(),
        });

        // Show up to 3 tensor leaves as compact attributes
        for &t_idx in member_indices.iter().take(3) {
            let t = &tensors[t_idx];
            let leaf = t
                .name
                .strip_prefix(prefix.as_str())
                .map(|s| s.trim_start_matches(['.', '/']))
                .filter(|s| !s.is_empty())
                .unwrap_or(t.name.as_str());
            attributes.push(Attribute {
                name: leaf.to_string(),
                value: t.type_text.clone(),
            });
        }

        nodes.push(Node {
            name: prefix.clone(),
            op_type,
            domain: String::new(),
            inputs,
            outputs: vec![out_name],
            attributes,
        });
    }

    let mut outputs = Vec::new();
    if let Some(last_out) = prev_activation {
        let last_type = value_info
            .last()
            .map(|v| v.type_text.clone())
            .unwrap_or_default();
        outputs.push(ValueInfo {
            name: last_out,
            type_text: last_type,
        });
    }

    Graph {
        name: graph_name,
        doc_string: String::new(),
        nodes,
        inputs: Vec::new(),
        outputs,
        initializers: tensors,
        value_info,
    }
}

fn module_prefix(name: &str) -> String {
    let sep = if name.contains('/') { '/' } else { '.' };
    let parts: Vec<&str> = name.split(sep).filter(|s| !s.is_empty()).collect();
    if parts.len() <= 1 {
        return (*parts.first().unwrap_or(&"root")).to_string();
    }

    // Strip common parameter leaf names (`weight`, `bias`, `scale`, `qweight`, etc.)
    let last = *parts.last().unwrap();
    let is_param_leaf = matches!(
        last,
        "weight"
            | "bias"
            | "scale"
            | "zero_point"
            | "qweight"
            | "qzeros"
            | "scales"
            | "g_idx"
            | "running_mean"
            | "running_var"
            | "num_batches_tracked"
            | "kernel"
            | "gamma"
            | "beta"
            | "moving_mean"
            | "moving_variance"
            | "embeddings"
    );

    let cut = if is_param_leaf && parts.len() >= 2 {
        parts.len() - 1
    } else {
        parts.len()
    };

    // Keep up to 4 segments so deep models group cleanly at sub-block level
    // e.g. `model.layers.0.self_attn` or `blk.0.attn_q`
    let max_depth = if parts.iter().any(|p| p.chars().all(|c| c.is_ascii_digit())) {
        4.min(cut)
    } else {
        3.min(cut)
    };

    parts[..max_depth.max(1)].join(&sep.to_string())
}

fn infer_module_type(prefix: &str, indices: &[usize], tensors: &[TensorInfo]) -> String {
    let lower = prefix.to_ascii_lowercase();
    if lower.contains("embed") || lower.contains("wte") || lower.contains("wpe") || lower.contains("token_embd") {
        "Embedding".into()
    } else if lower.contains("self_attn") || lower.contains("attention") || lower.contains("attn") {
        "Attention".into()
    } else if lower.contains("mlp") || lower.contains("ffn") || lower.contains("feed_forward") {
        "MLP".into()
    } else if lower.contains("norm") || lower.contains("ln_") {
        "LayerNorm".into()
    } else if lower.contains("conv") {
        "Conv".into()
    } else if lower.contains("lm_head") || lower.contains("output") || lower.contains("classifier") || lower.contains("fc") || lower.contains("linear") || lower.contains("proj") {
        "Linear".into()
    } else if lower.contains("blk.") || lower.contains("layers.") || lower.contains("block.") || lower.contains("h.") {
        "Block".into()
    } else if indices.len() == 1 {
        let t = &tensors[indices[0]];
        if t.type_text.contains(',') {
            "Parameter".into()
        } else {
            "Vector".into()
        }
    } else {
        "Module".into()
    }
}

fn summarize_group_shapes(indices: &[usize], tensors: &[TensorInfo]) -> String {
    if let Some(&first) = indices.first() {
        tensors[first].type_text.clone()
    } else {
        String::new()
    }
}

fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    // Prioritize embedding at the top and head/norm/output at the bottom
    let rank = |s: &str| -> u8 {
        let l = s.to_ascii_lowercase();
        if l.contains("embed") || l.contains("token_embd") || l.contains("wte") || l.starts_with("conv1") || l.starts_with("stem") {
            0
        } else if l.contains("lm_head")
            || l.eq("output")
            || l.starts_with("output.")
            || l.contains("classifier")
            || l.ends_with("final_layernorm")
            || l.eq("model.norm")
            || l.eq("output_norm")
        {
            2
        } else {
            1
        }
    };

    match rank(a).cmp(&rank(b)) {
        std::cmp::Ordering::Equal => {}
        other => return other,
    }

    let mut ia = a.chars().peekable();
    let mut ib = b.chars().peekable();

    loop {
        match (ia.peek(), ib.peek()) {
            (None, None) => return std::cmp::Ordering::Equal,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (Some(&ca), Some(&cb)) if ca.is_ascii_digit() && cb.is_ascii_digit() => {
                let mut na = 0u64;
                while let Some(&c) = ia.peek() {
                    if c.is_ascii_digit() {
                        na = na.saturating_mul(10).saturating_add((c as u8 - b'0') as u64);
                        ia.next();
                    } else {
                        break;
                    }
                }
                let mut nb = 0u64;
                while let Some(&c) = ib.peek() {
                    if c.is_ascii_digit() {
                        nb = nb.saturating_mul(10).saturating_add((c as u8 - b'0') as u64);
                        ib.next();
                    } else {
                        break;
                    }
                }
                match na.cmp(&nb) {
                    std::cmp::Ordering::Equal => continue,
                    other => return other,
                }
            }
            (Some(&ca), Some(&cb)) => {
                match ca.cmp(&cb) {
                    std::cmp::Ordering::Equal => {
                        ia.next();
                        ib.next();
                    }
                    other => return other,
                }
            }
        }
    }
}
