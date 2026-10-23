//! Keras (`.keras`, `.h5`, `.hdf5`, `.json`) model parser.
//!
//! Supports:
//! 1. Keras v3 `.keras` ZIP archives (`config.json` + `metadata.json` + `model.weights.h5`)
//! 2. Standalone Keras JSON topology (`class_name: "Sequential" | "Functional" | "Model"`)
//! 3. HDF5 (`.h5` / `.hdf5`) files: extracts embedded `model_config` JSON attribute if present,
//!    or falls back to scanning HDF5 symbol-table / object headers for dataset paths.

use crate::hierarchy::build_hierarchy_graph;
use crate::json::{parse_json, JsonValue};
use crate::onnx::{Attribute, Graph, Model, Node, Opset, TensorInfo, ValueInfo};
use crate::zip_read::{list_zip_entries, looks_like_zip, read_zip_entry, ZipEntryInfo};

const HDF5_MAGIC: &[u8; 8] = b"\x89HDF\r\n\x1a\n";

pub fn looks_like_hdf5(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && &bytes[..8] == HDF5_MAGIC
}

pub fn looks_like_keras_zip(entries: &[ZipEntryInfo]) -> bool {
    entries
        .iter()
        .any(|e| e.name == "config.json" || e.name.ends_with("/config.json") || e.name == "model.weights.h5")
}

pub fn looks_like_keras_json(bytes: &[u8]) -> bool {
    let trimmed = trim_ascii_start(bytes);
    if !trimmed.starts_with(b"{") {
        return false;
    }
    let preview_len = trimmed.len().min(4096);
    let preview = String::from_utf8_lossy(&trimmed[..preview_len]);
    preview.contains("\"class_name\"") && preview.contains("\"config\"")
}

pub fn parse_keras(bytes: &[u8]) -> Result<Model, String> {
    if looks_like_zip(bytes) {
        let entries = list_zip_entries(bytes)?;
        return parse_keras_zip(bytes, &entries);
    }
    if looks_like_hdf5(bytes) {
        return parse_hdf5(bytes);
    }
    if looks_like_keras_json(bytes) {
        let text = std::str::from_utf8(bytes).map_err(|_| "Keras JSON 不是有效 UTF-8".to_string())?;
        return parse_keras_config_str(text, "Keras JSON", "");
    }
    Err("不是有效的 Keras (.keras / .h5 / .json) 模型".into())
}

pub fn parse_keras_zip(bytes: &[u8], entries: &[ZipEntryInfo]) -> Result<Model, String> {
    let mut keras_version = String::from("v3");
    if let Some(meta_entry) = entries
        .iter()
        .find(|e| e.name == "metadata.json" || e.name.ends_with("/metadata.json"))
    {
        if let Ok(raw) = read_zip_entry(bytes, meta_entry) {
            if let Ok(s) = std::str::from_utf8(&raw) {
                if let Ok(val) = parse_json(s) {
                    if let Some(kv) = val.get_str("keras_version") {
                        keras_version = kv.to_string();
                    }
                }
            }
        }
    }

    let config_entry = entries
        .iter()
        .find(|e| e.name == "config.json" || e.name.ends_with("/config.json"))
        .ok_or_else(|| "Keras 归档缺少 config.json".to_string())?;
    let config_bytes = read_zip_entry(bytes, config_entry)?;
    let config_str =
        std::str::from_utf8(&config_bytes).map_err(|_| "config.json 不是有效 UTF-8".to_string())?;

    let mut model = parse_keras_config_str(config_str, "Keras", &keras_version)?;

    // Also inspect weights.h5 if present to populate initializers count
    if let Some(w_entry) = entries.iter().find(|e| e.name.ends_with(".h5")) {
        if w_entry.uncompressed_size <= 64 * 1024 * 1024 {
            if let Ok(h5_bytes) = read_zip_entry(bytes, w_entry) {
                let h5_tensors = scan_hdf5_tensors(&h5_bytes);
                if !h5_tensors.is_empty() && model.graph.initializers.is_empty() {
                    model.graph.initializers = h5_tensors;
                }
            }
        }
    }

    Ok(model)
}

fn parse_keras_config_str(
    json_str: &str,
    producer: &str,
    version: &str,
) -> Result<Model, String> {
    let root = parse_json(json_str)?;
    let class_name = root.get_str("class_name").unwrap_or("Model");
    let config = root
        .get("config")
        .ok_or_else(|| "Keras 配置缺少 config 字段".to_string())?;

    let graph_name = config
        .get_str("name")
        .unwrap_or(class_name)
        .to_string();

    let layers = config
        .get("layers")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| "Keras config 缺少 layers 数组".to_string())?;

    let mut nodes = Vec::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut prev_output: Option<String> = None;

    for (idx, layer) in layers.iter().enumerate() {
        let l_class = layer.get_str("class_name").unwrap_or("Layer");
        let l_cfg = layer.get("config");
        let l_name = layer
            .get_str("name")
            .or_else(|| l_cfg.and_then(|c| c.get_str("name")))
            .map(str::to_string)
            .unwrap_or_else(|| format!("layer_{idx}"));

        let shape_text = l_cfg
            .and_then(|c| {
                c.get("batch_shape")
                    .or_else(|| c.get("batch_input_shape"))
                    .or_else(|| c.get("input_shape"))
            })
            .and_then(JsonValue::as_array)
            .map(|arr| {
                let dims: Vec<String> = arr
                    .iter()
                    .map(|v| {
                        v.as_i64()
                            .map(|n| n.to_string())
                            .unwrap_or_else(|| "?".into())
                    })
                    .collect();
                let dt = l_cfg
                    .and_then(|c| c.get_str("dtype"))
                    .unwrap_or("f32");
                format!("{dt}[{}]", dims.join(","))
            })
            .unwrap_or_default();

        if l_class == "InputLayer" {
            inputs.push(ValueInfo {
                name: l_name.clone(),
                type_text: shape_text,
            });
            prev_output = Some(l_name);
            continue;
        }

        // Determine inputs: either Functional `inbound_nodes` or Sequential chain
        let mut node_inputs = Vec::new();
        if let Some(inbound) = layer.get("inbound_nodes").and_then(JsonValue::as_array) {
            collect_inbound_names(inbound, &mut node_inputs);
        }
        if node_inputs.is_empty() {
            if idx == 0 && inputs.is_empty() {
                let in_name = format!("{l_name}_input");
                inputs.push(ValueInfo {
                    name: in_name.clone(),
                    type_text: shape_text.clone(),
                });
                node_inputs.push(in_name);
            } else if let Some(prev) = &prev_output {
                node_inputs.push(prev.clone());
            }
        }

        let mut attributes = Vec::new();
        if let Some(cfg_obj) = l_cfg.and_then(JsonValue::as_object) {
            for (k, v) in cfg_obj {
                if matches!(
                    k.as_str(),
                    "units"
                        | "filters"
                        | "kernel_size"
                        | "strides"
                        | "padding"
                        | "activation"
                        | "rate"
                        | "axis"
                        | "pool_size"
                        | "target_shape"
                        | "return_sequences"
                        | "use_bias"
                ) {
                    attributes.push(Attribute {
                        name: k.clone(),
                        value: v.summary(40),
                    });
                }
            }
        }

        prev_output = Some(l_name.clone());
        nodes.push(Node {
            name: l_name.clone(),
            op_type: l_class.to_string(),
            domain: "keras".into(),
            inputs: node_inputs,
            outputs: vec![l_name],
            attributes,
        });
    }

    // Functional `output_layers` if present, otherwise last node output
    if let Some(out_layers) = config.get("output_layers") {
        let mut out_names = Vec::new();
        collect_output_layer_names(out_layers, &mut out_names);
        for name in out_names {
            outputs.push(ValueInfo {
                name,
                type_text: String::new(),
            });
        }
    }
    if outputs.is_empty() {
        if let Some(last) = prev_output {
            outputs.push(ValueInfo {
                name: last,
                type_text: String::new(),
            });
        }
    }

    Ok(Model {
        ir_version: 3,
        producer_name: producer.to_string(),
        producer_version: version.to_string(),
        domain: "keras".into(),
        model_version: 1,
        doc_string: format!("class_name={class_name}"),
        opsets: vec![Opset {
            domain: "keras".into(),
            version: 3,
        }],
        graph: Graph {
            name: graph_name,
            doc_string: String::new(),
            nodes,
            inputs,
            outputs,
            initializers: Vec::new(),
            value_info: Vec::new(),
        },
        functions: Vec::new(),
    })
}

fn collect_inbound_names(items: &[JsonValue], out: &mut Vec<String>) {
    for item in items {
        match item {
            // Keras v2: `[[["conv1", 0, 0, {}]]]`
            JsonValue::Array(arr) => {
                if let Some(JsonValue::String(name)) = arr.first() {
                    if !out.contains(name) {
                        out.push(name.clone());
                    }
                } else {
                    collect_inbound_names(arr, out);
                }
            }
            // Keras v3: `[{"args": [{"class_name": "__keras_tensor__", "config": {"keras_history": ["conv1", 0, 0]}}]}]`
            JsonValue::Object(map) => {
                if let Some(hist) = map.get("keras_history").and_then(JsonValue::as_array) {
                    if let Some(JsonValue::String(name)) = hist.first() {
                        if !out.contains(name) {
                            out.push(name.clone());
                        }
                    }
                } else {
                    for v in map.values() {
                        collect_inbound_names(std::slice::from_ref(v), out);
                    }
                }
            }
            _ => {}
        }
    }
}

fn collect_output_layer_names(val: &JsonValue, out: &mut Vec<String>) {
    match val {
        JsonValue::Array(arr) => {
            if let Some(JsonValue::String(s)) = arr.first() {
                if !out.contains(s) {
                    out.push(s.clone());
                }
            } else {
                for item in arr {
                    collect_output_layer_names(item, out);
                }
            }
        }
        _ => {}
    }
}

fn parse_hdf5(bytes: &[u8]) -> Result<Model, String> {
    // 1. Check if `model_config` JSON is embedded in the HDF5 file (standard `model.save('m.h5')`)
    if let Some(json_str) = extract_embedded_keras_json(bytes) {
        if let Ok(mut model) = parse_keras_config_str(&json_str, "Keras HDF5", "h5") {
            let tensors = scan_hdf5_tensors(bytes);
            if !tensors.is_empty() {
                model.graph.initializers = tensors;
            }
            return Ok(model);
        }
    }

    // 2. Pure HDF5 weights file (`model.save_weights('w.h5')` or general HDF5 dataset container)
    let tensors = scan_hdf5_tensors(bytes);
    let graph = build_hierarchy_graph("hdf5".into(), tensors, Vec::new());
    Ok(Model {
        ir_version: 1,
        producer_name: "HDF5".into(),
        producer_version: "v1".into(),
        domain: "hdf5".into(),
        model_version: 1,
        doc_string: String::new(),
        opsets: vec![Opset {
            domain: "hdf5".into(),
            version: 1,
        }],
        graph,
        functions: Vec::new(),
    })
}

/// Locate embedded Keras `{"class_name": ..., "config": ...}` JSON inside an HDF5 binary.
fn extract_embedded_keras_json(bytes: &[u8]) -> Option<String> {
    let needle = b"\"class_name\"";
    let mut pos = 0usize;
    while pos + needle.len() <= bytes.len() {
        let rel = bytes[pos..]
            .windows(needle.len())
            .position(|w| w == needle)?;
        let hit = pos + rel;
        // Walk backwards to the opening `{`
        let mut start = hit;
        while start > 0 && hit - start < 64 && bytes[start] != b'{' {
            start -= 1;
        }
        if bytes[start] == b'{' {
            if let Some(end) = matching_brace(&bytes[start..]) {
                if let Ok(s) = std::str::from_utf8(&bytes[start..start + end]) {
                    if s.contains("\"config\"") && s.contains("\"layers\"") {
                        return Some(s.to_string());
                    }
                }
            }
        }
        pos = hit + needle.len();
    }
    None
}

fn matching_brace(bytes: &[u8]) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            0 => return None,
            _ => {}
        }
    }
    None
}

/// Extract HDF5 symbol/dataset paths from the local heaps & object headers.
fn scan_hdf5_tensors(bytes: &[u8]) -> Vec<TensorInfo> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    let scan_limit = bytes.len().min(8 * 1024 * 1024);
    let mut i = 8usize;

    while i < scan_limit {
        if is_ident_start(bytes[i]) {
            let start = i;
            while i < scan_limit && is_path_char(bytes[i]) {
                i += 1;
            }
            if i < scan_limit && bytes[i] == 0 && i - start >= 3 && i - start <= 128 {
                if let Ok(s) = std::str::from_utf8(&bytes[start..i]) {
                    if looks_like_tensor_symbol(s) && seen.insert(s.to_string()) {
                        out.push(TensorInfo {
                            name: s.trim_end_matches(":0").to_string(),
                            type_text: "tensor".into(),
                            external: false,
                        });
                    }
                }
            }
        }
        i += 1;
    }
    out
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_path_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'/' | b':' | b'-')
}

fn looks_like_tensor_symbol(s: &str) -> bool {
    if s.ends_with(":0") || s.contains("/kernel") || s.contains("/bias") || s.contains("/gamma") || s.contains("/beta") {
        return true;
    }
    matches!(
        s,
        "kernel"
            | "bias"
            | "gamma"
            | "beta"
            | "moving_mean"
            | "moving_variance"
            | "embeddings"
            | "weight"
    )
}

fn trim_ascii_start(mut bytes: &[u8]) -> &[u8] {
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        bytes = &bytes[3..];
    }
    while let Some((first, rest)) = bytes.split_first() {
        if first.is_ascii_whitespace() {
            bytes = rest;
        } else {
            break;
        }
    }
    bytes
}

#[cfg(test)]
pub fn tiny_keras_zip_bytes() -> Vec<u8> {
    let config = r#"{
        "class_name": "Sequential",
        "config": {
            "name": "tiny_seq",
            "layers": [
                {
                    "class_name": "InputLayer",
                    "config": {"name": "img_in", "batch_shape": [1, 28, 28, 1], "dtype": "float32"}
                },
                {
                    "class_name": "Conv2D",
                    "config": {"name": "conv2d_1", "filters": 16, "kernel_size": [3, 3], "activation": "relu"}
                },
                {
                    "class_name": "Dense",
                    "config": {"name": "dense_out", "units": 10, "activation": "softmax"}
                }
            ]
        }
    }"#;
    let metadata = r#"{"keras_version": "3.5.0", "date_saved": "2026-10-01"}"#;
    crate::zip_read::build_stored_zip(&[
        ("metadata.json", metadata.as_bytes()),
        ("config.json", config.as_bytes()),
    ])
}

#[cfg(test)]
pub fn tiny_hdf5_bytes() -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(HDF5_MAGIC);
    b.extend_from_slice(&[0u8; 32]);
    let config = r#"{"class_name":"Sequential","config":{"name":"h5_net","layers":[{"class_name":"Dense","config":{"name":"fc1","units":64,"batch_input_shape":[1,128]}}]}}"#;
    b.extend_from_slice(config.as_bytes());
    b.push(0);
    b.extend_from_slice(b"fc1/kernel:0\0fc1/bias:0\0");
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_keras_v3_zip() {
        let bytes = tiny_keras_zip_bytes();
        let model = parse_keras(&bytes).unwrap();
        assert_eq!(model.producer_name, "Keras");
        assert_eq!(model.producer_version, "3.5.0");
        assert_eq!(model.graph.name, "tiny_seq");
        assert_eq!(model.graph.inputs.len(), 1);
        assert_eq!(model.graph.nodes.len(), 2);
        assert_eq!(model.graph.nodes[0].op_type, "Conv2D");
        assert_eq!(model.graph.nodes[1].op_type, "Dense");
    }

    #[test]
    fn parses_hdf5_with_embedded_config() {
        let bytes = tiny_hdf5_bytes();
        assert!(looks_like_hdf5(&bytes));
        let model = parse_keras(&bytes).unwrap();
        assert_eq!(model.producer_name, "Keras HDF5");
        assert_eq!(model.graph.name, "h5_net");
        assert_eq!(model.graph.nodes[0].op_type, "Dense");
        assert_eq!(model.graph.initializers.len(), 2);
    }
}
