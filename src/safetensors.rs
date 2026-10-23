//! HuggingFace Safetensors (`.safetensors`) parser.
//!
//! Format spec: 8-byte little-endian `u64` header size N, followed by N bytes
//! of UTF-8 JSON mapping tensor names (and optional `__metadata__`) to dtype,
//! shape, and `data_offsets`.

use crate::hierarchy::build_hierarchy_graph;
use crate::json::{parse_json, JsonValue};
use crate::onnx::{Attribute, Model, Opset, TensorInfo};

pub fn looks_like_safetensors(bytes: &[u8]) -> bool {
    if bytes.len() < 10 {
        return false;
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    if header_len == 0 || header_len > 100 * 1024 * 1024 || 8 + header_len > bytes.len() {
        return false;
    }
    let header = &bytes[8..8 + header_len];
    let first = header.iter().copied().find(|b| !b.is_ascii_whitespace());
    first == Some(b'{')
}

pub fn parse_safetensors(bytes: &[u8]) -> Result<Model, String> {
    if bytes.len() < 8 {
        return Err("Safetensors 文件过短".into());
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    if header_len == 0 || 8 + header_len > bytes.len() {
        return Err("Safetensors 头部长度越界".into());
    }
    let header_str = std::str::from_utf8(&bytes[8..8 + header_len])
        .map_err(|_| "Safetensors 头部不是有效 UTF-8".to_string())?;
    let root = parse_json(header_str)?;
    let obj = root
        .as_object()
        .ok_or_else(|| "Safetensors 头部不是 JSON 对象".to_string())?;

    let mut tensors = Vec::new();
    let mut metadata_attrs = Vec::new();
    let mut producer_version = String::new();
    let mut doc_parts = Vec::new();

    for (key, value) in obj {
        if key == "__metadata__" {
            if let Some(meta) = value.as_object() {
                for (mk, mv) in meta {
                    let val_str = mv.as_str().map(str::to_string).unwrap_or_else(|| mv.summary(48));
                    if mk == "format" && producer_version.is_empty() {
                        producer_version = val_str.clone();
                    }
                    doc_parts.push(format!("{mk}={val_str}"));
                    metadata_attrs.push(Attribute {
                        name: format!("meta.{mk}"),
                        value: val_str,
                    });
                }
            }
            continue;
        }

        let entry = value
            .as_object()
            .ok_or_else(|| format!("Safetensors 张量 {key} 不是对象"))?;
        let dtype_raw = entry
            .get("dtype")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| format!("Safetensors 张量 {key} 缺少 dtype"))?;
        let dtype = normalize_dtype(dtype_raw);
        let shape: Vec<String> = entry
            .get("shape")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| format!("Safetensors 张量 {key} 缺少 shape"))?
            .iter()
            .map(|v| {
                v.as_i64()
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "?".into())
            })
            .collect();

        let type_text = format!("{dtype}[{}]", shape.join(","));
        tensors.push(TensorInfo {
            name: key.clone(),
            type_text,
            external: false,
        });
    }

    let graph = build_hierarchy_graph("safetensors".into(), tensors, metadata_attrs);

    Ok(Model {
        ir_version: 1,
        producer_name: "Safetensors".into(),
        producer_version,
        domain: "safetensors".into(),
        model_version: 1,
        doc_string: doc_parts.join(", "),
        opsets: vec![Opset {
            domain: "safetensors".into(),
            version: 1,
        }],
        graph,
        functions: Vec::new(),
    })
}

fn normalize_dtype(raw: &str) -> &str {
    match raw {
        "F64" => "f64",
        "F32" => "f32",
        "F16" => "f16",
        "BF16" => "bf16",
        "I64" => "i64",
        "I32" => "i32",
        "I16" => "i16",
        "I8" => "i8",
        "U64" => "u64",
        "U32" => "u32",
        "U16" => "u16",
        "U8" => "u8",
        "BOOL" => "bool",
        "F8_E4M3" => "f8e4m3fn",
        "F8_E5M2" => "f8e5m2",
        other => other,
    }
}

#[cfg(test)]
pub fn tiny_safetensors_bytes() -> Vec<u8> {
    let header = r#"{"__metadata__":{"format":"pt"},"model.embed_tokens.weight":{"dtype":"F16","shape":[32000,4096],"data_offsets":[0,16]},"model.layers.0.self_attn.q_proj.weight":{"dtype":"F16","shape":[4096,4096],"data_offsets":[16,32]},"model.layers.0.mlp.up_proj.weight":{"dtype":"F16","shape":[11008,4096],"data_offsets":[32,48]},"lm_head.weight":{"dtype":"F16","shape":[32000,4096],"data_offsets":[48,64]}}"#;
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&[0u8; 64]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_safetensors_header_into_hierarchy() {
        let bytes = tiny_safetensors_bytes();
        assert!(looks_like_safetensors(&bytes));
        let model = parse_safetensors(&bytes).unwrap();
        assert_eq!(model.producer_name, "Safetensors");
        assert_eq!(model.graph.initializers.len(), 4);
        assert!(model
            .graph
            .nodes
            .iter()
            .any(|n| n.op_type == "Embedding"));
        assert!(model
            .graph
            .nodes
            .iter()
            .any(|n| n.op_type == "Attention"));
    }
}
