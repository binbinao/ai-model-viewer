//! ONNX `ModelProto` reader.
//!
//! Field numbers are the public schema in `onnx/onnx.proto` (ONNX 1.17.0).
//! Tensor storage (`raw_data`, typed data arrays, external files) is skipped;
//! only names, dtypes, and shapes are kept.

use crate::gguf;
use crate::keras;
use crate::proto::{each_field, packed_f32, packed_varints, utf8, Field, ProtoError};
use crate::pytorch;
use crate::safetensors;
use crate::tf_pb;
use crate::tflite;
use crate::zip_read;
use std::fmt;

#[derive(Debug)]
pub struct Model {
    pub ir_version: i64,
    pub producer_name: String,
    pub producer_version: String,
    pub domain: String,
    pub model_version: i64,
    pub doc_string: String,
    pub opsets: Vec<Opset>,
    pub graph: Graph,
    pub functions: Vec<Function>,
}

#[derive(Debug)]
pub struct Opset {
    pub domain: String,
    pub version: i64,
}

#[derive(Debug)]
pub struct Function {
    pub name: String,
    pub domain: String,
    pub node_count: usize,
}

#[derive(Debug, Default)]
pub struct Graph {
    pub name: String,
    pub doc_string: String,
    pub nodes: Vec<Node>,
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
    pub initializers: Vec<TensorInfo>,
    pub value_info: Vec<ValueInfo>,
}

#[derive(Debug)]
pub struct Node {
    pub name: String,
    pub op_type: String,
    pub domain: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: Vec<Attribute>,
}

#[derive(Debug, Clone)]
pub struct Attribute {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Clone)]
pub struct ValueInfo {
    pub name: String,
    pub type_text: String,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub type_text: String,
    pub external: bool,
}

#[derive(Debug)]
pub enum OnnxError {
    Proto(ProtoError),
    Message(String),
}

impl fmt::Display for OnnxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Proto(err) => write!(f, "{err}"),
            Self::Message(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for OnnxError {}

impl From<ProtoError> for OnnxError {
    fn from(err: ProtoError) -> Self {
        Self::Proto(err)
    }
}

pub fn parse_model(bytes: &[u8]) -> Result<Model, OnnxError> {
    if bytes.is_empty() {
        return Err(OnnxError::Message("文件是空的".into()));
    }
    // 1. Check magic numbers and unambiguous headers first
    if gguf::looks_like_gguf(bytes) {
        return gguf::parse_gguf(bytes).map_err(OnnxError::Message);
    }
    if tflite::looks_like_tflite(bytes) {
        return tflite::parse_tflite(bytes).map_err(OnnxError::Message);
    }
    if keras::looks_like_hdf5(bytes) {
        return keras::parse_keras(bytes).map_err(OnnxError::Message);
    }
    if zip_read::looks_like_zip(bytes) {
        let entries = zip_read::list_zip_entries(bytes).map_err(OnnxError::Message)?;
        if keras::looks_like_keras_zip(&entries) && !pytorch::looks_like_pytorch_zip(&entries) {
            return keras::parse_keras_zip(bytes, &entries).map_err(OnnxError::Message);
        }
        return pytorch::parse_pytorch_zip(bytes, &entries).map_err(OnnxError::Message);
    }
    if pytorch::looks_like_pt_stream(bytes) {
        return pytorch::parse_pytorch(bytes).map_err(OnnxError::Message);
    }
    if safetensors::looks_like_safetensors(bytes) {
        return safetensors::parse_safetensors(bytes).map_err(OnnxError::Message);
    }
    if keras::looks_like_keras_json(bytes) {
        return keras::parse_keras(bytes).map_err(OnnxError::Message);
    }

    if looks_like_text(bytes) {
        return Err(OnnxError::Message(
            "这是文本，不是支持的二进制或 JSON 模型格式".into(),
        ));
    }

    let mut model = Model {
        ir_version: 0,
        producer_name: String::new(),
        producer_version: String::new(),
        domain: String::new(),
        model_version: 0,
        doc_string: String::new(),
        opsets: Vec::new(),
        graph: Graph::default(),
        functions: Vec::new(),
    };
    let mut saw_graph = false;

    let proto_res = each_field(bytes, |field, value| {
        match (field, value) {
            (1, Field::Varint(v)) => model.ir_version = v as i64,
            (2, Field::Bytes(b)) => model.producer_name = utf8(b),
            (3, Field::Bytes(b)) => model.producer_version = utf8(b),
            (4, Field::Bytes(b)) => model.domain = utf8(b),
            (5, Field::Varint(v)) => model.model_version = v as i64,
            (6, Field::Bytes(b)) => model.doc_string = utf8(b),
            (7, Field::Bytes(b)) => {
                model.graph = parse_graph(b)?;
                saw_graph = true;
            }
            (8, Field::Bytes(b)) => model.opsets.push(parse_opset(b)?),
            (25, Field::Bytes(b)) => model.functions.push(parse_function(b)?),
            _ => {}
        }
        Ok(())
    });

    if proto_res.is_ok() && saw_graph {
        return Ok(model);
    }

    // Try TensorFlow GraphDef / SavedModel (.pb)
    if let Ok(tf_model) = tf_pb::parse_tf_pb(bytes) {
        return Ok(tf_model);
    }

    if let Err(err) = proto_res {
        return Err(err.into());
    }
    Err(OnnxError::Message(
        "无法识别模型格式（支持 ONNX、Safetensors、GGUF、TFLite、PyTorch、TensorFlow GraphDef、Keras）".into(),
    ))
}

fn looks_like_text(bytes: &[u8]) -> bool {
    let head = &bytes[..bytes.len().min(64)];
    head.iter()
        .all(|b| b.is_ascii_graphic() || b.is_ascii_whitespace())
        && head.windows(3).any(|w| w == b"ir_" || w == b"gra")
}

fn parse_opset(buf: &[u8]) -> Result<Opset, ProtoError> {
    let mut opset = Opset {
        domain: String::new(),
        version: 0,
    };
    each_field(buf, |field, value| {
        match (field, value) {
            (1, Field::Bytes(b)) => opset.domain = utf8(b),
            (2, Field::Varint(v)) => opset.version = v as i64,
            _ => {}
        }
        Ok(())
    })?;
    Ok(opset)
}

fn parse_function(buf: &[u8]) -> Result<Function, ProtoError> {
    let mut function = Function {
        name: String::new(),
        domain: String::new(),
        node_count: 0,
    };
    each_field(buf, |field, value| {
        match (field, value) {
            (1, Field::Bytes(b)) => function.name = utf8(b),
            (7, Field::Bytes(_)) => function.node_count += 1,
            (10, Field::Bytes(b)) => function.domain = utf8(b),
            _ => {}
        }
        Ok(())
    })?;
    Ok(function)
}

fn parse_graph(buf: &[u8]) -> Result<Graph, ProtoError> {
    let mut graph = Graph::default();
    each_field(buf, |field, value| {
        match (field, value) {
            (1, Field::Bytes(b)) => graph.nodes.push(parse_node(b)?),
            (2, Field::Bytes(b)) => graph.name = utf8(b),
            (5, Field::Bytes(b)) => graph.initializers.push(parse_tensor(b)?),
            (10, Field::Bytes(b)) => graph.doc_string = utf8(b),
            (11, Field::Bytes(b)) => graph.inputs.push(parse_value_info(b)?),
            (12, Field::Bytes(b)) => graph.outputs.push(parse_value_info(b)?),
            (13, Field::Bytes(b)) => graph.value_info.push(parse_value_info(b)?),
            _ => {}
        }
        Ok(())
    })?;
    Ok(graph)
}

fn parse_node(buf: &[u8]) -> Result<Node, ProtoError> {
    let mut node = Node {
        name: String::new(),
        op_type: String::new(),
        domain: String::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        attributes: Vec::new(),
    };
    each_field(buf, |field, value| {
        match (field, value) {
            (1, Field::Bytes(b)) => node.inputs.push(utf8(b)),
            (2, Field::Bytes(b)) => node.outputs.push(utf8(b)),
            (3, Field::Bytes(b)) => node.name = utf8(b),
            (4, Field::Bytes(b)) => node.op_type = utf8(b),
            (5, Field::Bytes(b)) => node.attributes.push(parse_attribute(b)?),
            (7, Field::Bytes(b)) => node.domain = utf8(b),
            _ => {}
        }
        Ok(())
    })?;
    Ok(node)
}

fn parse_attribute(buf: &[u8]) -> Result<Attribute, ProtoError> {
    let mut name = String::new();
    let mut parts = Vec::new();
    each_field(buf, |field, value| {
        match (field, value) {
            (1, Field::Bytes(b)) => name = utf8(b),
            (2, Field::Fixed32(v)) => parts.push(format_f32(f32::from_bits(v))),
            (3, Field::Varint(v)) => parts.push((v as i64).to_string()),
            (4, Field::Bytes(b)) => parts.push(format!("\"{}\"", clip(&utf8(b), 48))),
            (5, Field::Bytes(b)) => parts.push(parse_tensor(b)?.type_text),
            (6, Field::Bytes(_)) => parts.push("<graph>".into()),
            (7, Field::Fixed32(v)) => parts.push(format_f32(f32::from_bits(v))),
            (7, Field::Bytes(b)) => {
                for value in packed_f32(b)? {
                    parts.push(format_f32(value));
                }
            }
            (8, Field::Varint(v)) => parts.push((v as i64).to_string()),
            (8, Field::Bytes(b)) => {
                for value in packed_varints(b)? {
                    parts.push((value as i64).to_string());
                }
            }
            (9, Field::Bytes(b)) => parts.push(format!("\"{}\"", clip(&utf8(b), 32))),
            _ => {}
        }
        Ok(())
    })?;
    let value = join_clipped(&parts, 8, 80);
    Ok(Attribute { name, value })
}

fn parse_value_info(buf: &[u8]) -> Result<ValueInfo, ProtoError> {
    let mut info = ValueInfo {
        name: String::new(),
        type_text: String::new(),
    };
    each_field(buf, |field, value| {
        match (field, value) {
            (1, Field::Bytes(b)) => info.name = utf8(b),
            (2, Field::Bytes(b)) => info.type_text = parse_type(b)?,
            _ => {}
        }
        Ok(())
    })?;
    Ok(info)
}

fn parse_type(buf: &[u8]) -> Result<String, ProtoError> {
    let mut text = String::new();
    each_field(buf, |field, value| {
        if let Field::Bytes(body) = value {
            match field {
                1 => text = parse_tensor_type(body)?,
                4 => text = format!("seq({})", parse_type(body)?),
                5 => text = "map".into(),
                8 => text = format!("sparse {}", parse_tensor_type(body)?),
                9 => text = format!("opt({})", parse_type(body)?),
                _ => {}
            }
        }
        Ok(())
    })?;
    Ok(text)
}

fn parse_tensor_type(buf: &[u8]) -> Result<String, ProtoError> {
    let mut elem = 0i32;
    let mut shape = Vec::new();
    let mut saw_shape = false;
    each_field(buf, |field, value| {
        match (field, value) {
            (1, Field::Varint(v)) => elem = v as i32,
            (2, Field::Bytes(b)) => {
                shape = parse_shape(b)?;
                saw_shape = true;
            }
            _ => {}
        }
        Ok(())
    })?;
    Ok(format_type(
        elem,
        if saw_shape { Some(&shape) } else { None },
    ))
}

fn parse_shape(buf: &[u8]) -> Result<Vec<String>, ProtoError> {
    let mut dims = Vec::new();
    each_field(buf, |field, value| {
        if field == 1 {
            if let Field::Bytes(body) = value {
                dims.push(parse_dim(body)?);
            }
        }
        Ok(())
    })?;
    Ok(dims)
}

fn parse_dim(buf: &[u8]) -> Result<String, ProtoError> {
    let mut text = String::from("?");
    each_field(buf, |field, value| {
        match (field, value) {
            (1, Field::Varint(v)) => text = (v as i64).to_string(),
            (2, Field::Bytes(b)) => text = utf8(b),
            _ => {}
        }
        Ok(())
    })?;
    Ok(text)
}

fn parse_tensor(buf: &[u8]) -> Result<TensorInfo, ProtoError> {
    let mut name = String::new();
    let mut elem = 0i32;
    let mut dims = Vec::new();
    let mut saw_dims = false;
    let mut external = false;
    each_field(buf, |field, value| {
        match (field, value) {
            (1, Field::Varint(v)) => {
                dims.push(v as i64);
                saw_dims = true;
            }
            (1, Field::Bytes(b)) => {
                for value in packed_varints(b)? {
                    dims.push(value as i64);
                }
                saw_dims = true;
            }
            (2, Field::Varint(v)) => elem = v as i32,
            (8, Field::Bytes(b)) => name = utf8(b),
            (13, Field::Bytes(_)) => external = true,
            (14, Field::Varint(v)) if v == 1 => external = true,
            _ => {}
        }
        Ok(())
    })?;
    let dim_text: Vec<String> = dims.iter().map(|dim| dim.to_string()).collect();
    Ok(TensorInfo {
        name,
        type_text: format_type(elem, if saw_dims { Some(&dim_text) } else { None }),
        external,
    })
}

pub fn dtype_name(elem: i32) -> &'static str {
    match elem {
        1 => "f32",
        2 => "u8",
        3 => "i8",
        4 => "u16",
        5 => "i16",
        6 => "i32",
        7 => "i64",
        8 => "string",
        9 => "bool",
        10 => "f16",
        11 => "f64",
        12 => "u32",
        13 => "u64",
        14 => "c64",
        15 => "c128",
        16 => "bf16",
        17 => "f8e4m3fn",
        18 => "f8e4m3fnuz",
        19 => "f8e5m2",
        20 => "f8e5m2fnuz",
        21 => "u4",
        22 => "i4",
        _ => "?",
    }
}

fn format_type(elem: i32, dims: Option<&[String]>) -> String {
    match dims {
        Some(dims) => format!("{}[{}]", dtype_name(elem), dims.join(",")),
        None => dtype_name(elem).to_string(),
    }
}

fn format_f32(value: f32) -> String {
    if value.is_finite() && value.fract() == 0.0 && value.abs() < 1e6 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

fn clip(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars {
        return text.to_string();
    }
    let kept: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    format!("{kept}…")
}

fn join_clipped(parts: &[String], max_items: usize, max_chars: usize) -> String {
    if parts.is_empty() {
        return String::new();
    }
    let shown = parts.len().min(max_items);
    let mut text = parts[..shown].join(", ");
    if parts.len() > shown {
        text.push_str(&format!(" …({} 项)", parts.len()));
    }
    clip(&text, max_chars)
}

/// Minimal ModelProto used by tests: input x -> Relu -> Add(w) -> output y.
#[cfg(test)]
pub fn tiny_model_bytes() -> Vec<u8> {
    fn key(field: u32, wire: u32) -> u8 {
        ((field << 3) | wire) as u8
    }
    fn varint(mut value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if value == 0 {
                break;
            }
        }
        out
    }
    fn field_varint(field: u32, value: u64) -> Vec<u8> {
        let mut out = vec![key(field, 0)];
        out.extend(varint(value));
        out
    }
    fn field_bytes(field: u32, bytes: &[u8]) -> Vec<u8> {
        let mut out = vec![key(field, 2)];
        out.extend(varint(bytes.len() as u64));
        out.extend(bytes);
        out
    }
    fn str_field(field: u32, text: &str) -> Vec<u8> {
        field_bytes(field, text.as_bytes())
    }

    let dim = |value: i64| field_varint(1, value as u64);
    let shape = |dims: &[i64]| {
        let mut body = Vec::new();
        for value in dims {
            body.extend(field_bytes(1, &dim(*value)));
        }
        body
    };
    let tensor_type = |elem: i32, dims: &[i64]| {
        let mut body = field_varint(1, elem as u64);
        body.extend(field_bytes(2, &shape(dims)));
        body
    };
    let value_info = |name: &str, elem: i32, dims: &[i64]| {
        let mut body = str_field(1, name);
        body.extend(field_bytes(2, &field_bytes(1, &tensor_type(elem, dims))));
        body
    };

    let relu = {
        let mut body = str_field(1, "x");
        body.extend(str_field(2, "h"));
        body.extend(str_field(3, "relu1"));
        body.extend(str_field(4, "Relu"));
        body
    };
    let axis = {
        let mut body = str_field(1, "axis");
        body.extend(field_varint(3, 1));
        body
    };
    let add = {
        let mut body = str_field(1, "h");
        body.extend(str_field(1, "w"));
        body.extend(str_field(2, "y"));
        body.extend(str_field(3, "add1"));
        body.extend(str_field(4, "Add"));
        body.extend(field_bytes(5, &axis));
        body
    };
    let weight = {
        let mut body = field_bytes(1, &varint(4));
        body.extend(field_varint(2, 1));
        body.extend(str_field(8, "w"));
        // raw_data is present and must be skipped
        body.extend(field_bytes(
            9,
            &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ));
        body
    };
    let mut graph = str_field(2, "tiny");
    graph.extend(field_bytes(1, &relu));
    graph.extend(field_bytes(1, &add));
    graph.extend(field_bytes(5, &weight));
    graph.extend(field_bytes(11, &value_info("x", 1, &[1, 4])));
    graph.extend(field_bytes(11, &value_info("w", 1, &[4])));
    graph.extend(field_bytes(12, &value_info("y", 1, &[1, 4])));

    let mut model = field_varint(1, 9);
    model.extend(str_field(2, "ai-model-viewer-test"));
    model.extend(field_bytes(7, &graph));
    let mut opset = str_field(1, "");
    opset.extend(field_varint(2, 18));
    model.extend(field_bytes(8, &opset));
    model
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tiny_model_and_skips_weight_bytes() {
        let model = parse_model(&tiny_model_bytes()).unwrap();
        assert_eq!(model.ir_version, 9);
        assert_eq!(model.producer_name, "ai-model-viewer-test");
        assert_eq!(model.opsets[0].version, 18);
        assert_eq!(model.graph.name, "tiny");
        assert_eq!(model.graph.nodes.len(), 2);
        assert_eq!(model.graph.nodes[0].op_type, "Relu");
        assert_eq!(model.graph.nodes[0].inputs, vec!["x"]);
        assert_eq!(model.graph.nodes[1].op_type, "Add");
        assert_eq!(model.graph.nodes[1].attributes[0].name, "axis");
        assert_eq!(model.graph.nodes[1].attributes[0].value, "1");
        assert_eq!(model.graph.initializers[0].name, "w");
        assert_eq!(model.graph.initializers[0].type_text, "f32[4]");
        assert_eq!(model.graph.inputs[0].type_text, "f32[1,4]");
        assert_eq!(model.graph.outputs[0].name, "y");
    }

    #[test]
    fn parses_all_supported_formats_via_unified_entrypoint() {
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("onnx", tiny_model_bytes()),
            ("safetensors", crate::safetensors::tiny_safetensors_bytes()),
            ("gguf", crate::gguf::tiny_gguf_bytes()),
            ("tflite", crate::tflite::tiny_tflite_bytes()),
            ("tf_pb", crate::tf_pb::tiny_tf_pb_bytes()),
            ("pytorch", crate::pytorch::tiny_pytorch_zip_bytes()),
            ("keras", crate::keras::tiny_keras_zip_bytes()),
            ("hdf5", crate::keras::tiny_hdf5_bytes()),
        ];

        for (label, bytes) in cases {
            let model = parse_model(&bytes).unwrap_or_else(|e| panic!("{label} failed: {e}"));
            assert!(!model.graph.nodes.is_empty(), "{label} produced no nodes");
        }
    }

    #[test]
    fn rejects_text() {
        let err = parse_model(b"ir_version: 8\ngraph {\n").unwrap_err();
        assert!(err.to_string().contains("文本"));
    }
}
