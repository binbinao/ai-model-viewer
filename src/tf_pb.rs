//! TensorFlow `GraphDef` / `SavedModel` (`.pb`) protobuf reader.
//!
//! Supports both raw `GraphDef` files (repeated `NodeDef` at field 1) and
//! `SavedModel` wrappers (`MetaGraphDef` at field 2 -> `GraphDef` at field 1).

use crate::onnx::{Attribute, Function, Graph, Model, Node, Opset, TensorInfo, ValueInfo};
use crate::proto::{each_field, packed_f32, packed_varints, utf8, Field, ProtoError};

pub fn parse_tf_pb(bytes: &[u8]) -> Result<Model, String> {
    if bytes.is_empty() {
        return Err("文件是空的".into());
    }

    // Try SavedModel first (field 1 = saved_model_schema_version, field 2 = meta_graphs)
    let mut saved_model_graph: Option<Vec<u8>> = None;
    let mut sm_version = 0i64;
    let _ = each_field(bytes, |field, value| {
        match (field, value) {
            (1, Field::Varint(v)) => sm_version = v as i64,
            (2, Field::Bytes(meta_graph)) if saved_model_graph.is_none() => {
                // MetaGraphDef field 1 is GraphDef
                let _ = each_field(meta_graph, |mf, mv| {
                    if mf == 1 {
                        if let Field::Bytes(gd) = mv {
                            if looks_like_graph_def(gd) {
                                saved_model_graph = Some(gd.to_vec());
                            }
                        }
                    }
                    Ok(())
                });
            }
            _ => {}
        }
        Ok(())
    });

    if let Some(gd_bytes) = saved_model_graph {
        let mut model = parse_graph_def(&gd_bytes).map_err(|e| e.to_string())?;
        model.producer_name = "TensorFlow SavedModel".into();
        if sm_version > 0 {
            model.ir_version = sm_version;
        }
        return Ok(model);
    }

    if !looks_like_graph_def(bytes) {
        return Err("不是有效的 TensorFlow GraphDef 或 SavedModel".into());
    }

    parse_graph_def(bytes).map_err(|e| e.to_string())
}

pub fn looks_like_graph_def(bytes: &[u8]) -> bool {
    let mut node_count = 0usize;
    let mut valid_ops = 0usize;
    let ok = each_field(bytes, |field, value| {
        if field == 1 {
            if let Field::Bytes(b) = value {
                node_count += 1;
                if node_count <= 8 && is_plausible_nodedef(b) {
                    valid_ops += 1;
                }
            }
        }
        Ok(())
    })
    .is_ok();
    ok && node_count > 0 && valid_ops > 0
}

fn is_plausible_nodedef(buf: &[u8]) -> bool {
    let mut has_name = false;
    let mut has_op = false;
    let ok = each_field(buf, |field, value| {
        match (field, value) {
            (1, Field::Bytes(b)) if !b.is_empty() && b.iter().all(|c| c.is_ascii_graphic()) => {
                has_name = true;
            }
            (2, Field::Bytes(b)) if !b.is_empty() && b.iter().all(|c| c.is_ascii_graphic()) => {
                has_op = true;
            }
            _ => {}
        }
        Ok(())
    })
    .is_ok();
    ok && has_name && has_op
}

fn parse_graph_def(bytes: &[u8]) -> Result<Model, ProtoError> {
    let mut raw_nodes = Vec::new();
    let mut producer = 0i64;
    let mut functions = Vec::new();

    each_field(bytes, |field, value| {
        match (field, value) {
            (1, Field::Bytes(b)) => raw_nodes.push(parse_nodedef(b)?),
            (2, Field::Bytes(b)) => functions.extend(parse_function_library(b)?),
            (4, Field::Bytes(b)) => {
                let _ = each_field(b, |vf, vv| {
                    if vf == 1 {
                        if let Field::Varint(v) = vv {
                            producer = v as i64;
                        }
                    }
                    Ok(())
                });
            }
            _ => {}
        }
        Ok(())
    })?;

    let mut inputs = Vec::new();
    let mut initializers = Vec::new();
    let mut value_info = Vec::new();
    let mut nodes = Vec::new();
    let mut consumed = std::collections::HashSet::new();

    for raw in raw_nodes {
        let type_text = raw.type_text();
        if !type_text.is_empty() {
            value_info.push(ValueInfo {
                name: raw.name.clone(),
                type_text: type_text.clone(),
            });
        }
        for inp in &raw.inputs {
            consumed.insert(inp.clone());
        }

        match raw.op.as_str() {
            "Placeholder" | "PlaceholderWithDefault" => {
                inputs.push(ValueInfo {
                    name: raw.name,
                    type_text,
                });
            }
            "Const" => {
                initializers.push(TensorInfo {
                    name: raw.name,
                    type_text,
                    external: false,
                });
            }
            _ => {
                nodes.push(Node {
                    name: raw.name.clone(),
                    op_type: raw.op,
                    domain: raw.device,
                    inputs: raw.inputs,
                    outputs: vec![raw.name],
                    attributes: raw.attributes,
                });
            }
        }
    }

    // Leaf nodes that are not consumed by any other node become graph outputs
    let mut outputs = Vec::new();
    for node in &nodes {
        for out in &node.outputs {
            if !consumed.contains(out) {
                let ty = value_info
                    .iter()
                    .find(|v| v.name == *out)
                    .map(|v| v.type_text.clone())
                    .unwrap_or_default();
                outputs.push(ValueInfo {
                    name: out.clone(),
                    type_text: ty,
                });
            }
        }
    }

    Ok(Model {
        ir_version: producer.max(1),
        producer_name: "TensorFlow".into(),
        producer_version: if producer > 0 {
            format!("GraphDef v{producer}")
        } else {
            "GraphDef".into()
        },
        domain: "tensorflow".into(),
        model_version: producer,
        doc_string: String::new(),
        opsets: vec![Opset {
            domain: "tf".into(),
            version: producer.max(1),
        }],
        graph: Graph {
            name: "GraphDef".into(),
            doc_string: String::new(),
            nodes,
            inputs,
            outputs,
            initializers,
            value_info,
        },
        functions,
    })
}

struct TfNode {
    name: String,
    op: String,
    inputs: Vec<String>,
    device: String,
    attributes: Vec<Attribute>,
    dtype: Option<&'static str>,
    shape: Option<Vec<String>>,
}

impl TfNode {
    fn type_text(&self) -> String {
        match (self.dtype, &self.shape) {
            (Some(dt), Some(dims)) => format!("{dt}[{}]", dims.join(",")),
            (Some(dt), None) => dt.to_string(),
            (None, Some(dims)) => format!("?[{}]", dims.join(",")),
            (None, None) => String::new(),
        }
    }
}

fn parse_nodedef(buf: &[u8]) -> Result<TfNode, ProtoError> {
    let mut node = TfNode {
        name: String::new(),
        op: String::new(),
        inputs: Vec::new(),
        device: String::new(),
        attributes: Vec::new(),
        dtype: None,
        shape: None,
    };

    each_field(buf, |field, value| {
        match (field, value) {
            (1, Field::Bytes(b)) => node.name = utf8(b),
            (2, Field::Bytes(b)) => node.op = utf8(b),
            (3, Field::Bytes(b)) => {
                let raw = utf8(b);
                let clean = raw
                    .trim_start_matches('^')
                    .split(':')
                    .next()
                    .unwrap_or(&raw)
                    .to_string();
                if !clean.is_empty() {
                    node.inputs.push(clean);
                }
            }
            (4, Field::Bytes(b)) => node.device = utf8(b),
            (5, Field::Bytes(b)) => {
                if let Some((k, v, dt, sh)) = parse_attr_entry(b)? {
                    if dt.is_some() && (k == "dtype" || k == "T" || node.dtype.is_none()) {
                        node.dtype = dt;
                    }
                    if sh.is_some() && (k == "shape" || k == "_output_shapes" || node.shape.is_none()) {
                        node.shape = sh;
                    }
                    if !k.starts_with('_') {
                        node.attributes.push(Attribute { name: k, value: v });
                    }
                }
            }
            _ => {}
        }
        Ok(())
    })?;
    Ok(node)
}

fn parse_attr_entry(
    buf: &[u8],
) -> Result<Option<(String, String, Option<&'static str>, Option<Vec<String>>)>, ProtoError> {
    let mut key = String::new();
    let mut val_bytes: Option<&[u8]> = None;
    each_field(buf, |field, value| {
        match (field, value) {
            (1, Field::Bytes(b)) => key = utf8(b),
            (2, Field::Bytes(b)) => val_bytes = Some(b),
            _ => {}
        }
        Ok(())
    })?;
    let Some(vb) = val_bytes else {
        return Ok(None);
    };
    let mut text = String::new();
    let mut dtype = None;
    let mut shape = None;

    each_field(vb, |field, value| {
        match (field, value) {
            (1, Field::Bytes(b)) => text = format!("\"{}\"", utf8(b)),
            (2, Field::Varint(v)) => text = (v as i64).to_string(),
            (3, Field::Fixed32(v)) => text = format!("{}", f32::from_bits(v)),
            (4, Field::Varint(v)) => text = (v != 0).to_string(),
            (5, Field::Varint(v)) => {
                let dt = tf_dtype_name(v as i32);
                dtype = Some(dt);
                text = dt.to_string();
            }
            (6, Field::Bytes(b)) => {
                let dims = parse_tf_shape(b)?;
                text = format!("[{}]", dims.join(","));
                shape = Some(dims);
            }
            (7, Field::Bytes(b)) => {
                let (dt, dims) = parse_tf_tensor(b)?;
                dtype = dt;
                shape = Some(dims.clone());
                text = format!("{}[{}]", dt.unwrap_or("?"), dims.join(","));
            }
            (8, Field::Bytes(b)) => {
                text = parse_attr_list(b)?;
            }
            _ => {}
        }
        Ok(())
    })?;

    Ok(Some((key, text, dtype, shape)))
}

fn parse_tf_shape(buf: &[u8]) -> Result<Vec<String>, ProtoError> {
    let mut dims = Vec::new();
    each_field(buf, |field, value| {
        if field == 2 {
            if let Field::Bytes(dim_buf) = value {
                let mut size = -1i64;
                each_field(dim_buf, |df, dv| {
                    if df == 1 {
                        if let Field::Varint(v) = dv {
                            size = v as i64;
                        }
                    }
                    Ok(())
                })?;
                if size < 0 {
                    dims.push("?".into());
                } else {
                    dims.push(size.to_string());
                }
            }
        }
        Ok(())
    })?;
    Ok(dims)
}

fn parse_tf_tensor(buf: &[u8]) -> Result<(Option<&'static str>, Vec<String>), ProtoError> {
    let mut dt = None;
    let mut dims = Vec::new();
    each_field(buf, |field, value| {
        match (field, value) {
            (1, Field::Varint(v)) => dt = Some(tf_dtype_name(v as i32)),
            (2, Field::Bytes(b)) => dims = parse_tf_shape(b)?,
            _ => {}
        }
        Ok(())
    })?;
    Ok((dt, dims))
}

fn parse_attr_list(buf: &[u8]) -> Result<String, ProtoError> {
    let mut items = Vec::new();
    each_field(buf, |field, value| {
        match (field, value) {
            (2, Field::Varint(v)) => items.push((v as i64).to_string()),
            (2, Field::Bytes(b)) => {
                for v in packed_varints(b)? {
                    items.push((v as i64).to_string());
                }
            }
            (3, Field::Fixed32(v)) => items.push(format!("{}", f32::from_bits(v))),
            (3, Field::Bytes(b)) => {
                for v in packed_f32(b)? {
                    items.push(format!("{v}"));
                }
            }
            _ => {}
        }
        Ok(())
    })?;
    Ok(format!("[{}]", items.into_iter().take(6).collect::<Vec<_>>().join(", ")))
}

fn parse_function_library(buf: &[u8]) -> Result<Vec<Function>, ProtoError> {
    let mut out = Vec::new();
    each_field(buf, |field, value| {
        if field == 1 {
            if let Field::Bytes(fdef) = value {
                let mut name = String::new();
                let mut node_count = 0usize;
                each_field(fdef, |ff, fv| {
                    match (ff, fv) {
                        (1, Field::Bytes(sig)) => {
                            each_field(sig, |sf, sv| {
                                if sf == 1 {
                                    if let Field::Bytes(n) = sv {
                                        name = utf8(n);
                                    }
                                }
                                Ok(())
                            })?;
                        }
                        (3, Field::Bytes(_)) => node_count += 1,
                        _ => {}
                    }
                    Ok(())
                })?;
                out.push(Function {
                    name,
                    domain: "tf.function".into(),
                    node_count,
                });
            }
        }
        Ok(())
    })?;
    Ok(out)
}

fn tf_dtype_name(code: i32) -> &'static str {
    match code {
        1 => "f32",
        2 => "f64",
        3 => "i32",
        4 => "u8",
        5 => "i16",
        6 => "i8",
        7 => "string",
        8 => "c64",
        9 => "i64",
        10 => "bool",
        14 => "bf16",
        19 => "f16",
        20 => "resource",
        21 => "variant",
        22 => "u32",
        23 => "u64",
        _ => "?",
    }
}

#[cfg(test)]
pub fn tiny_tf_pb_bytes() -> Vec<u8> {
    fn field_bytes(field: u32, bytes: &[u8]) -> Vec<u8> {
        let mut out = vec![((field << 3) | 2) as u8, bytes.len() as u8];
        out.extend(bytes);
        out
    }
    fn field_str(field: u32, s: &str) -> Vec<u8> {
        field_bytes(field, s.as_bytes())
    }
    let nodedef = |name: &str, op: &str, inputs: &[&str]| {
        let mut b = field_str(1, name);
        b.extend(field_str(2, op));
        for inp in inputs {
            b.extend(field_str(3, inp));
        }
        b
    };
    let mut gd = field_bytes(1, &nodedef("x", "Placeholder", &[]));
    gd.extend(field_bytes(1, &nodedef("w", "Const", &[])));
    gd.extend(field_bytes(1, &nodedef("matmul", "MatMul", &["x", "w"])));
    gd.extend(field_bytes(1, &nodedef("y", "Relu", &["matmul"])));
    gd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tensorflow_graphdef() {
        let bytes = tiny_tf_pb_bytes();
        assert!(looks_like_graph_def(&bytes));
        let model = parse_tf_pb(&bytes).unwrap();
        assert_eq!(model.producer_name, "TensorFlow");
        assert_eq!(model.graph.inputs.len(), 1);
        assert_eq!(model.graph.initializers.len(), 1);
        assert_eq!(model.graph.nodes.len(), 2);
        assert_eq!(model.graph.outputs[0].name, "y");
    }
}
