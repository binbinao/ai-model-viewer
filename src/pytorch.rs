//! PyTorch / TorchScript (`.pt`, `.pth`, `.bin`) parser.
//!
//! Supports:
//! 1. ZIP-based PyTorch archives (`archive/data.pkl`, `archive/code/*.py`, `archive/version`)
//!    - TorchScript models: extracts operator graph from `code/*.py` and pairs with weights
//!    - Checkpoint / `state_dict` archives: statically decodes tensor descriptors (names, dtypes, shapes)
//! 2. Legacy binary tensor stream files (`.pt` / `.bin`)

use crate::hierarchy::build_hierarchy_graph;
use crate::onnx::{Attribute, Graph, Model, Node, Opset, TensorInfo, ValueInfo};
use crate::zip_read::{list_zip_entries, looks_like_zip, read_zip_entry, ZipEntryInfo};
use std::collections::HashMap;

pub fn looks_like_pt_stream(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && bytes[0] == 0x80 && (2..=5).contains(&bytes[1])
}

pub fn looks_like_pytorch_zip(entries: &[ZipEntryInfo]) -> bool {
    entries.iter().any(|e| {
        e.name.ends_with("/data.pkl")
            || e.name == "data.pkl"
            || e.name.ends_with("/constants.pkl")
            || e.name.contains("/code/")
    })
}

pub fn parse_pytorch(bytes: &[u8]) -> Result<Model, String> {
    if looks_like_zip(bytes) {
        let entries = list_zip_entries(bytes)?;
        return parse_pytorch_zip(bytes, &entries);
    }
    if looks_like_pt_stream(bytes) {
        let tensors = decode_pt_tensor_stream(bytes)?;
        let graph = build_hierarchy_graph("pytorch".into(), tensors, Vec::new());
        return Ok(Model {
            ir_version: 2,
            producer_name: "PyTorch".into(),
            producer_version: "checkpoint".into(),
            domain: "pytorch".into(),
            model_version: 1,
            doc_string: String::new(),
            opsets: vec![Opset {
                domain: "aten".into(),
                version: 1,
            }],
            graph,
            functions: Vec::new(),
        });
    }
    Err("不是有效的 PyTorch / TorchScript 文件".into())
}

pub fn parse_pytorch_zip(bytes: &[u8], entries: &[ZipEntryInfo]) -> Result<Model, String> {
    let mut version_text = String::from("archive");
    if let Some(ver_entry) = entries
        .iter()
        .find(|e| e.name.ends_with("/version") || e.name == "version")
    {
        if let Ok(raw) = read_zip_entry(bytes, ver_entry) {
            let s = String::from_utf8_lossy(&raw).trim().to_string();
            if !s.is_empty() {
                version_text = format!("v{s}");
            }
        }
    }

    let mut tensors = Vec::new();
    for entry in entries {
        if (entry.name.ends_with("/data.pkl")
            || entry.name == "data.pkl"
            || entry.name.ends_with("/constants.pkl"))
            && entry.uncompressed_size <= 32 * 1024 * 1024
        {
            if let Ok(stream) = read_zip_entry(bytes, entry) {
                if let Ok(found) = decode_pt_tensor_stream(&stream) {
                    tensors.extend(found);
                }
            }
        }
    }

    // Check for TorchScript code files (`*/code/*.py`)
    let mut code_sources = Vec::new();
    for entry in entries {
        if entry.name.contains("/code/")
            && entry.name.ends_with(".py")
            && entry.uncompressed_size <= 4 * 1024 * 1024
        {
            if let Ok(py_bytes) = read_zip_entry(bytes, entry) {
                code_sources.push(String::from_utf8_lossy(&py_bytes).into_owned());
            }
        }
    }

    if !code_sources.is_empty() {
        if let Some(mut graph) = parse_torchscript_code(&code_sources, tensors.clone()) {
            if graph.name.is_empty() {
                graph.name = "TorchScript".into();
            }
            return Ok(Model {
                ir_version: 1,
                producer_name: "TorchScript".into(),
                producer_version: version_text,
                domain: "aten".into(),
                model_version: 1,
                doc_string: String::new(),
                opsets: vec![Opset {
                    domain: "aten".into(),
                    version: 1,
                }],
                graph,
                functions: Vec::new(),
            });
        }
    }

    // Fallback: if no tensors were extracted from stream, list `data/*` blobs
    if tensors.is_empty() {
        for entry in entries {
            if entry.name.contains("/data/") && !entry.name.ends_with('/') {
                tensors.push(TensorInfo {
                    name: entry.name.clone(),
                    type_text: format!("bytes[{}]", entry.uncompressed_size),
                    external: false,
                });
            }
        }
    }

    let graph = build_hierarchy_graph("state_dict".into(), tensors, Vec::new());
    Ok(Model {
        ir_version: 1,
        producer_name: "PyTorch".into(),
        producer_version: version_text,
        domain: "pytorch".into(),
        model_version: 1,
        doc_string: String::new(),
        opsets: vec![Opset {
            domain: "pytorch".into(),
            version: 1,
        }],
        graph,
        functions: Vec::new(),
    })
}

/// Parse TorchScript `def forward(self, ...)` Python IR inside `archive/code/*.py`.
fn parse_torchscript_code(sources: &[String], initializers: Vec<TensorInfo>) -> Option<Graph> {
    let mut nodes = Vec::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    for src in sources {
        let mut in_forward = false;
        for raw_line in src.lines() {
            let line = raw_line.trim();
            if line.starts_with("def forward(") {
                in_forward = true;
                if let Some(open) = line.find('(') {
                    let close = line.rfind(')').unwrap_or(line.len());
                    let args_str = &line[open + 1..close];
                    for part in args_str.split(',') {
                        let name = part
                            .split(':')
                            .next()
                            .unwrap_or("")
                            .trim()
                            .to_string();
                        if !name.is_empty() && name != "self" {
                            inputs.push(ValueInfo {
                                name,
                                type_text: "Tensor".into(),
                            });
                        }
                    }
                }
                continue;
            }
            if !in_forward {
                continue;
            }
            if line.starts_with("def ") || line.starts_with("class ") {
                break;
            }
            if let Some(rest) = line.strip_prefix("return ") {
                for tok in rest.trim_matches(['(', ')']).split(',') {
                    let out_name = tok.trim().to_string();
                    if !out_name.is_empty() {
                        outputs.push(ValueInfo {
                            name: out_name,
                            type_text: "Tensor".into(),
                        });
                    }
                }
                continue;
            }
            if let Some((lhs, rhs)) = line.split_once(" = ") {
                if let Some(call_open) = rhs.find('(') {
                    let callee = rhs[..call_open].trim();
                    let call_close = rhs.rfind(')').unwrap_or(rhs.len());
                    let arg_blob = &rhs[call_open + 1..call_close];

                    let op_type = if let Some(op) = callee.strip_prefix("torch.ops.aten.") {
                        op.to_string()
                    } else if let Some(op) = callee.strip_prefix("torch.") {
                        op.to_string()
                    } else if callee.ends_with(".forward") {
                        callee
                            .trim_end_matches(".forward")
                            .trim_start_matches("self.")
                            .to_string()
                    } else {
                        continue;
                    };

                    let node_outputs: Vec<String> = lhs
                        .trim_matches(['(', ')'])
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty() && !s.contains(':'))
                        .collect();

                    let mut node_inputs = Vec::new();
                    let mut attrs = Vec::new();
                    for raw_arg in arg_blob.split(',') {
                        let a = raw_arg.trim();
                        if a.is_empty() || a == "None" || a == "False" || a == "True" {
                            continue;
                        }
                        if a.starts_with('[')
                            || a.ends_with(']')
                            || a.chars()
                                .all(|c| c.is_ascii_digit() || c == '-' || c == '.')
                        {
                            attrs.push(Attribute {
                                name: "arg".into(),
                                value: a.to_string(),
                            });
                        } else if let Some(attr) = a.strip_prefix("self.") {
                            node_inputs.push(attr.to_string());
                        } else if a.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                            node_inputs.push(a.to_string());
                        }
                    }

                    if !node_outputs.is_empty() {
                        nodes.push(Node {
                            name: node_outputs[0].clone(),
                            op_type,
                            domain: "aten".into(),
                            inputs: node_inputs,
                            outputs: node_outputs,
                            attributes: attrs,
                        });
                    }
                }
            }
        }
    }

    if nodes.is_empty() {
        return None;
    }

    Some(Graph {
        name: "TorchScript".into(),
        doc_string: String::new(),
        nodes,
        inputs,
        outputs,
        initializers,
        value_info: Vec::new(),
    })
}

/// Pure static stack machine to reconstruct PyTorch `state_dict` tensor descriptors.
#[derive(Clone, Debug)]
enum SVal {
    None,
    Mark,
    Int(i64),
    Str(String),
    Tuple(Vec<SVal>),
    List(Vec<SVal>),
    Dict(Vec<(String, SVal)>),
    Sym(String, String),
    Storage { dtype: String },
    Tensor { dtype: String, shape: Vec<i64> },
}

pub fn decode_pt_tensor_stream(bytes: &[u8]) -> Result<Vec<TensorInfo>, String> {
    let mut stack: Vec<SVal> = Vec::new();
    let mut memo: HashMap<u32, SVal> = HashMap::new();
    let mut pos = 0usize;
    let mut next_memo = 0u32;

    while pos < bytes.len() {
        let op = bytes[pos];
        pos += 1;
        match op {
            0x80 => {
                pos += 1;
            }
            0x95 => {
                pos = (pos + 8).min(bytes.len());
            }
            b'.' => break,
            b'(' => stack.push(SVal::Mark),
            b'N' => stack.push(SVal::None),
            0x88 => stack.push(SVal::Int(1)),
            0x89 => stack.push(SVal::Int(0)),
            b'K' => {
                if pos < bytes.len() {
                    stack.push(SVal::Int(bytes[pos] as i64));
                    pos += 1;
                }
            }
            b'M' => {
                if pos + 2 <= bytes.len() {
                    let v = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as i64;
                    stack.push(SVal::Int(v));
                    pos += 2;
                }
            }
            b'J' => {
                if pos + 4 <= bytes.len() {
                    let v = i32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as i64;
                    stack.push(SVal::Int(v));
                    pos += 4;
                }
            }
            0x8a => {
                if pos < bytes.len() {
                    let n = bytes[pos] as usize;
                    pos += 1;
                    let mut v = 0i64;
                    for i in 0..n.min(8) {
                        if pos + i < bytes.len() {
                            v |= (bytes[pos + i] as i64) << (i * 8);
                        }
                    }
                    pos = (pos + n).min(bytes.len());
                    stack.push(SVal::Int(v));
                }
            }
            b'G' => {
                pos = (pos + 8).min(bytes.len());
                stack.push(SVal::None);
            }
            0x8c => {
                if pos < bytes.len() {
                    let len = bytes[pos] as usize;
                    pos += 1;
                    let end = (pos + len).min(bytes.len());
                    let s = String::from_utf8_lossy(&bytes[pos..end]).into_owned();
                    pos = end;
                    stack.push(SVal::Str(s));
                }
            }
            b'X' | 0x8d => {
                let len = if op == b'X' && pos + 4 <= bytes.len() {
                    let l = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
                    pos += 4;
                    l
                } else if op == 0x8d && pos + 8 <= bytes.len() {
                    let l = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize;
                    pos += 8;
                    l
                } else {
                    break;
                };
                let end = (pos + len).min(bytes.len());
                let s = String::from_utf8_lossy(&bytes[pos..end]).into_owned();
                pos = end;
                stack.push(SVal::Str(s));
            }
            b'U' => {
                if pos < bytes.len() {
                    let len = bytes[pos] as usize;
                    pos += 1;
                    let end = (pos + len).min(bytes.len());
                    let s = String::from_utf8_lossy(&bytes[pos..end]).into_owned();
                    pos = end;
                    stack.push(SVal::Str(s));
                }
            }
            b'T' | b'B' => {
                if pos + 4 <= bytes.len() {
                    let len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
                    pos = (pos + 4 + len).min(bytes.len());
                    stack.push(SVal::None);
                }
            }
            b'C' => {
                if pos < bytes.len() {
                    let len = bytes[pos] as usize;
                    pos = (pos + 1 + len).min(bytes.len());
                    stack.push(SVal::None);
                }
            }
            b'c' => {
                let m = read_line(bytes, &mut pos);
                let n = read_line(bytes, &mut pos);
                stack.push(SVal::Sym(m, n));
            }
            0x93 => {
                let n = pop_str(&mut stack);
                let m = pop_str(&mut stack);
                stack.push(SVal::Sym(m, n));
            }
            b')' => stack.push(SVal::Tuple(Vec::new())),
            0x85 => {
                let a = stack.pop().unwrap_or(SVal::None);
                stack.push(SVal::Tuple(vec![a]));
            }
            0x86 => {
                let b = stack.pop().unwrap_or(SVal::None);
                let a = stack.pop().unwrap_or(SVal::None);
                stack.push(SVal::Tuple(vec![a, b]));
            }
            0x87 => {
                let c = stack.pop().unwrap_or(SVal::None);
                let b = stack.pop().unwrap_or(SVal::None);
                let a = stack.pop().unwrap_or(SVal::None);
                stack.push(SVal::Tuple(vec![a, b, c]));
            }
            b't' => {
                let items = pop_to_mark(&mut stack);
                stack.push(SVal::Tuple(items));
            }
            b']' => stack.push(SVal::List(Vec::new())),
            b'l' => {
                let items = pop_to_mark(&mut stack);
                stack.push(SVal::List(items));
            }
            b'}' => stack.push(SVal::Dict(Vec::new())),
            b'd' => {
                let items = pop_to_mark(&mut stack);
                let mut pairs = Vec::new();
                for chunk in items.chunks(2) {
                    if let [SVal::Str(k), v] = chunk {
                        pairs.push((k.clone(), v.clone()));
                    }
                }
                stack.push(SVal::Dict(pairs));
            }
            b'a' => {
                let val = stack.pop().unwrap_or(SVal::None);
                if let Some(SVal::List(list)) = stack.last_mut() {
                    list.push(val);
                }
            }
            b'e' => {
                let items = pop_to_mark(&mut stack);
                if let Some(SVal::List(list)) = stack.last_mut() {
                    list.extend(items);
                }
            }
            b's' => {
                let v = stack.pop().unwrap_or(SVal::None);
                let k = stack.pop().unwrap_or(SVal::None);
                if let (SVal::Str(key), Some(SVal::Dict(dict))) = (k, stack.last_mut()) {
                    dict.push((key, v));
                }
            }
            b'u' => {
                let items = pop_to_mark(&mut stack);
                if let Some(SVal::Dict(dict)) = stack.last_mut() {
                    for chunk in items.chunks(2) {
                        if let [SVal::Str(k), v] = chunk {
                            dict.push((k.clone(), v.clone()));
                        }
                    }
                }
            }
            b'q' => {
                if pos < bytes.len() {
                    let idx = bytes[pos] as u32;
                    pos += 1;
                    if let Some(top) = stack.last().cloned() {
                        memo.insert(idx, top);
                    }
                }
            }
            b'r' => {
                if pos + 4 <= bytes.len() {
                    let idx = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
                    pos += 4;
                    if let Some(top) = stack.last().cloned() {
                        memo.insert(idx, top);
                    }
                }
            }
            0x94 => {
                if let Some(top) = stack.last().cloned() {
                    memo.insert(next_memo, top);
                    next_memo += 1;
                }
            }
            b'h' => {
                if pos < bytes.len() {
                    let idx = bytes[pos] as u32;
                    pos += 1;
                    stack.push(memo.get(&idx).cloned().unwrap_or(SVal::None));
                }
            }
            b'j' => {
                if pos + 4 <= bytes.len() {
                    let idx = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
                    pos += 4;
                    stack.push(memo.get(&idx).cloned().unwrap_or(SVal::None));
                }
            }
            b'Q' => {
                let pid = stack.pop().unwrap_or(SVal::None);
                stack.push(resolve_persid(pid));
            }
            b'R' | 0x81 => {
                let args = stack.pop().unwrap_or(SVal::None);
                let func = stack.pop().unwrap_or(SVal::None);
                stack.push(eval_construct(func, args));
            }
            b'b' => {
                let state = stack.pop().unwrap_or(SVal::None);
                if let Some(top) = stack.last_mut() {
                    if matches!(top, SVal::None | SVal::Sym(_, _)) {
                        *top = state;
                    }
                }
            }
            b'0' => {
                let _ = stack.pop();
            }
            _ => {}
        }
    }

    let mut out = Vec::new();
    for val in &stack {
        collect_tensors("", val, &mut out);
    }
    Ok(out)
}

fn read_line(bytes: &[u8], pos: &mut usize) -> String {
    let start = *pos;
    while *pos < bytes.len() && bytes[*pos] != b'\n' {
        *pos += 1;
    }
    let s = String::from_utf8_lossy(&bytes[start..*pos]).into_owned();
    if *pos < bytes.len() {
        *pos += 1;
    }
    s
}

fn pop_str(stack: &mut Vec<SVal>) -> String {
    match stack.pop() {
        Some(SVal::Str(s)) => s,
        _ => String::new(),
    }
}

fn pop_to_mark(stack: &mut Vec<SVal>) -> Vec<SVal> {
    let mut out = Vec::new();
    while let Some(v) = stack.pop() {
        if matches!(v, SVal::Mark) {
            break;
        }
        out.push(v);
    }
    out.reverse();
    out
}

fn resolve_persid(pid: SVal) -> SVal {
    if let SVal::Tuple(items) = pid {
        for item in &items {
            if let SVal::Sym(_, name) = item {
                if let Some(dt) = storage_dtype(name) {
                    return SVal::Storage { dtype: dt.into() };
                }
            }
        }
    }
    SVal::Storage { dtype: "f32".into() }
}

fn eval_construct(func: SVal, args: SVal) -> SVal {
    if let SVal::Sym(module, name) = &func {
        if name.starts_with("_rebuild_tensor") || name == "_rebuild_parameter" {
            if let SVal::Tuple(items) = args {
                if name == "_rebuild_parameter" {
                    if let Some(first) = items.into_iter().next() {
                        return first;
                    }
                    return SVal::None;
                }
                let dtype = match items.first() {
                    Some(SVal::Storage { dtype }) => dtype.clone(),
                    _ => "f32".into(),
                };
                let mut shape = Vec::new();
                if let Some(SVal::Tuple(dims) | SVal::List(dims)) = items.get(2) {
                    for d in dims {
                        if let SVal::Int(n) = d {
                            shape.push(*n);
                        }
                    }
                }
                return SVal::Tensor { dtype, shape };
            }
        }
        if module == "collections" && name == "OrderedDict" {
            return SVal::Dict(Vec::new());
        }
    }
    args
}

fn storage_dtype(name: &str) -> Option<&'static str> {
    match name {
        "FloatStorage" => Some("f32"),
        "HalfStorage" => Some("f16"),
        "BFloat16Storage" => Some("bf16"),
        "DoubleStorage" => Some("f64"),
        "LongStorage" => Some("i64"),
        "IntStorage" => Some("i32"),
        "ShortStorage" => Some("i16"),
        "CharStorage" => Some("i8"),
        "ByteStorage" | "QUInt8Storage" => Some("u8"),
        "QInt8Storage" => Some("i8"),
        "BoolStorage" => Some("bool"),
        _ => None,
    }
}

fn collect_tensors(prefix: &str, val: &SVal, out: &mut Vec<TensorInfo>) {
    match val {
        SVal::Tensor { dtype, shape } => {
            let name = if prefix.is_empty() {
                format!("tensor_{}", out.len())
            } else {
                prefix.to_string()
            };
            let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            out.push(TensorInfo {
                name,
                type_text: format!("{dtype}[{}]", dims.join(",")),
                external: false,
            });
        }
        SVal::Dict(pairs) => {
            for (k, v) in pairs {
                let next = if prefix.is_empty() || matches!(prefix, "state_dict" | "model") {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                collect_tensors(&next, v, out);
            }
        }
        SVal::Tuple(items) | SVal::List(items) => {
            for item in items {
                collect_tensors(prefix, item, out);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
pub fn tiny_pytorch_zip_bytes() -> Vec<u8> {
    let mut stream = vec![
        0x80, 0x02,
        b'}',
        b'(',
    ];

    let push_tensor = |buf: &mut Vec<u8>, name: &str, dims: &[u8]| {
        buf.push(0x8c);
        buf.push(name.len() as u8);
        buf.extend_from_slice(name.as_bytes());

        buf.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n");
        buf.push(b'(');
        buf.push(b'(');
        buf.extend_from_slice(b"ctorch\nFloatStorage\n");
        buf.push(b't');
        buf.push(b'Q');
        buf.extend_from_slice(&[b'K', 0]);
        buf.push(b'(');
        for &d in dims {
            buf.extend_from_slice(&[b'K', d]);
        }
        buf.push(b't');
        buf.push(b')');
        buf.push(b't');
        buf.push(b'R');
    };

    push_tensor(&mut stream, "conv1.weight", &[16, 3, 3, 3]);
    push_tensor(&mut stream, "fc.weight", &[10, 16]);
    stream.push(b'u');
    stream.push(b'.');

    crate::zip_read::build_stored_zip(&[
        ("archive/version", b"3\n"),
        ("archive/data.pkl", &stream),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_pytorch_zip_state_dict() {
        let bytes = tiny_pytorch_zip_bytes();
        let model = parse_pytorch(&bytes).unwrap();
        assert_eq!(model.producer_name, "PyTorch");
        assert_eq!(model.graph.initializers.len(), 2);
        assert_eq!(model.graph.initializers[0].name, "conv1.weight");
        assert_eq!(model.graph.initializers[0].type_text, "f32[16,3,3,3]");
    }
}
