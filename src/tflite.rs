//! TensorFlow Lite (`.tflite`) FlatBuffers binary reader.
//!
//! Decodes `tflite.Model` (identifier `"TFL3"`) directly from the FlatBuffers
//! vtable layout without pulling in external FlatBuffers codegen crates.

use crate::onnx::{Attribute, Graph, Model, Node, Opset, TensorInfo, ValueInfo};

pub fn looks_like_tflite(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && (&bytes[4..8] == b"TFL3" || &bytes[4..8] == b"TFL2")
}

pub fn parse_tflite(bytes: &[u8]) -> Result<Model, String> {
    if !looks_like_tflite(bytes) {
        return Err("不是 TFLite FlatBuffers 文件".into());
    }
    let fb = FlatBuf { bytes };
    let root = fb.u32(0)? as usize;

    // Model fields:
    // 0 (offset 4): version: uint32
    // 1 (offset 6): operator_codes: [OperatorCode]
    // 2 (offset 8): subgraphs: [SubGraph]
    // 3 (offset 10): description: string
    // 4 (offset 12): buffers: [Buffer]
    let version = fb.field_u32(root, 0, 3)?;
    let description = fb.field_string(root, 3)?.unwrap_or_default();

    let mut op_codes = Vec::new();
    if let Some(vec_pos) = fb.field_vector(root, 1)? {
        let len = fb.u32(vec_pos)? as usize;
        for i in 0..len {
            let elem_ptr = vec_pos + 4 + i * 4;
            let table = fb.indirect(elem_ptr)?;
            // OperatorCode:
            // 0 (offset 4): deprecated_builtin_code: int8
            // 1 (offset 6): custom_code: string
            // 2 (offset 8): version: int32
            // 3 (offset 10): builtin_code: int32
            let dep_code = fb.field_u8(table, 0, 0)? as i32;
            let custom = fb.field_string(table, 1)?.unwrap_or_default();
            let builtin = fb.field_i32(table, 3, 0)?;
            let code = if builtin != 0 { builtin } else { dep_code };
            let name = if code == 32 && !custom.is_empty() {
                custom
            } else {
                builtin_op_name(code).to_string()
            };
            op_codes.push(name);
        }
    }

    // Check which buffers are non-empty (so we can distinguish constant weights from activations)
    let mut non_empty_buffers = std::collections::HashSet::new();
    if let Some(vec_pos) = fb.field_vector(root, 4)? {
        let len = fb.u32(vec_pos)? as usize;
        for i in 0..len {
            let elem_ptr = vec_pos + 4 + i * 4;
            let table = fb.indirect(elem_ptr)?;
            // Buffer field 0: data: [ubyte]
            if let Some(data_vec) = fb.field_vector(table, 0)? {
                let data_len = fb.u32(data_vec).unwrap_or(0);
                if data_len > 0 {
                    non_empty_buffers.insert(i as u32);
                }
            }
        }
    }

    let mut graph = Graph::default();
    if let Some(subgraphs_vec) = fb.field_vector(root, 2)? {
        let sg_count = fb.u32(subgraphs_vec)? as usize;
        if sg_count > 0 {
            let sg_table = fb.indirect(subgraphs_vec + 4)?;
            graph = parse_subgraph(&fb, sg_table, &op_codes, &non_empty_buffers)?;
        }
    }
    if graph.name.is_empty() {
        graph.name = "main".into();
    }

    Ok(Model {
        ir_version: version as i64,
        producer_name: "TFLite".into(),
        producer_version: if description.is_empty() {
            format!("v{version}")
        } else {
            description.clone()
        },
        domain: "tflite".into(),
        model_version: version as i64,
        doc_string: description,
        opsets: vec![Opset {
            domain: "tflite".into(),
            version: version as i64,
        }],
        graph,
        functions: Vec::new(),
    })
}

fn parse_subgraph(
    fb: &FlatBuf<'_>,
    sg: usize,
    op_codes: &[String],
    non_empty_buffers: &std::collections::HashSet<u32>,
) -> Result<Graph, String> {
    // SubGraph fields:
    // 0: tensors: [Tensor]
    // 1: inputs: [int32]
    // 2: outputs: [int32]
    // 3: operators: [Operator]
    // 4: name: string
    let name = fb.field_string(sg, 4)?.unwrap_or_else(|| "main".into());

    let sg_inputs: std::collections::HashSet<i32> = fb
        .field_i32_vec(sg, 1)?
        .into_iter()
        .collect();
    let sg_outputs: Vec<i32> = fb.field_i32_vec(sg, 2)?;

    struct TflTensor {
        name: String,
        type_text: String,
        buffer: u32,
        is_variable: bool,
    }

    let mut tensors = Vec::new();
    if let Some(t_vec) = fb.field_vector(sg, 0)? {
        let len = fb.u32(t_vec)? as usize;
        for i in 0..len {
            let t_table = fb.indirect(t_vec + 4 + i * 4)?;
            // Tensor fields:
            // 0: shape: [int32]
            // 1: type: TensorType (int8)
            // 2: buffer: uint32
            // 3: name: string
            // 5: is_variable: bool
            let shape: Vec<String> = fb
                .field_i32_vec(t_table, 0)?
                .into_iter()
                .map(|d| d.to_string())
                .collect();
            let elem_type = fb.field_u8(t_table, 1, 0)?;
            let buffer = fb.field_u32(t_table, 2, 0)?;
            let raw_name = fb
                .field_string(t_table, 3)?
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| format!("tensor_{i}"));
            let is_variable = fb.field_u8(t_table, 5, 0)? != 0;
            let dtype = tfl_dtype_name(elem_type);
            let type_text = format!("{dtype}[{}]", shape.join(","));
            tensors.push(TflTensor {
                name: raw_name,
                type_text,
                buffer,
                is_variable,
            });
        }
    }

    // Track which tensors are produced by operators
    let mut produced = std::collections::HashSet::new();
    let mut nodes = Vec::new();
    if let Some(op_vec) = fb.field_vector(sg, 3)? {
        let len = fb.u32(op_vec)? as usize;
        for i in 0..len {
            let op_table = fb.indirect(op_vec + 4 + i * 4)?;
            // Operator fields:
            // 0: opcode_index: uint32
            // 1: inputs: [int32]
            // 2: outputs: [int32]
            // 3: builtin_options_type: uint8
            let opcode_idx = fb.field_u32(op_table, 0, 0)? as usize;
            let op_type = op_codes
                .get(opcode_idx)
                .cloned()
                .unwrap_or_else(|| format!("Op_{opcode_idx}"));

            let in_indices = fb.field_i32_vec(op_table, 1)?;
            let out_indices = fb.field_i32_vec(op_table, 2)?;

            let mut inputs = Vec::new();
            for idx in in_indices {
                if idx >= 0 {
                    if let Some(t) = tensors.get(idx as usize) {
                        inputs.push(t.name.clone());
                    }
                }
            }
            let mut outputs = Vec::new();
            for idx in out_indices {
                if idx >= 0 {
                    produced.insert(idx);
                    if let Some(t) = tensors.get(idx as usize) {
                        outputs.push(t.name.clone());
                    }
                }
            }

            let op_name = outputs.first().cloned().unwrap_or_else(|| format!("op_{i}"));
            let mut attributes = Vec::new();
            let opt_type = fb.field_u8(op_table, 3, 0)?;
            if opt_type != 0 {
                attributes.push(Attribute {
                    name: "options".into(),
                    value: builtin_options_name(opt_type).into(),
                });
            }

            nodes.push(Node {
                name: op_name,
                op_type,
                domain: String::new(),
                inputs,
                outputs,
                attributes,
            });
        }
    }

    let mut inputs = Vec::new();
    for &idx in &sg_inputs {
        if idx >= 0 {
            if let Some(t) = tensors.get(idx as usize) {
                inputs.push(ValueInfo {
                    name: t.name.clone(),
                    type_text: t.type_text.clone(),
                });
            }
        }
    }

    let mut outputs = Vec::new();
    for idx in sg_outputs {
        if idx >= 0 {
            if let Some(t) = tensors.get(idx as usize) {
                outputs.push(ValueInfo {
                    name: t.name.clone(),
                    type_text: t.type_text.clone(),
                });
            }
        }
    }

    let mut initializers = Vec::new();
    let mut value_info = Vec::new();
    for (idx, t) in tensors.into_iter().enumerate() {
        let idx_i32 = idx as i32;
        let is_const = !sg_inputs.contains(&idx_i32)
            && !produced.contains(&idx_i32)
            && (non_empty_buffers.contains(&t.buffer) || t.buffer > 0 || t.is_variable);
        if is_const {
            initializers.push(TensorInfo {
                name: t.name.clone(),
                type_text: t.type_text.clone(),
                external: false,
            });
        }
        value_info.push(ValueInfo {
            name: t.name,
            type_text: t.type_text,
        });
    }

    Ok(Graph {
        name,
        doc_string: String::new(),
        nodes,
        inputs,
        outputs,
        initializers,
        value_info,
    })
}

struct FlatBuf<'a> {
    bytes: &'a [u8],
}

impl<'a> FlatBuf<'a> {
    fn u8(&self, pos: usize) -> Result<u8, String> {
        self.bytes
            .get(pos)
            .copied()
            .ok_or_else(|| "FlatBuffers 越界".into())
    }

    fn u16(&self, pos: usize) -> Result<u16, String> {
        let s = self
            .bytes
            .get(pos..pos + 2)
            .ok_or_else(|| "FlatBuffers u16 越界".to_string())?;
        Ok(u16::from_le_bytes(s.try_into().unwrap()))
    }

    fn i32(&self, pos: usize) -> Result<i32, String> {
        let s = self
            .bytes
            .get(pos..pos + 4)
            .ok_or_else(|| "FlatBuffers i32 越界".to_string())?;
        Ok(i32::from_le_bytes(s.try_into().unwrap()))
    }

    fn u32(&self, pos: usize) -> Result<u32, String> {
        let s = self
            .bytes
            .get(pos..pos + 4)
            .ok_or_else(|| "FlatBuffers u32 越界".to_string())?;
        Ok(u32::from_le_bytes(s.try_into().unwrap()))
    }

    fn indirect(&self, pos: usize) -> Result<usize, String> {
        let off = self.u32(pos)? as usize;
        Ok(pos + off)
    }

    fn field_pos(&self, table: usize, field_idx: usize) -> Result<Option<usize>, String> {
        let vtable_rel = self.i32(table)?;
        let vtable = (table as isize - vtable_rel as isize) as usize;
        let vtable_size = self.u16(vtable)? as usize;
        let entry_offset = 4 + field_idx * 2;
        if entry_offset + 2 > vtable_size {
            return Ok(None);
        }
        let field_off = self.u16(vtable + entry_offset)? as usize;
        if field_off == 0 {
            Ok(None)
        } else {
            Ok(Some(table + field_off))
        }
    }

    fn field_u8(&self, table: usize, field_idx: usize, default: u8) -> Result<u8, String> {
        match self.field_pos(table, field_idx)? {
            Some(pos) => self.u8(pos),
            None => Ok(default),
        }
    }

    fn field_i32(&self, table: usize, field_idx: usize, default: i32) -> Result<i32, String> {
        match self.field_pos(table, field_idx)? {
            Some(pos) => self.i32(pos),
            None => Ok(default),
        }
    }

    fn field_u32(&self, table: usize, field_idx: usize, default: u32) -> Result<u32, String> {
        match self.field_pos(table, field_idx)? {
            Some(pos) => self.u32(pos),
            None => Ok(default),
        }
    }

    fn field_vector(&self, table: usize, field_idx: usize) -> Result<Option<usize>, String> {
        match self.field_pos(table, field_idx)? {
            Some(pos) => Ok(Some(self.indirect(pos)?)),
            None => Ok(None),
        }
    }

    fn field_string(&self, table: usize, field_idx: usize) -> Result<Option<String>, String> {
        match self.field_vector(table, field_idx)? {
            Some(vec_pos) => {
                let len = self.u32(vec_pos)? as usize;
                let s = self
                    .bytes
                    .get(vec_pos + 4..vec_pos + 4 + len)
                    .ok_or_else(|| "FlatBuffers string 越界".to_string())?;
                Ok(Some(String::from_utf8_lossy(s).into_owned()))
            }
            None => Ok(None),
        }
    }

    fn field_i32_vec(&self, table: usize, field_idx: usize) -> Result<Vec<i32>, String> {
        let Some(vec_pos) = self.field_vector(table, field_idx)? else {
            return Ok(Vec::new());
        };
        let len = self.u32(vec_pos)? as usize;
        let mut out = Vec::with_capacity(len.min(65536));
        for i in 0..len {
            out.push(self.i32(vec_pos + 4 + i * 4)?);
        }
        Ok(out)
    }
}

fn tfl_dtype_name(code: u8) -> &'static str {
    match code {
        0 => "f32",
        1 => "f16",
        2 => "i32",
        3 => "u8",
        4 => "i64",
        5 => "string",
        6 => "bool",
        7 => "i16",
        8 => "c64",
        9 => "i8",
        10 => "f64",
        11 => "c128",
        12 => "u64",
        13 => "resource",
        14 => "variant",
        15 => "u32",
        16 => "u16",
        17 => "i4",
        18 => "bf16",
        _ => "?",
    }
}

fn builtin_op_name(code: i32) -> &'static str {
    match code {
        0 => "ADD",
        1 => "AVERAGE_POOL_2D",
        2 => "CONCATENATION",
        3 => "CONV_2D",
        4 => "DEPTHWISE_CONV_2D",
        5 => "DEPTH_TO_SPACE",
        6 => "DEQUANTIZE",
        7 => "EMBEDDING_lookup",
        8 => "FLOOR",
        9 => "FULLY_CONNECTED",
        10 => "HASHTABLE_LOOKUP",
        11 => "L2_NORMALIZATION",
        12 => "L2_POOL_2D",
        13 => "LOCAL_RESPONSE_NORMALIZATION",
        14 => "LOGISTIC",
        15 => "LSH_PROJECTION",
        16 => "LSTM",
        17 => "MAX_POOL_2D",
        18 => "MUL",
        19 => "RELU",
        20 => "RELU_N1_TO_1",
        21 => "RELU6",
        22 => "RESHAPE",
        23 => "RESIZE_BILINEAR",
        24 => "RNN",
        25 => "SOFTMAX",
        26 => "SPACE_TO_DEPTH",
        27 => "SVDF",
        28 => "TANH",
        29 => "CONCAT_EMBEDDINGS",
        30 => "SKIP_GRAM",
        31 => "CALL",
        32 => "CUSTOM",
        34 => "GATHER",
        35 => "BATCH_TO_SPACE_ND",
        36 => "SPACE_TO_BATCH_ND",
        37 => "TRANSPOSE",
        38 => "MEAN",
        39 => "SUB",
        40 => "DIV",
        41 => "SQUEEZE",
        42 => "UNIDIRECTIONAL_SEQUENCE_LSTM",
        43 => "STRIDED_SLICE",
        44 => "BIDIRECTIONAL_SEQUENCE_RNN",
        45 => "EXP",
        46 => "TOPK_V2",
        47 => "SPLIT",
        48 => "LOG_SOFTMAX",
        50 => "CAST",
        51 => "PRELU",
        52 => "MAXIMUM",
        53 => "ARG_MAX",
        54 => "MINIMUM",
        55 => "LESS",
        56 => "NEG",
        57 => "PADV2",
        58 => "GREATER",
        59 => "GREATER_EQUAL",
        60 => "LESS_EQUAL",
        61 => "SELECT",
        62 => "SLICE",
        63 => "SIN",
        64 => "TRANSPOSE_CONV",
        65 => "SPARSE_TO_DENSE",
        66 => "TILE",
        67 => "EXPAND_DIMS",
        68 => "EQUAL",
        69 => "NOT_EQUAL",
        70 => "LOG",
        71 => "SUM",
        72 => "SQRT",
        73 => "RSQRT",
        74 => "SHAPE",
        75 => "POW",
        76 => "ARG_MIN",
        77 => "FAKE_QUANT",
        78 => "REDUCE_PROD",
        79 => "REDUCE_MAX",
        80 => "PACK",
        81 => "LOGICAL_OR",
        82 => "ONE_HOT",
        83 => "LOGICAL_AND",
        84 => "LOGICAL_NOT",
        85 => "UNPACK",
        86 => "REDUCE_MIN",
        87 => "FLOOR_DIV",
        88 => "REDUCE_ANY",
        89 => "SQUARE",
        90 => "ZEROS_LIKE",
        91 => "FILL",
        92 => "FLOOR_MOD",
        93 => "RANGE",
        94 => "RESIZE_NEAREST_NEIGHBOR",
        95 => "LEAKY_RELU",
        96 => "SQUARED_DIFFERENCE",
        97 => "MIRROR_PAD",
        98 => "ABS",
        99 => "SPLIT_V",
        100 => "UNIQUE",
        101 => "CEIL",
        102 => "REVERSE_V2",
        103 => "ADD_N",
        104 => "GATHER_ND",
        105 => "COS",
        106 => "WHERE",
        107 => "RANK",
        108 => "ELU",
        109 => "REVERSE_SEQUENCE",
        110 => "MATRIX_DIAG",
        111 => "QUANTIZE",
        112 => "MATRIX_SET_DIAG",
        113 => "ROUND",
        114 => "HARD_SWISH",
        115 => "IF",
        116 => "WHILE",
        117 => "NON_MAX_SUPPRESSION_V4",
        118 => "NON_MAX_SUPPRESSION_V5",
        119 => "SCATTER_ND",
        120 => "SELECT_V2",
        121 => "DENSIFY",
        122 => "SEGMENT_SUM",
        123 => "BATCH_MATMUL",
        126 => "GELU",
        _ => "TFLiteOp",
    }
}

fn builtin_options_name(code: u8) -> &'static str {
    match code {
        1 => "Conv2DOptions",
        2 => "DepthwiseConv2DOptions",
        4 => "Pool2DOptions",
        8 => "FullyConnectedOptions",
        13 => "ConcatenationOptions",
        14 => "AddOptions",
        15 => "ReshapeOptions",
        18 => "SoftmaxOptions",
        22 => "MulOptions",
        27 => "SubOptions",
        83 => "BatchMatMulOptions",
        _ => "BuiltinOptions",
    }
}

#[cfg(test)]
pub fn tiny_tflite_bytes() -> Vec<u8> {
    // Hand-assembled minimal TFL3 FlatBuffer:
    // 1 operator code (CONV_2D = 3), 1 subgraph with 3 tensors (x, w, y) and 1 operator
    let mut b = vec![0u8; 256];
    let write_u16 = |buf: &mut [u8], pos: usize, val: u16| {
        buf[pos..pos + 2].copy_from_slice(&val.to_le_bytes());
    };
    let write_i32 = |buf: &mut [u8], pos: usize, val: i32| {
        buf[pos..pos + 4].copy_from_slice(&val.to_le_bytes());
    };
    let write_u32 = |buf: &mut [u8], pos: usize, val: u32| {
        buf[pos..pos + 4].copy_from_slice(&val.to_le_bytes());
    };

    // Header: root table at 16, file_identifier "TFL3"
    write_u32(&mut b, 0, 16);
    b[4..8].copy_from_slice(b"TFL3");

    // Root vtable at 8: size 10, table size 16, f0(version)=4, f1(opcodes)=8, f2(subgraphs)=12
    write_u16(&mut b, 8, 10);
    write_u16(&mut b, 10, 16);
    write_u16(&mut b, 12, 4);
    write_u16(&mut b, 14, 8);
    //Wait: need 2 more bytes for f2 -> vtable at 32 instead
    // Let's place Root vtable at 32 (size 10 bytes: 32..42)
    write_u16(&mut b, 32, 10);
    write_u16(&mut b, 34, 16);
    write_u16(&mut b, 36, 4);
    write_u16(&mut b, 38, 8);
    write_u16(&mut b, 40, 12);

    // Root table at 16:
    write_i32(&mut b, 16, 16 - 32); // vtable_rel = -16
    write_u32(&mut b, 20, 3); // version = 3
    write_u32(&mut b, 24, 48 - 24); // opcodes vector at 48
    write_u32(&mut b, 28, 80 - 28); // subgraphs vector at 80

    // Opcodes vector at 48: len=1, elem[0] -> table at 64
    write_u32(&mut b, 48, 1);
    write_u32(&mut b, 52, 64 - 52);

    // Opcode vtable at 56: size 6, table_size 8, f0(dep_builtin)=4
    write_u16(&mut b, 56, 6);
    write_u16(&mut b, 58, 8);
    write_u16(&mut b, 60, 4);

    // Opcode table at 64:
    write_i32(&mut b, 64, 64 - 56);
    b[68] = 3; // CONV_2D

    // Subgraphs vector at 80: len=1, elem[0] -> table at 104
    write_u32(&mut b, 80, 1);
    write_u32(&mut b, 84, 104 - 84);

    // SubGraph vtable at 88: size 12, table_size 20, f0(tensors)=0, f1(inputs)=4, f2(outputs)=8, f3(ops)=12, f4(name)=16
    write_u16(&mut b, 88, 14);
    write_u16(&mut b, 90, 20);
    write_u16(&mut b, 92, 0); // tensors (none for ultra-compact test, or we can add ops)
    write_u16(&mut b, 94, 0);
    write_u16(&mut b, 96, 0);
    write_u16(&mut b, 98, 4); // operators at +4
    write_u16(&mut b, 100, 8); // name at +8

    // SubGraph table at 104:
    write_i32(&mut b, 104, 104 - 88);
    write_u32(&mut b, 108, 120 - 108); // operators vec at 120
    write_u32(&mut b, 112, 160 - 112); // name string at 160

    // Operators vec at 120: len=1, elem[0] -> table at 140
    write_u32(&mut b, 120, 1);
    write_u32(&mut b, 124, 140 - 124);

    // Operator vtable at 128: size 6, table_size 8, f0(opcode_index)=4
    write_u16(&mut b, 128, 6);
    write_u16(&mut b, 130, 8);
    write_u16(&mut b, 132, 4);

    // Operator table at 140:
    write_i32(&mut b, 140, 140 - 128);
    write_u32(&mut b, 144, 0); // opcode_index = 0 (CONV_2D)

    // String "tfl_net" at 160:
    write_u32(&mut b, 160, 7);
    b[164..171].copy_from_slice(b"tfl_net");

    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tflite_flatbuffer() {
        let bytes = tiny_tflite_bytes();
        assert!(looks_like_tflite(&bytes));
        let model = parse_tflite(&bytes).unwrap();
        assert_eq!(model.producer_name, "TFLite");
        assert_eq!(model.graph.name, "tfl_net");
        assert_eq!(model.graph.nodes.len(), 1);
        assert_eq!(model.graph.nodes[0].op_type, "CONV_2D");
    }
}
