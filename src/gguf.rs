//! GGUF (`.gguf`, GGML Universal File) binary format reader (v1, v2, v3).
//!
//! Reads the magic `"GGUF"`, version, tensor count, metadata KV pairs, and
//! tensor info directory without reading the aligned weight payload.

use crate::hierarchy::build_hierarchy_graph;
use crate::onnx::{Attribute, Model, Opset, TensorInfo};

pub fn looks_like_gguf(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && &bytes[..4] == b"GGUF"
}

pub fn parse_gguf(bytes: &[u8]) -> Result<Model, String> {
    if !looks_like_gguf(bytes) {
        return Err("不是 GGUF 文件".into());
    }
    let mut cur = Cursor { bytes, pos: 4 };
    let version = cur.u32()?;
    if !(1..=3).contains(&version) {
        return Err(format!("不支持的 GGUF 版本 {version}"));
    }

    let tensor_count = if version == 1 {
        cur.u32()? as usize
    } else {
        cur.u64()? as usize
    };
    let kv_count = if version == 1 {
        cur.u32()? as usize
    } else {
        cur.u64()? as usize
    };

    let mut arch = String::new();
    let mut model_name = String::new();
    let mut meta_attrs = Vec::new();

    for _ in 0..kv_count {
        let key = cur.string(version)?;
        let val_type = cur.u32()?;
        let val_summary = cur.read_kv_value(val_type, version)?;
        if key == "general.architecture" {
            arch = val_summary.trim_matches('"').to_string();
        } else if key == "general.name" {
            model_name = val_summary.trim_matches('"').to_string();
        }
        if !key.starts_with("tokenizer.ggml.tokens")
            && !key.starts_with("tokenizer.ggml.scores")
            && !key.starts_with("tokenizer.ggml.token_type")
            && !key.starts_with("tokenizer.ggml.merges")
            && meta_attrs.len() < 16
        {
            meta_attrs.push(Attribute {
                name: key,
                value: val_summary,
            });
        }
    }

    let mut tensors = Vec::with_capacity(tensor_count.min(65536));
    for _ in 0..tensor_count {
        let name = cur.string(version)?;
        let n_dims = cur.u32()? as usize;
        if n_dims > 16 {
            return Err(format!("GGUF 张量维度异常: {n_dims}"));
        }
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            let d = if version == 1 {
                cur.u32()? as u64
            } else {
                cur.u64()?
            };
            dims.push(d.to_string());
        }
        let ggml_type = cur.u32()?;
        let _offset = cur.u64()?;
        let dtype = ggml_type_name(ggml_type);
        let type_text = format!("{dtype}[{}]", dims.join(","));
        tensors.push(TensorInfo {
            name,
            type_text,
            external: false,
        });
    }

    let graph_name = if !model_name.is_empty() {
        model_name
    } else if !arch.is_empty() {
        arch.clone()
    } else {
        "gguf".into()
    };

    let graph = build_hierarchy_graph(graph_name, tensors, meta_attrs);

    Ok(Model {
        ir_version: version as i64,
        producer_name: "GGUF".into(),
        producer_version: format!("v{version}"),
        domain: if arch.is_empty() {
            "ggml".into()
        } else {
            arch.clone()
        },
        model_version: version as i64,
        doc_string: String::new(),
        opsets: vec![Opset {
            domain: if arch.is_empty() {
                "gguf".into()
            } else {
                format!("gguf.{arch}")
            },
            version: version as i64,
        }],
        graph,
        functions: Vec::new(),
    })
}

fn ggml_type_name(code: u32) -> &'static str {
    match code {
        0 => "f32",
        1 => "f16",
        2 => "q4_0",
        3 => "q4_1",
        6 => "q5_0",
        7 => "q5_1",
        8 => "q8_0",
        9 => "q8_1",
        10 => "q2_k",
        11 => "q3_k",
        12 => "q4_k",
        13 => "q5_k",
        14 => "q6_k",
        15 => "q8_k",
        16 => "iq2_xxs",
        17 => "iq2_xs",
        18 => "iq3_xxs",
        19 => "iq1_s",
        20 => "iq4_nl",
        21 => "iq3_s",
        22 => "iq2_s",
        23 => "iq4_xs",
        24 => "i8",
        25 => "i16",
        26 => "i32",
        27 => "i64",
        28 => "f64",
        29 => "iq1_m",
        30 => "bf16",
        _ => "ggml",
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.bytes.len() {
            return Err("GGUF 文件被截断".into());
        }
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn i8(&mut self) -> Result<i8, String> {
        Ok(self.u8()? as i8)
    }

    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn i16(&mut self) -> Result<i16, String> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn i64(&mut self) -> Result<i64, String> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self, version: u32) -> Result<String, String> {
        let len = if version == 1 {
            self.u32()? as usize
        } else {
            self.u64()? as usize
        };
        let raw = self.take(len)?;
        Ok(String::from_utf8_lossy(raw).into_owned())
    }

    fn read_kv_value(&mut self, val_type: u32, version: u32) -> Result<String, String> {
        match val_type {
            0 => Ok(self.u8()?.to_string()),
            1 => Ok(self.i8()?.to_string()),
            2 => Ok(self.u16()?.to_string()),
            3 => Ok(self.i16()?.to_string()),
            4 => Ok(self.u32()?.to_string()),
            5 => Ok(self.i32()?.to_string()),
            6 => Ok(format!("{}", self.f32()?)),
            7 => Ok((self.u8()? != 0).to_string()),
            8 => {
                let s = self.string(version)?;
                if s.chars().count() > 48 {
                    let kept: String = s.chars().take(47).collect();
                    Ok(format!("\"{kept}…\""))
                } else {
                    Ok(format!("\"{s}\""))
                }
            }
            9 => {
                let elem_type = self.u32()?;
                let len = if version == 1 {
                    self.u32()? as usize
                } else {
                    self.u64()? as usize
                };
                let mut preview = Vec::new();
                for i in 0..len {
                    let item = self.read_kv_value(elem_type, version)?;
                    if i < 4 {
                        preview.push(item);
                    }
                }
                if len > 4 {
                    Ok(format!("[{}, …({len} 项)]", preview.join(", ")))
                } else {
                    Ok(format!("[{}]", preview.join(", ")))
                }
            }
            10 => Ok(self.u64()?.to_string()),
            11 => Ok(self.i64()?.to_string()),
            12 => Ok(format!("{}", self.f64()?)),
            other => Err(format!("未知的 GGUF KV 类型 {other}")),
        }
    }
}

#[cfg(test)]
pub fn tiny_gguf_bytes() -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes()); // v3
    b.extend_from_slice(&3u64.to_le_bytes()); // 3 tensors
    b.extend_from_slice(&2u64.to_le_bytes()); // 2 KV pairs

    let push_str = |buf: &mut Vec<u8>, s: &str| {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    };

    // KV 1: general.architecture = "llama"
    push_str(&mut b, "general.architecture");
    b.extend_from_slice(&8u32.to_le_bytes());
    push_str(&mut b, "llama");

    // KV 2: general.name = "Tiny-Llama"
    push_str(&mut b, "general.name");
    b.extend_from_slice(&8u32.to_le_bytes());
    push_str(&mut b, "Tiny-Llama");

    let push_tensor = |buf: &mut Vec<u8>, name: &str, dims: &[u64], ggml_type: u32, offset: u64| {
        push_str(buf, name);
        buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&ggml_type.to_le_bytes());
        buf.extend_from_slice(&offset.to_le_bytes());
    };

    push_tensor(&mut b, "token_embd.weight", &[4096, 32000], 1, 0);
    push_tensor(&mut b, "blk.0.attn_q.weight", &[4096, 4096], 12, 64);
    push_tensor(&mut b, "output.weight", &[4096, 32000], 14, 128);
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gguf_v3_header() {
        let bytes = tiny_gguf_bytes();
        assert!(looks_like_gguf(&bytes));
        let model = parse_gguf(&bytes).unwrap();
        assert_eq!(model.producer_name, "GGUF");
        assert_eq!(model.graph.name, "Tiny-Llama");
        assert_eq!(model.graph.initializers.len(), 3);
        assert_eq!(model.graph.initializers[1].type_text, "q4_k[4096,4096]");
    }
}
