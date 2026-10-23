//! Protobuf wire decoder for the ONNX messages this viewer reads.
//!
//! Field numbers follow onnx/onnx.proto at the 1.17.0 tag. Unknown fields are
//! skipped so tensor payloads (`raw_data` and the typed data arrays) are not kept.

use std::fmt;

#[derive(Debug)]
pub struct ProtoError {
    message: String,
}

impl ProtoError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ProtoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ProtoError {}

#[derive(Clone, Copy)]
pub enum Field<'a> {
    Varint(u64),
    Fixed64,
    Bytes(&'a [u8]),
    Fixed32(u32),
}

pub fn each_field<'a>(
    mut buf: &'a [u8],
    mut visit: impl FnMut(u32, Field<'a>) -> Result<(), ProtoError>,
) -> Result<(), ProtoError> {
    while !buf.is_empty() {
        let (key, rest) = read_varint(buf)?;
        buf = rest;
        let field = (key >> 3) as u32;
        let wire = (key & 7) as u32;
        if field == 0 {
            return Err(ProtoError::new("protobuf 字段号为 0"));
        }
        let value = match wire {
            0 => {
                let (v, rest) = read_varint(buf)?;
                buf = rest;
                Field::Varint(v)
            }
            1 => {
                let (_, rest) = split_at(buf, 8, "fixed64")?;
                buf = rest;
                Field::Fixed64
            }
            2 => {
                let (len, rest) = read_varint(buf)?;
                let (bytes, rest) = split_at(rest, len as usize, "length-delimited")?;
                buf = rest;
                Field::Bytes(bytes)
            }
            5 => {
                let (bytes, rest) = split_at(buf, 4, "fixed32")?;
                buf = rest;
                Field::Fixed32(u32::from_le_bytes(bytes.try_into().expect("4 bytes")))
            }
            _ => {
                return Err(ProtoError::new(format!(
                    "不支持的 protobuf wire type {wire}"
                )))
            }
        };
        visit(field, value)?;
    }
    Ok(())
}

pub fn utf8(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

pub fn packed_varints(buf: &[u8]) -> Result<Vec<u64>, ProtoError> {
    let mut out = Vec::new();
    let mut rest = buf;
    while !rest.is_empty() {
        let (v, next) = read_varint(rest)?;
        out.push(v);
        rest = next;
    }
    Ok(out)
}

pub fn packed_f32(buf: &[u8]) -> Result<Vec<f32>, ProtoError> {
    if buf.len() % 4 != 0 {
        return Err(ProtoError::new("packed float 长度不是 4 的倍数"));
    }
    Ok(buf
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4 bytes")))
        .collect())
}

fn read_varint(buf: &[u8]) -> Result<(u64, &[u8]), ProtoError> {
    let mut out = 0u64;
    let mut shift = 0;
    for (i, byte) in buf.iter().copied().enumerate() {
        if i == 10 {
            return Err(ProtoError::new("varint 超过 10 字节"));
        }
        let bits = u64::from(byte & 0x7f);
        if shift == 63 && bits > 1 {
            return Err(ProtoError::new("varint 溢出"));
        }
        out |= bits << shift;
        if byte & 0x80 == 0 {
            return Ok((out, &buf[i + 1..]));
        }
        shift += 7;
    }
    Err(ProtoError::new("varint 被截断"))
}

fn split_at<'a>(buf: &'a [u8], len: usize, what: &str) -> Result<(&'a [u8], &'a [u8]), ProtoError> {
    if len > buf.len() {
        return Err(ProtoError::new(format!("{what} 被截断")));
    }
    Ok(buf.split_at(len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_varint_and_skips_unknown_length() {
        // field 1, varint 150; field 9, length-delimited 3 bytes (skipped by visitor)
        let buf = [0x08, 0x96, 0x01, 0x4a, 0x03, b'a', b'b', b'c'];
        let mut seen = Vec::new();
        each_field(&buf, |field, value| {
            if let Field::Varint(v) = value {
                seen.push((field, v));
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, vec![(1, 150)]);
    }

    #[test]
    fn rejects_truncated_length() {
        let buf = [0x0a, 0x05, b'x'];
        let err = each_field(&buf, |_, _| Ok(())).unwrap_err();
        assert!(err.to_string().contains("截断"));
    }
}
