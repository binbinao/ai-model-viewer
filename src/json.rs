//! Minimal JSON value parser for model metadata and config files.

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Number(n) => Some(*n as i64),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            Self::Array(arr) => Some(arr),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
        match self {
            Self::Object(obj) => Some(obj),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object()?.get(key)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key)?.as_str()
    }

    pub fn summary(&self, max_len: usize) -> String {
        let raw = match self {
            Self::Null => "null".to_string(),
            Self::Bool(b) => b.to_string(),
            Self::Number(n) => {
                if n.is_finite() && n.fract() == 0.0 && n.abs() < 1e15 {
                    format!("{}", *n as i64)
                } else {
                    format!("{n}")
                }
            }
            Self::String(s) => format!("\"{s}\""),
            Self::Array(items) => {
                let parts: Vec<String> = items.iter().take(6).map(|v| v.summary(24)).collect();
                let mut text = format!("[{}]", parts.join(", "));
                if items.len() > 6 {
                    text = format!("[{}, …({} 项)]", parts.join(", "), items.len());
                }
                text
            }
            Self::Object(map) => {
                let parts: Vec<String> = map
                    .iter()
                    .take(4)
                    .map(|(k, v)| format!("{k}: {}", v.summary(20)))
                    .collect();
                format!("{{{}}}", parts.join(", "))
            }
        };
        if raw.chars().count() <= max_len {
            raw
        } else {
            let kept: String = raw.chars().take(max_len.saturating_sub(1)).collect();
            format!("{kept}…")
        }
    }
}

pub fn parse_json(input: &str) -> Result<JsonValue, String> {
    let mut parser = Parser {
        bytes: input.as_bytes(),
        pos: 0,
    };
    parser.skip_ws();
    let value = parser.parse_value()?;
    parser.skip_ws();
    if parser.pos < parser.bytes.len() {
        return Err("JSON 末尾有多余字符".into());
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                // Skip UTF-8 BOM if at start
                0xEF if self.pos == 0
                    && self.bytes.len() >= 3
                    && self.bytes[1] == 0xBB
                    && self.bytes[2] == 0xBF =>
                {
                    self.pos += 3;
                }
                _ => break,
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn next(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    fn parse_value(&mut self) -> Result<JsonValue, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => self.parse_string().map(JsonValue::String),
            Some(b't') => self.parse_literal(b"true", JsonValue::Bool(true)),
            Some(b'f') => self.parse_literal(b"false", JsonValue::Bool(false)),
            Some(b'n') => self.parse_literal(b"null", JsonValue::Null),
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            Some(ch) => Err(format!("JSON 意外字符 '{}'", ch as char)),
            None => Err("JSON 意外结束".into()),
        }
    }

    fn parse_literal(&mut self, expected: &[u8], value: JsonValue) -> Result<JsonValue, String> {
        if self.bytes.get(self.pos..self.pos + expected.len()) == Some(expected) {
            self.pos += expected.len();
            Ok(value)
        } else {
            Err("JSON 字面量无效".into())
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, String> {
        self.pos += 1; // '{'
        self.skip_ws();
        let mut map = BTreeMap::new();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(map));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err("JSON 对象键必须是字符串".into());
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.next() != Some(b':') {
                return Err("JSON 对象缺少 ':'".into());
            }
            let val = self.parse_value()?;
            map.insert(key, val);
            self.skip_ws();
            match self.next() {
                Some(b',') => {
                    self.skip_ws();
                    if self.peek() == Some(b'}') {
                        self.pos += 1;
                        break;
                    }
                }
                Some(b'}') => break,
                _ => return Err("JSON 对象期望 ',' 或 '}'".into()),
            }
        }
        Ok(JsonValue::Object(map))
    }

    fn parse_array(&mut self) -> Result<JsonValue, String> {
        self.pos += 1; // '['
        self.skip_ws();
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            let val = self.parse_value()?;
            items.push(val);
            self.skip_ws();
            match self.next() {
                Some(b',') => {
                    self.skip_ws();
                    if self.peek() == Some(b']') {
                        self.pos += 1;
                        break;
                    }
                }
                Some(b']') => break,
                _ => return Err("JSON 数组期望 ',' 或 ']'".into()),
            }
        }
        Ok(JsonValue::Array(items))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        if self.next() != Some(b'"') {
            return Err("JSON 字符串缺少起始引号".into());
        }
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            self.pos += 1;
            match b {
                b'"' => return Ok(out),
                b'\\' => {
                    let esc = self.next().ok_or_else(|| "JSON 转义被截断".to_string())?;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let code = self.parse_hex4()?;
                            if (0xD800..=0xDBFF).contains(&code) {
                                if self.bytes.get(self.pos..self.pos + 2) == Some(b"\\u") {
                                    self.pos += 2;
                                    let low = self.parse_hex4()?;
                                    if (0xDC00..=0xDFFF).contains(&low) {
                                        let combined = 0x10000
                                            + (((code as u32 - 0xD800) << 10)
                                                | (low as u32 - 0xDC00));
                                        if let Some(ch) = char::from_u32(combined) {
                                            out.push(ch);
                                            continue;
                                        }
                                    }
                                }
                            }
                            if let Some(ch) = char::from_u32(code as u32) {
                                out.push(ch);
                            } else {
                                out.push('\u{FFFD}');
                            }
                        }
                        other => out.push(other as char),
                    }
                }
                other if other < 0x80 => out.push(other as char),
                _ => {
                    // Multi-byte UTF-8 sequence
                    let start = self.pos - 1;
                    let width = if b & 0b1110_0000 == 0b1100_0000 {
                        2
                    } else if b & 0b1111_0000 == 0b1110_0000 {
                        3
                    } else if b & 0b1111_1000 == 0b1111_0000 {
                        4
                    } else {
                        1
                    };
                    let end = (start + width).min(self.bytes.len());
                    self.pos = end;
                    out.push_str(&String::from_utf8_lossy(&self.bytes[start..end]));
                }
            }
        }
        Err("JSON 字符串未闭合".into())
    }

    fn parse_hex4(&mut self) -> Result<u16, String> {
        let slice = self
            .bytes
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| "JSON \\u 转义长度不足".to_string())?;
        self.pos += 4;
        let text = std::str::from_utf8(slice).map_err(|_| "JSON \\u 非法字符".to_string())?;
        u16::from_str_radix(text, 16).map_err(|_| "JSON \\u 非法十六进制".to_string())
    }

    fn parse_number(&mut self) -> Result<JsonValue, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while let Some(b'0'..=b'9') = self.peek() {
            self.pos += 1;
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while let Some(b'0'..=b'9') = self.peek() {
                self.pos += 1;
            }
        }
        if let Some(b'e' | b'E') = self.peek() {
            self.pos += 1;
            if let Some(b'+' | b'-') = self.peek() {
                self.pos += 1;
            }
            while let Some(b'0'..=b'9') = self.peek() {
                self.pos += 1;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| "JSON 数字编码错误".to_string())?;
        let num = text
            .parse::<f64>()
            .map_err(|_| format!("JSON 数字无效: {text}"))?;
        Ok(JsonValue::Number(num))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_json() {
        let raw = r#"{"__metadata__":{"format":"pt"},"w":{"dtype":"F32","shape":[2,4],"data_offsets":[0,32]}}"#;
        let val = parse_json(raw).unwrap();
        assert_eq!(
            val.get("__metadata__").unwrap().get_str("format"),
            Some("pt")
        );
        let w = val.get("w").unwrap();
        assert_eq!(w.get_str("dtype"), Some("F32"));
        assert_eq!(w.get("shape").unwrap().as_array().unwrap().len(), 2);
    }
}
