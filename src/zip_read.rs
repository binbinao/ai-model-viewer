//! Minimal ZIP central directory reader and RFC 1951 DEFLATE decompressor.
//!
//! Used by PyTorch (`.pt` / `.pth`) and Keras v3 (`.keras`) archive parsers to
//! inspect `data.pkl`, `code/*.py`, `version`, and `config.json` without
//! extracting multi-gigabyte tensor payload entries.

#[derive(Debug, Clone)]
pub struct ZipEntryInfo {
    pub name: String,
    pub method: u16,
    pub compressed_size: usize,
    pub uncompressed_size: usize,
    pub local_header_offset: usize,
}

pub fn looks_like_zip(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && &bytes[..4] == b"PK\x03\x04"
}

pub fn list_zip_entries(bytes: &[u8]) -> Result<Vec<ZipEntryInfo>, String> {
    if !looks_like_zip(bytes) {
        return Err("不是有效的 ZIP 归档".into());
    }

    // Try Central Directory via EOCD at the end of the file first
    if let Some(entries) = read_central_directory(bytes) {
        if !entries.is_empty() {
            return Ok(entries);
        }
    }

    // Fallback: scan local file headers from offset 0
    scan_local_headers(bytes)
}

pub fn read_zip_entry(bytes: &[u8], entry: &ZipEntryInfo) -> Result<Vec<u8>, String> {
    let pos = entry.local_header_offset;
    if pos + 30 > bytes.len() || &bytes[pos..pos + 4] != b"PK\x03\x04" {
        return Err(format!("ZIP 条目 {} 本地头无效", entry.name));
    }
    let name_len = u16::from_le_bytes(bytes[pos + 26..pos + 28].try_into().unwrap()) as usize;
    let extra_len = u16::from_le_bytes(bytes[pos + 28..pos + 30].try_into().unwrap()) as usize;
    let data_start = pos + 30 + name_len + extra_len;
    let data_end = data_start
        .checked_add(entry.compressed_size)
        .ok_or_else(|| "ZIP 条目长度溢出".to_string())?;
    if data_end > bytes.len() {
        return Err(format!("ZIP 条目 {} 数据被截断", entry.name));
    }
    let slice = &bytes[data_start..data_end];
    match entry.method {
        0 => Ok(slice.to_vec()),
        8 => inflate_raw(slice, entry.uncompressed_size),
        other => Err(format!("不支持的 ZIP 压缩方式 {other}")),
    }
}

fn read_central_directory(bytes: &[u8]) -> Option<Vec<ZipEntryInfo>> {
    let min_eocd = 22;
    if bytes.len() < min_eocd {
        return None;
    }
    let search_start = bytes.len().saturating_sub(65557);
    let mut eocd_pos = None;
    for i in (search_start..=bytes.len() - min_eocd).rev() {
        if &bytes[i..i + 4] == b"PK\x05\x06" {
            eocd_pos = Some(i);
            break;
        }
    }
    let eocd = eocd_pos?;
    let total_entries = u16::from_le_bytes(bytes[eocd + 10..eocd + 12].try_into().ok()?) as usize;
    let cd_size = u32::from_le_bytes(bytes[eocd + 12..eocd + 16].try_into().ok()?) as usize;
    let cd_offset = u32::from_le_bytes(bytes[eocd + 16..eocd + 20].try_into().ok()?) as usize;

    // Handle Zip64 EOCD locator if values are 0xFFFF / 0xFFFFFFFF
    let (mut pos, count) = if cd_offset == 0xFFFF_FFFF || total_entries == 0xFFFF {
        read_zip64_eocd(bytes, eocd)?
    } else {
        if cd_offset + cd_size > bytes.len() {
            return None;
        }
        (cd_offset, total_entries)
    };

    let mut out = Vec::with_capacity(count.min(16384));
    for _ in 0..count {
        if pos + 46 > bytes.len() || &bytes[pos..pos + 4] != b"PK\x01\x02" {
            break;
        }
        let method = u16::from_le_bytes(bytes[pos + 10..pos + 12].try_into().ok()?);
        let mut comp_size =
            u32::from_le_bytes(bytes[pos + 20..pos + 24].try_into().ok()?) as usize;
        let mut uncomp_size =
            u32::from_le_bytes(bytes[pos + 24..pos + 28].try_into().ok()?) as usize;
        let name_len = u16::from_le_bytes(bytes[pos + 28..pos + 30].try_into().ok()?) as usize;
        let extra_len = u16::from_le_bytes(bytes[pos + 30..pos + 32].try_into().ok()?) as usize;
        let comment_len = u16::from_le_bytes(bytes[pos + 32..pos + 34].try_into().ok()?) as usize;
        let mut local_off =
            u32::from_le_bytes(bytes[pos + 42..pos + 46].try_into().ok()?) as usize;

        let name_end = pos + 46 + name_len;
        let extra_end = name_end + extra_len;
        if extra_end + comment_len > bytes.len() {
            break;
        }
        let name = String::from_utf8_lossy(&bytes[pos + 46..name_end]).into_owned();

        // Parse Zip64 extra field if needed
        if comp_size == 0xFFFF_FFFF || uncomp_size == 0xFFFF_FFFF || local_off == 0xFFFF_FFFF {
            parse_zip64_extra(
                &bytes[name_end..extra_end],
                &mut uncomp_size,
                &mut comp_size,
                &mut local_off,
            );
        }

        out.push(ZipEntryInfo {
            name,
            method,
            compressed_size: comp_size,
            uncompressed_size: uncomp_size,
            local_header_offset: local_off,
        });
        pos = extra_end + comment_len;
    }
    Some(out)
}

fn read_zip64_eocd(bytes: &[u8], eocd: usize) -> Option<(usize, usize)> {
    if eocd < 20 || &bytes[eocd - 20..eocd - 16] != b"PK\x06\x07" {
        return None;
    }
    let z64_off = u64::from_le_bytes(bytes[eocd - 12..eocd - 4].try_into().ok()?) as usize;
    if z64_off + 56 > bytes.len() || &bytes[z64_off..z64_off + 4] != b"PK\x06\x06" {
        return None;
    }
    let count = u64::from_le_bytes(bytes[z64_off + 32..z64_off + 40].try_into().ok()?) as usize;
    let cd_off = u64::from_le_bytes(bytes[z64_off + 48..z64_off + 56].try_into().ok()?) as usize;
    Some((cd_off, count))
}

fn parse_zip64_extra(
    mut extra: &[u8],
    uncomp: &mut usize,
    comp: &mut usize,
    local_off: &mut usize,
) {
    while extra.len() >= 4 {
        let id = u16::from_le_bytes([extra[0], extra[1]]);
        let sz = u16::from_le_bytes([extra[2], extra[3]]) as usize;
        extra = &extra[4..];
        if sz > extra.len() {
            break;
        }
        if id == 0x0001 {
            let mut body = &extra[..sz];
            if *uncomp == 0xFFFF_FFFF && body.len() >= 8 {
                *uncomp = u64::from_le_bytes(body[..8].try_into().unwrap()) as usize;
                body = &body[8..];
            }
            if *comp == 0xFFFF_FFFF && body.len() >= 8 {
                *comp = u64::from_le_bytes(body[..8].try_into().unwrap()) as usize;
                body = &body[8..];
            }
            if *local_off == 0xFFFF_FFFF && body.len() >= 8 {
                *local_off = u64::from_le_bytes(body[..8].try_into().unwrap()) as usize;
            }
            break;
        }
        extra = &extra[sz..];
    }
}

fn scan_local_headers(bytes: &[u8]) -> Result<Vec<ZipEntryInfo>, String> {
    let mut pos = 0usize;
    let mut out = Vec::new();
    while pos + 30 <= bytes.len() && &bytes[pos..pos + 4] == b"PK\x03\x04" {
        let flags = u16::from_le_bytes(bytes[pos + 6..pos + 8].try_into().unwrap());
        let method = u16::from_le_bytes(bytes[pos + 8..pos + 10].try_into().unwrap());
        let comp_size = u32::from_le_bytes(bytes[pos + 18..pos + 22].try_into().unwrap()) as usize;
        let uncomp_size =
            u32::from_le_bytes(bytes[pos + 22..pos + 26].try_into().unwrap()) as usize;
        let name_len = u16::from_le_bytes(bytes[pos + 26..pos + 28].try_into().unwrap()) as usize;
        let extra_len = u16::from_le_bytes(bytes[pos + 28..pos + 30].try_into().unwrap()) as usize;
        let header_end = pos + 30 + name_len + extra_len;
        if header_end > bytes.len() {
            break;
        }
        let name = String::from_utf8_lossy(&bytes[pos + 30..pos + 30 + name_len]).into_owned();
        if (flags & 0x08) != 0 && comp_size == 0 {
            break;
        }
        out.push(ZipEntryInfo {
            name,
            method,
            compressed_size: comp_size,
            uncompressed_size: uncomp_size,
            local_header_offset: pos,
        });
        pos = header_end + comp_size;
    }
    Ok(out)
}

/// Minimal RFC 1951 raw DEFLATE decompressor.
pub fn inflate_raw(input: &[u8], expected_size: usize) -> Result<Vec<u8>, String> {
    let mut bits = BitReader {
        bytes: input,
        byte_pos: 0,
        bit_pos: 0,
    };
    let mut out = Vec::with_capacity(expected_size.min(16 * 1024 * 1024));

    loop {
        let bfinal = bits.read_bits(1)?;
        let btype = bits.read_bits(2)?;
        match btype {
            0 => {
                bits.align_byte();
                let len = bits.read_u16()? as usize;
                let _nlen = bits.read_u16()?;
                let slice = bits.read_bytes(len)?;
                out.extend_from_slice(slice);
            }
            1 => {
                let (lit_tree, dist_tree) = fixed_trees();
                decode_huffman_block(&mut bits, &lit_tree, &dist_tree, &mut out)?;
            }
            2 => {
                let (lit_tree, dist_tree) = read_dynamic_trees(&mut bits)?;
                decode_huffman_block(&mut bits, &lit_tree, &dist_tree, &mut out)?;
            }
            _ => return Err("DEFLATE 块类型无效".into()),
        }
        if bfinal == 1 {
            break;
        }
    }
    Ok(out)
}

struct BitReader<'a> {
    bytes: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    fn read_bits(&mut self, count: u8) -> Result<u32, String> {
        let mut out = 0u32;
        for i in 0..count {
            if self.byte_pos >= self.bytes.len() {
                return Err("DEFLATE 流被截断".into());
            }
            let bit = (self.bytes[self.byte_pos] >> self.bit_pos) & 1;
            out |= (bit as u32) << i;
            self.bit_pos += 1;
            if self.bit_pos == 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }
        Ok(out)
    }

    fn align_byte(&mut self) {
        if self.bit_pos != 0 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
    }

    fn read_u16(&mut self) -> Result<u16, String> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], String> {
        self.align_byte();
        if self.byte_pos + len > self.bytes.len() {
            return Err("DEFLATE 字节块被截断".into());
        }
        let s = &self.bytes[self.byte_pos..self.byte_pos + len];
        self.byte_pos += len;
        Ok(s)
    }
}

#[derive(Default)]
struct HuffTree {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl HuffTree {
    fn from_lengths(lengths: &[u8]) -> Result<Self, String> {
        let mut counts = [0u16; 16];
        for &len in lengths {
            if len as usize >= 16 {
                return Err("Huffman 码长超过 15".into());
            }
            if len > 0 {
                counts[len as usize] += 1;
            }
        }
        let mut offsets = [0u16; 16];
        let mut total = 0u16;
        for bits in 1..16 {
            offsets[bits] = total;
            total += counts[bits];
        }
        let mut symbols = vec![0u16; total as usize];
        for (sym, &len) in lengths.iter().enumerate() {
            if len > 0 {
                let idx = offsets[len as usize] as usize;
                offsets[len as usize] += 1;
                symbols[idx] = sym as u16;
            }
        }
        Ok(Self { counts, symbols })
    }

    fn decode(&self, bits: &mut BitReader<'_>) -> Result<u16, String> {
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0i32;
        for len in 1..16 {
            code |= bits.read_bits(1)? as i32;
            let count = self.counts[len] as i32;
            if code - count < first {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        Err("Huffman 解码失败".into())
    }
}

fn fixed_trees() -> (HuffTree, HuffTree) {
    let mut lit = vec![0u8; 288];
    lit[..144].fill(8);
    lit[144..256].fill(9);
    lit[256..280].fill(7);
    lit[280..288].fill(8);
    let dist = vec![5u8; 32];
    (
        HuffTree::from_lengths(&lit).unwrap(),
        HuffTree::from_lengths(&dist).unwrap(),
    )
}

fn read_dynamic_trees(bits: &mut BitReader<'_>) -> Result<(HuffTree, HuffTree), String> {
    let hlit = bits.read_bits(5)? as usize + 257;
    let hdist = bits.read_bits(5)? as usize + 1;
    let hclen = bits.read_bits(4)? as usize + 4;
    const ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    let mut code_lengths = [0u8; 19];
    for i in 0..hclen {
        code_lengths[ORDER[i]] = bits.read_bits(3)? as u8;
    }
    let code_tree = HuffTree::from_lengths(&code_lengths)?;

    let total = hlit + hdist;
    let mut lengths = Vec::with_capacity(total);
    while lengths.len() < total {
        let sym = code_tree.decode(bits)?;
        match sym {
            0..=15 => lengths.push(sym as u8),
            16 => {
                let prev = *lengths
                    .last()
                    .ok_or_else(|| "DEFLATE 重复码缺少前值".to_string())?;
                let repeat = bits.read_bits(2)? as usize + 3;
                for _ in 0..repeat {
                    lengths.push(prev);
                }
            }
            17 => {
                let repeat = bits.read_bits(3)? as usize + 3;
                lengths.resize(lengths.len() + repeat, 0);
            }
            18 => {
                let repeat = bits.read_bits(7)? as usize + 11;
                lengths.resize(lengths.len() + repeat, 0);
            }
            _ => return Err("DEFLATE 码长符号无效".into()),
        }
    }

    let lit_tree = HuffTree::from_lengths(&lengths[..hlit])?;
    let dist_tree = HuffTree::from_lengths(&lengths[hlit..total])?;
    Ok((lit_tree, dist_tree))
}

fn decode_huffman_block(
    bits: &mut BitReader<'_>,
    lit_tree: &HuffTree,
    dist_tree: &HuffTree,
    out: &mut Vec<u8>,
) -> Result<(), String> {
    const LEN_BASE: [usize; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
        131, 163, 195, 227, 258,
    ];
    const LEN_EXTRA: [u8; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    const DIST_BASE: [usize; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
        2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
    ];
    const DIST_EXTRA: [u8; 30] = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
        13, 13,
    ];

    loop {
        let sym = lit_tree.decode(bits)? as usize;
        if sym < 256 {
            out.push(sym as u8);
        } else if sym == 256 {
            break;
        } else if sym <= 285 {
            let l_idx = sym - 257;
            let len = LEN_BASE[l_idx] + bits.read_bits(LEN_EXTRA[l_idx])? as usize;
            let d_sym = dist_tree.decode(bits)? as usize;
            if d_sym >= 30 {
                return Err("DEFLATE 距离符号越界".into());
            }
            let dist = DIST_BASE[d_sym] + bits.read_bits(DIST_EXTRA[d_sym])? as usize;
            if dist == 0 || dist > out.len() {
                return Err("DEFLATE 回溯距离越界".into());
            }
            for _ in 0..len {
                let b = out[out.len() - dist];
                out.push(b);
            }
        } else {
            return Err("DEFLATE 长度符号越界".into());
        }
    }
    Ok(())
}

/// Helper for tests: build an uncompressed (method 0) ZIP archive from `(name, bytes)` pairs.
#[cfg(test)]
pub fn build_stored_zip(files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut cd = Vec::new();

    for &(name, data) in files {
        let local_off = out.len() as u32;
        out.extend_from_slice(b"PK\x03\x04");
        out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&0u16.to_le_bytes()); // method 0 (stored)
        out.extend_from_slice(&0u16.to_le_bytes()); // time
        out.extend_from_slice(&0u16.to_le_bytes()); // date
        out.extend_from_slice(&0u32.to_le_bytes()); // crc32
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra len
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(data);

        cd.extend_from_slice(b"PK\x01\x02");
        cd.extend_from_slice(&20u16.to_le_bytes());
        cd.extend_from_slice(&20u16.to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes());
        cd.extend_from_slice(&0u32.to_le_bytes());
        cd.extend_from_slice(&(data.len() as u32).to_le_bytes());
        cd.extend_from_slice(&(data.len() as u32).to_le_bytes());
        cd.extend_from_slice(&(name.len() as u16).to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes());
        cd.extend_from_slice(&0u32.to_le_bytes());
        cd.extend_from_slice(&local_off.to_le_bytes());
        cd.extend_from_slice(name.as_bytes());
    }

    let cd_off = out.len() as u32;
    let cd_size = cd.len() as u32;
    out.extend_from_slice(&cd);

    out.extend_from_slice(b"PK\x05\x06");
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(files.len() as u16).to_le_bytes());
    out.extend_from_slice(&(files.len() as u16).to_le_bytes());
    out.extend_from_slice(&cd_size.to_le_bytes());
    out.extend_from_slice(&cd_off.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out
}
