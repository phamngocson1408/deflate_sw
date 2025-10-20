from typing import List, Tuple, Dict

import configparser

def load_config(path: str):
    """Load INI config if exists. Expect [paths] with input_file, encode_table, outdir."""
    cfgp = Path(path)
    if not cfgp.exists():
        return None
    cfg = configparser.ConfigParser()
    cfg.read(cfgp.as_posix(), encoding='utf-8')
    return {
        'input_file':  cfg.get('paths', 'input_file', fallback=''),
        'encode_table': cfg.get('paths', 'encode_table', fallback='huffman_out/huffman_litlen_encode_table.csv'),
        'outdir': cfg.get('paths', 'outdir', fallback='deflate_out'),
    }


# -*- coding: utf-8 -*-
"""
3_deflate_csvbits_msb (1).py — LZ77 + Huffman (MSB-first via CSV code_bits)
===========================================================================
Mục tiêu
- Sao chép đầy đủ tính năng của 3_deflate.py nhưng đổi toàn bộ encode/decode
  sang **MSB-first** và **chỉ** dùng `code_bits` từ CSV (MSB-first) cho lit/len.

Điểm chính
- Header: "HLZ1" + version(=4 như gốc).
- Token count: ghi/đọc 32 bit **MSB-first** trước payload token.
- Lit/Len:
  * Encode: dùng `huffman_out/huffman_litlen_encode_table.csv` (cột: symbol, code_bits_msb, base_length, extra_bit).
  * Decode: dựng trie từ **encode CSV** để match prefix MSB-first (không cần fast_fulltable).
- Distance: fixed Huffman 5-bit (MSB-first) + extra bits MSB-first.
- Preview: in 20 symbol đầu của HLZ1 kèm code_bits (MSB) + loại symbol.
- Xuất biến thể "tokens-only" (bỏ header + version + n_tok).
- Xuất biến thể "tokenbits_512" (<= 512 bits đầu của payload token).
- Stats + summary.txt
- CLI: như bản gốc (compress/decompress) + chế độ mặc định chạy full pipeline nếu không có argv.
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List

# ===================== Utils (MSB-first bit packing) =====================
def bits_to_bytes_msb(bitstr: str) -> bytes:
    out = bytearray(); acc = 0; n = 0
    for ch in bitstr:
        bit = 1 if ch == '1' else 0
        acc = ((acc << 1) | bit) & 0xFF; n += 1
        if n == 8: out.append(acc); n = 0; acc = 0
    if n > 0: out.append((acc << (8 - n)) & 0xFF)
    return bytes(out)
def bytes_to_bits_msb(b: bytes) -> str:
    s = []
    for v in b:
        for k in range(7, -1, -1): s.append('1' if ((v >> k) & 1) else '0')
    return ''.join(s)
def u32_to_bits_msb(x: int) -> str: return format(x & 0xFFFFFFFF, '032b')
def bits_to_u32_msb(bs: str) -> int: return int(bs, 2) & 0xFFFFFFFF
def _printable_ascii(b: int) -> str: return chr(b) if 32 <= b <= 126 else '.'
def _preview_chars(label: str, b: bytes, n: int = 20) -> None:
    s = ''.join(_printable_ascii(x) for x in b[:n])
    print(f"{label} (first {min(n, len(b))} chars): {s}")

# ===================== LZ77 stream helpers =====================
def lz_stream_to_tokens(lz: bytes) -> List[Tuple]:
    toks = []; i = 0
    while i < len(lz):
        t = lz[i]; i += 1
        if t == 0x00:
            if i >= len(lz): raise ValueError("Stream .lz lỗi: thiếu literal")
            toks.append(("LIT", lz[i])); i += 1
        elif t == 0x01:
            if i + 2 >= len(lz): raise ValueError("Stream .lz lỗi: thiếu match")
            off = (lz[i] << 8) | lz[i+1]; L = lz[i+2]; i += 3
            toks.append(("MATCH", off, L))
        else: raise ValueError(f"Mã token .lz không hợp lệ: 0x{t:02X}")
    return toks
def tokens_to_lz_stream(tokens: List[Tuple]) -> bytes:
    out = bytearray()
    for t in tokens:
        if t[0] == "LIT":
            out.append(0x00); out.append(t[1] & 0xFF)
        else:
            _, off, L = t
            out.append(0x01)
            out.append((off >> 8) & 0xFF); out.append(off & 0xFF); out.append(L & 0xFF)
    return bytes(out)

# ===================== Import lz77_with_dict.py =====================
import importlib.util
def _import_lz77_module():
    mod_path = Path(__file__).with_name("lz77_with_dict.py")
    if not mod_path.exists(): raise FileNotFoundError("Thiếu lz77_with_dict.py cạnh script")
    spec = importlib.util.spec_from_file_location("lz77_with_dict", mod_path.as_posix())
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
_lz77 = _import_lz77_module()

# ===================== Lit/Len tables (MSB-first from CSV) =====================
class LitLenEncodeTableCSV_MSB:
    def __init__(self, csv_path: Path):
        self.enc_map: Dict[int, str] = {}
        self.len_rows: List[Tuple[int, int, int, str]] = []
        with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                sym = int(row['symbol'])
                cb = (row.get('code_bits_msb') or row.get('code_bits') or "").strip()
                if cb == "": raise ValueError("Thiếu cột code_bits_msb trong encode CSV")
                self.enc_map[sym] = cb
                bl = row.get('base_length', row.get('base_len', '0'))
                eb = row.get('extra_bit',  row.get('extra_bits','0'))
                try: base_len = int(float(bl))
                except: base_len = 0
                try: extra_bits = int(float(eb))
                except: extra_bits = 0
                if 257 <= sym <= 285: self.len_rows.append((sym, base_len, extra_bits, cb))
        self.len_rows.sort(key=lambda x: x[0])
    def encode_length(self, L: int) -> Tuple[int, str, int, int]:
        for sym, base_len, ebits, cb in self.len_rows:
            hi = base_len + ((1 << ebits) - 1) if ebits > 0 else base_len
            if base_len <= L <= hi: return sym, cb, ebits, (L - base_len)
        raise ValueError(f"Không tìm thấy length symbol cho L={L}")

class LitLenDecodeTrie_MSB:
    def __init__(self, csv_path: Path):
        self.trie = {}; self.meta = {}; self.max_len = 0
        with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                sym = int(row['symbol'])
                cb = (row.get('code_bits_msb') or row.get('code_bits') or "").strip()
                if cb == "": raise ValueError("Thiếu cột code_bits_msb trong encode CSV")
                bl = row.get('base_length', row.get('base_len', '0'))
                eb = row.get('extra_bit',  row.get('extra_bits','0'))
                try: base_len = int(float(bl))
                except: base_len = 0
                try: extra_bits = int(float(eb))
                except: extra_bits = 0
                node = self.trie
                for ch in cb: node = node.setdefault(ch, {})
                node['_leaf'] = True; self.meta[id(node)] = (sym, base_len, extra_bits)
                if len(cb) > self.max_len: self.max_len = len(cb)
    def decode_symbol(self, bit_iter) -> Tuple[int, int, int, int, str]:
        node = self.trie; buf = []
        for _ in range(self.max_len + 1):
            b = next(bit_iter); buf.append(b); node = node.get(b)
            if node is None: raise ValueError("Không decode được symbol (MSB)")
            if '_leaf' in node:
                sym, bl, eb = self.meta[id(node)]; return sym, bl, eb, len(buf), ''.join(buf)
        raise ValueError("Không decode được trong giới hạn max_len")

# ===================== Distance (fixed Huffman, MSB-first) =====================
DIST_TABLE = [
    (1,0),(2,0),(3,0),(4,0),(5,1),(7,1),(9,2),(13,2),
    (17,3),(25,3),(33,4),(49,4),(65,5),(97,5),(129,6),(193,6),
    (257,7),(385,7),(513,8),(769,8),(1025,9),(1537,9),(2049,10),(3073,10),
    (4097,11),(6145,11),(8193,12),(12289,12),(16385,13),(24577,13)
]
def encode_distance_bits_msb(dist: int) -> Tuple[str, int, int]:
    for code, (base, ebits) in enumerate(DIST_TABLE):
        hi = base + ((1 << ebits) - 1) if ebits > 0 else base
        if base <= dist <= hi: 
            return format(code, '05b'), ebits, (dist - base)
    raise ValueError(f"distance quá lớn: {dist}")
def decode_distance_bits_msb(bit_iter) -> int:
    code = int(''.join(next(bit_iter) for _ in range(5)), 2)
    if code >= len(DIST_TABLE): raise ValueError(f"distance code không hợp lệ: {code}")
    base, ebits = DIST_TABLE[code]
    extra = int(''.join(next(bit_iter) for _ in range(ebits)), 2) if ebits > 0 else 0
    return base + extra

# ===================== Pack/Unpack HLZ1 (MSB-first) =====================
def encode_with_huff_table_msb(tokens: List[Tuple], enc_tbl: LitLenEncodeTableCSV_MSB) -> bytes:
    bits = [u32_to_bits_msb(len(tokens))]
    for t in tokens:
        if t[0] == "LIT":
            bits.append(enc_tbl.enc_map[t[1]])
        else:
            _, off, L = t
            _, len_bits, len_ebits, len_extra = enc_tbl.encode_length(L)
            bits.append(len_bits)
            if len_ebits > 0: bits.append(format(len_extra, f"0{len_ebits}b"))
            d_bits, d_ebits, d_extra = encode_distance_bits_msb(off)
            bits.append(d_bits)
            if d_ebits > 0: bits.append(format(d_extra, f"0{d_ebits}b"))
    payload = bits_to_bytes_msb(''.join(bits))
    out = bytearray(); out.extend(b"HLZ1"); out.append(4); out.extend(payload); return bytes(out)
def decode_with_trie_msb(data: bytes, dec_trie: LitLenDecodeTrie_MSB) -> List[Tuple]:
    if data[:4] != b"HLZ1": raise ValueError("Bad magic")
    if data[4] < 4: raise ValueError("Version HLZ1 < 4")
    bitstr = bytes_to_bits_msb(data[5:]); it = iter(bitstr)
    n_tok = bits_to_u32_msb(''.join(next(it) for _ in range(32)))
    toks: List[Tuple] = []
    for _ in range(n_tok):
        sym, base_len, extra_bits, _, bits_used = dec_trie.decode_symbol(it)
        if 0 <= sym <= 255: toks.append(("LIT", sym))
        elif 257 <= sym <= 285:
            extra_val = int(''.join(next(it) for _ in range(extra_bits)), 2) if extra_bits > 0 else 0
            L = base_len + extra_val; dist = decode_distance_bits_msb(it)
            toks.append(("MATCH", dist, L))
        elif sym == 256: continue
        else: raise ValueError(f"Symbol không hợp lệ: {sym}")
    return toks

# ===================== Preview code_bits (MSB-first) =====================
def preview_hlz_codebits_msb(label: str, comp: bytes, enc_csv: Path, n: int = 20):
    if comp[:4] != b"HLZ1": print(f"{label}: bad magic"); return
    if comp[4] < 4: print(f"{label}: unsupported version"); return
    dec_trie = LitLenDecodeTrie_MSB(enc_csv)
    bits = bytes_to_bits_msb(comp[5:]); it = iter(bits)
    n_tok = bits_to_u32_msb(''.join(next(it) for _ in range(32)))
    outs = []; taken = 0
    while taken < n and taken < n_tok:
        sym, base_len, extra_bits, used, used_bits = dec_trie.decode_symbol(it)
        if 0 <= sym <= 255:
            desc = f"LIT '{_printable_ascii(sym)}'(0x{sym:02X})"; outs.append((used_bits, desc))
        elif 257 <= sym <= 285:
            eb = ''.join(next(it) for _ in range(extra_bits)) if extra_bits > 0 else ""
            dcode = ''.join(next(it) for _ in range(5))
            base, deb = DIST_TABLE[int(dcode, 2)]
            dexb = ''.join(next(it) for _ in range(deb)) if deb > 0 else ""
            L = base_len + (int(eb, 2) if eb else 0); dist = base + (int(dexb, 2) if dexb else 0)
            outs.append((used_bits + eb + dcode + dexb, f"MATCH len={L}, dist={dist}"))
        elif sym == 256: outs.append((used_bits, "EOB"))
        else: outs.append((used_bits, f"?sym={sym}"))
        taken += 1
    if not outs: print(f"{label}: (no symbols)"); return
    print(f"{label} (first {len(outs)} symbols):")
    for i, (bstr, desc) in enumerate(outs, 1):
        print(f"{i:02d}: {bstr}  — {desc}")

# ===================== Tokens-only / Tokenbits-512 =====================

def _emit_tokens_only(outdir: Path, comp: bytes, enc_csv: Path, stem: str = "tokens_chunked"):
    """
    Tạo tokens_chunked.mem theo yêu cầu mới:
      - Chunk = 4096 bytes uncompressed.
      - Với mỗi chunk: dùng phương pháp nén như hiện tại (decode tokens từ comp -> dùng chính bit của token).
      - Chia chunk thành các transfer 512-bit:
          + Normal transfer: 5b header = offset(0..31), + payload có đúng 'used_bits = 507 - offset' (KHÔNG đệm).
            (Các normal transfer sẽ được NỐI liên tục rồi mới cắt thành các dòng 512 bit khi ghi file .mem.)
          + Final transfer: 5b header = 11111, + payload còn lại + EOB(256) + pad 0 cho đủ 512 bit.
      - Chỉ ghi 20 chunk đầu tiên.
      - Ghi thêm file tokens_chunked.log.csv với thống kê theo chunk.
    """
    if comp[:4] != b"HLZ1":
        raise ValueError("Bad magic: HLZ1")

    outdir.mkdir(parents=True, exist_ok=True)

    # Trie decode cho lit/len và encode table để lấy EOB bits
    dec_trie = LitLenDecodeTrie_MSB(Path(enc_csv))
    enc_tbl  = LitLenEncodeTableCSV_MSB(Path(enc_csv))

    # Đọc toàn bộ bit sau header "HLZ1"+version (5 bytes), n_tok 32 bit
    all_bits = bytes_to_bits_msb(comp[5:])
    it = iter(all_bits)
    n_tok = bits_to_u32_msb(''.join(next(it) for _ in range(32)))

    # Thu thập bitstring của từng token + số byte uncompressed do token tạo ra
    token_bitstrs: List[str] = []
    token_uncomp:  List[int] = []  # LIT=1, MATCH=L
    for _ in range(n_tok):
        sym, base_len, extra_bits, _meta, sym_bits = dec_trie.decode_symbol(it)
        parts = [sym_bits]
        if 0 <= sym <= 255:
            # literal
            token_bitstrs.append(''.join(parts))
            token_uncomp.append(1)
        elif sym == 256:
            # EOB trong gói tổng thể: bỏ qua để tự chèn cho mỗi chunk
            # (không thêm vào token_bitstrs / token_uncomp)
            continue
        elif 257 <= sym <= 285:
            # len extra
            eb = ''.join(next(it) for _ in range(extra_bits)) if extra_bits > 0 else ""
            parts.append(eb)
            # distance code + extra
            dcode = ''.join(next(it) for _ in range(5))
            parts.append(dcode)
            _base, deb = DIST_TABLE[int(dcode, 2)]
            dexb = ''.join(next(it) for _ in range(deb)) if deb > 0 else ""
            if dexb:
                parts.append(dexb)
            token_bitstrs.append(''.join(parts))
            L = base_len + (int(eb, 2) if eb else 0)
            token_uncomp.append(L)
        else:
            raise ValueError(f"Symbol không hợp lệ: {sym}")

    # Hằng số đóng gói
    CHUNK_UC_BYTES = 4096
    CAP_NORM = 507                  # payload cap sau 5b header
    MIN_USED = 507 - 31             # để offset ∈ [0..31]
    EOB_BITS = enc_tbl.enc_map[256] # MSB-first

    # Gom token theo uncompressed 4096B
    chunks_tokbits: List[List[str]] = []
    chunks_ucbytes: List[int] = []
    chunks_tokcnt:  List[int] = []

    cur_bits_list: List[str] = []
    cur_uc = 0
    cur_tokcnt = 0

    for tb, uc in zip(token_bitstrs, token_uncomp):
        if cur_uc + uc > CHUNK_UC_BYTES and cur_uc > 0:
            # chốt chunk hiện tại
            chunks_tokbits.append(cur_bits_list)
            chunks_ucbytes.append(cur_uc)
            chunks_tokcnt.append(cur_tokcnt)
            # reset
            cur_bits_list = []
            cur_uc = 0
            cur_tokcnt = 0
        # add token vào chunk đang mở
        cur_bits_list.append(tb)
        cur_uc += uc
        cur_tokcnt += 1

    # chốt chunk cuối cùng nếu có
    if cur_tokcnt > 0:
        chunks_tokbits.append(cur_bits_list)
        chunks_ucbytes.append(cur_uc)
        chunks_tokcnt.append(cur_tokcnt)

    total_chunks = len(chunks_tokbits)
    write_chunks = min(total_chunks, 20)

    # Đóng gói thành các transfers (bitstring, KHÔNG pad normal-transfer)
    def build_transfers_for_chunk(tokbits: List[str]) -> Tuple[List[str], Dict[str, int]]:
        normal_transfers: List[str] = []
        remain_bits = []  # phần payload chưa flush -> sẽ vào final cùng EOB
        used = 0
        n_normal = 0
        total_payload_before_eob = 0

        def flush_normal():
            nonlocal remain_bits, used, n_normal, total_payload_before_eob
            if used == 0:
                return
            if used < MIN_USED:
                # Không đủ để tạo 1 normal-transfer hợp lệ (offset <=31),
                # giữ lại để dồn vào final.
                return
            offset = 507 - used
            header = format(offset, '05b')
            payload = ''.join(remain_bits)  # đúng = used bits
            normal_transfers.append(header + payload)  # KHÔNG pad, nối thẳng
            total_payload_before_eob += used
            n_normal += 1
            remain_bits = []
            used = 0

        for tb in tokbits:
            bl = len(tb)
            if bl > CAP_NORM:
                raise ValueError(f"Token {bl}b > 507b: không thể đóng gói transfer.")
            if used + bl > CAP_NORM:
                # nếu đang có đủ để flush normal
                if used >= MIN_USED:
                    flush_normal()
                # thêm token
                remain_bits.append(tb)
                used += bl
                if used > CAP_NORM:
                    raise ValueError("Vượt CAP_NORM sau khi thêm token.")
            else:
                remain_bits.append(tb)
                used += bl

        # Final transfer: header=11111 + remain + EOB
        header_final = '11111'
        final_payload = ''.join(remain_bits) + EOB_BITS
        total_payload_before_eob += used

        # Tổng bit từ các normal transfers
        sum_normals_bits = sum(len(s) for s in normal_transfers)

        # Bit trước pad của final (tạm thời)
        final_bits_no_pad = header_final + final_payload

        # Pad sao cho (sum_normals_bits + len(final_bits)) % 512 == 0
        rem = (sum_normals_bits + len(final_bits_no_pad)) % 512
        pad_len = (512 - rem) % 512
        final_bits = final_bits_no_pad + '0' * pad_len

        transfers = normal_transfers + [final_bits]
        stats = {
            "normal_transfers": n_normal,
            "final_total_bits": len(final_bits),
            "final_payload_bits_before_pad": used,
            "total_payload_bits_before_eob": total_payload_before_eob,
            "pad_bits_in_final": pad_len,
        }
        return transfers, stats

    # Viết .mem: nối TẤT CẢ transfers của từng chunk thành bitstream rồi cắt 512b/line
    mem_path = outdir / f"{stem}.mem"
    csv_path = outdir / f"{stem}.log.csv"
    with mem_path.open("w", encoding="utf-8") as fmem, csv_path.open("w", encoding="utf-8", newline="") as flog:
        w = csv.writer(flog)
        w.writerow(["chunk_idx","uc_bytes","tokens",
                    "num_transfers","normal_transfers",
                    "final_bits","final_payload_bits_before_pad",
                    "total_payload_bits_before_eob","final_pad_bits",
                    "lines_written"])
        total_lines = 0
        for cidx in range(write_chunks):
            transfers, st = build_transfers_for_chunk(chunks_tokbits[cidx])
            # Nối các transfer rồi cắt 512b/line
            bitstream = ''.join(transfers)
            # đảm bảo độ dài là bội số của 512 (do final đã pad cho chunk)
            if len(bitstream) % 512 != 0:
                # điều này không nên xảy ra
                raise AssertionError("Chunk bitstream không chia hết 512 mặc dù final đã pad.")
            nlines_before = total_lines
            for i in range(0, len(bitstream), 512):
                fmem.write(bitstream[i:i+512] + "\n")
                total_lines += 1
            w.writerow([cidx, chunks_ucbytes[cidx], chunks_tokcnt[cidx],
                        len(transfers), st["normal_transfers"],
                        st["final_total_bits"], st["final_payload_bits_before_pad"],
                        st["total_payload_bits_before_eob"], st["pad_bits_in_final"],
                        total_lines - nlines_before])

    print(f"[tokens-chunked] wrote -> {mem_path.as_posix()} (chunks_written={write_chunks}/{total_chunks})")
    print(f"[tokens-chunked] log    -> {csv_path.as_posix()}")

def _emit_tokenbits_512_only(outdir: Path, comp: bytes, enc_csv: Path, stem: str = "tokenbits_512"):
    """
    Emit 512-bit token bit-slice in three files:
      - {stem}.bits    : exact 512 bits (string '0'/'1')
      - {stem}.nbits   : meaningful bit count before padding
      - {stem}.hex     : 5 display lines:
           1) continuous hex (lower)
           2) continuous hex (UPPER)
           3) byte-spaced hex "aa bb ..."
           4) byte-colon hex "AA:BB:..."
           5) C array "0xAA,0xBB,..."
    """
    if comp[:4] != b"HLZ1":
        raise ValueError("Bad magic: HLZ1")
    dec_trie = LitLenDecodeTrie_MSB(enc_csv)
    bits = bytes_to_bits_msb(comp[5:])
    it = iter(bits)

    n_tok = bits_to_u32_msb(''.join(next(it) for _ in range(32)))

    total = 0
    used_bits_all = []
    taken = 0
    while taken < n_tok:
        sym, base_len, extra_bits, used, used_bits = dec_trie.decode_symbol(it)
        used_bits_all.append(used_bits)
        total += len(used_bits)

        if 257 <= sym <= 285:
            if extra_bits > 0:
                eb = ''.join(next(it) for _ in range(extra_bits))
                used_bits_all.append(eb)
                total += len(eb)
            dcode = ''.join(next(it) for _ in range(5))
            used_bits_all.append(dcode)
            total += 5
            base, deb = DIST_TABLE[int(dcode, 2)]
            if deb > 0:
                dexb = ''.join(next(it) for _ in range(deb))
                used_bits_all.append(dexb)
                total += len(dexb)

        if total >= 512:
            taken += 1
            break
        taken += 1

    payload_bits_raw = ''.join(used_bits_all)
    meaningful_bits = min(len(payload_bits_raw), 512)
    payload_bits = (payload_bits_raw + '0' * 512)[:512]

    outdir.mkdir(parents=True, exist_ok=True)

    bits_path = outdir / f"{stem}.bits"
    with bits_path.open("w", encoding="utf-8") as fbits:
        fbits.write(payload_bits)

    nbits_path = outdir / f"{stem}.nbits"
    with nbits_path.open("w", encoding="utf-8") as fmeta:
        fmeta.write(str(meaningful_bits) + "\n")

    bytes_512 = bits_to_bytes_msb(payload_bits)
    if len(bytes_512) != 64:
        bytes_512 = (bytes_512 + b"\x00" * 64)[:64]

    hex_path = outdir / f"{stem}.hex"

    hex_lower = bytes_512.hex()
    hex_upper = hex_lower.upper()

    def _chunk(s: str, n: int):
        return [s[i:i+n] for i in range(0, len(s), n)]

    byte_pairs = _chunk(hex_lower, 2)

    line3 = ' '.join(byte_pairs)
    line4 = ':'.join(bp.upper() for bp in byte_pairs)
    line5 = ','.join('0x' + bp.upper() for bp in byte_pairs)

    with hex_path.open("w", encoding="utf-8") as fhex:
        fhex.write(hex_lower + "\n")
        fhex.write(hex_upper + "\n")
        fhex.write(line3 + "\n")
        fhex.write(line4 + "\n")
        fhex.write(line5 + "\n")

    print(
        f"[tokenbits-512] wrote -> {bits_path.as_posix()} "
        f"(meaningful_bits={meaningful_bits}, padded={512 - meaningful_bits}, tokens={taken}); "
        f"hex(5 lines) -> {hex_path.as_posix()}"
    )



    n_tok = bits_to_u32_msb(''.join(next(it) for _ in range(32)))
    total = 0; used_bits_all = []; taken = 0
    while taken < n_tok:
        sym, base_len, extra_bits, used, used_bits = dec_trie.decode_symbol(it)
        used_bits_all.append(used_bits); total += len(used_bits)

        if 257 <= sym <= 285:
            if extra_bits > 0:
                eb = ''.join(next(it) for _ in range(extra_bits))
                used_bits_all.append(eb); total += len(eb)
            dcode = ''.join(next(it) for _ in range(5))
            used_bits_all.append(dcode); total += 5
            base, deb = DIST_TABLE[int(dcode, 2)]
            if deb > 0:
                dexb = ''.join(next(it) for _ in range(deb))
                used_bits_all.append(dexb); total += len(dexb)

        # Dừng khi đã vượt hoặc chạm 512 bit
        if total >= 512: 
            taken += 1  # token hiện tại đã được lấy vào
            break

        taken += 1

    # Ghép bit đã thu
    payload_bits_raw = ''.join(used_bits_all)

    # Số bit thực sự có ý nghĩa trong file đầu ra (bị cắt nếu >512, hoặc được pad nếu <512)
    meaningful_bits = min(len(payload_bits_raw), 512)

    # Pad đuôi bằng '0' cho đủ 512 bit
    payload_bits = (payload_bits_raw + '0' * 512)[:512]

    outdir.mkdir(parents=True, exist_ok=True)
    bits_path = outdir / f"{stem}.bits"
    with bits_path.open("w", encoding="utf-8") as fbits:
        fbits.write(payload_bits)

    # Ghi thêm file .nbits chứa số bit có ý nghĩa
    nbits_path = outdir / f"{stem}.nbits"
    with nbits_path.open("w", encoding="utf-8") as fmeta:
        fmeta.write(str(meaningful_bits) + "\n")

    # In ra màn hình
    print(
        f"[tokenbits-512] wrote -> {bits_path.as_posix()} "
        f"(meaningful_bits={meaningful_bits}, padded={512 - meaningful_bits}, tokens={taken})"
    )

def _verify_tokens_chunked_mem_nocmp(outdir: Path, enc_csv: Path, raw_input_path: Path | None = None):
    """
    Verify tokens_chunked.mem WITHOUT reading compressed.hlz.
    Uses tokens_chunked.meta.json (contains n_tok) and tokens_chunked.mem to reconstruct HLZ1 v4.
    Produces:
      - recon_from_mem_nocmp.hlz
      - recon_from_mem_nocmp.lz
      - (optional) recon_from_mem_nocmp.raw
      - verify_from_mem.txt
    """
    import json

    mem_path = outdir / "tokens_chunked.mem"
    meta_path = outdir / "tokens_chunked.meta.json"
    src_lz_path = outdir / "source.lz"

    if not mem_path.exists():
        raise FileNotFoundError(f"Missing {mem_path.as_posix()} (run compress first)")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path.as_posix()} (cannot infer n_tok)")
    if not src_lz_path.exists():
        raise FileNotFoundError(f"Missing {src_lz_path.as_posix()} (for .lz comparison)")

    # Read n_tok
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        n_tok = int(meta.get("n_tok"))
        if n_tok < 0:
            raise ValueError("n_tok < 0")
    except Exception as e:
        raise ValueError(f"Failed reading n_tok from {meta_path.name}: {e}")

    # Read bitstream from .mem
    bitstream = ''.join(line.strip() for line in mem_path.read_text(encoding="utf-8").splitlines())
    N = len(bitstream); idx = 0

    def _need(nbits: int):
        nonlocal idx, N
        if idx + nbits > N:
            raise ValueError(f"Bitstream underflow: need {nbits} @ {idx}/{N}")

    def _read(nbits: int) -> str:
        nonlocal idx
        _need(nbits)
        s = bitstream[idx:idx+nbits]
        idx += nbits
        return s

    # Parse chunks to recover token payload bits
    payload_bits_parts = []
    while True:
        hdr = _read(5)
        if hdr != '11111':
            offset = int(hdr, 2)
            used = 507 - offset
            if not (0 <= used <= 507):
                raise ValueError(f"Bad header: offset={offset} => used={used}")
            payload_bits_parts.append(_read(used))
        else:
            used9 = int(_read(9), 2)
            if not (0 <= used9 <= 498):
                raise ValueError(f"Bad final used_bits: {used9}")
            payload_bits_parts.append(_read(used9))
            break

    token_payload_bits = ''.join(payload_bits_parts)

    # Rebuild HLZ1 v4 from n_tok + token_payload_bits (MSB-first)
    rebuilt_payload_bits = u32_to_bits_msb(n_tok) + token_payload_bits
    rebuilt_payload_bytes = bits_to_bytes_msb(rebuilt_payload_bits)
    recon_hlz = b"HLZ1" + bytes([4]) + rebuilt_payload_bytes

    # Decode -> tokens -> .lz
    dec_trie = LitLenDecodeTrie_MSB(enc_csv)
    tokens = decode_with_trie_msb(recon_hlz, dec_trie)
    lz_stream = tokens_to_lz_stream(tokens)

    # Write outputs
    (outdir / "recon_from_mem_nocmp.hlz").write_bytes(recon_hlz)
    (outdir / "recon_from_mem_nocmp.lz").write_bytes(lz_stream)

    # Compare with source.lz
    src_lz = src_lz_path.read_bytes()
    same_lz = (lz_stream == src_lz)

    # Optional: compare to raw
    same_raw = None
    try:
        if raw_input_path and raw_input_path.exists() and hasattr(_lz77, "lz77_decompress"):
            raw_recon = _lz77.lz77_decompress(lz_stream)
            (outdir / "recon_from_mem_nocmp.raw").write_bytes(raw_recon)
            raw_bytes = raw_input_path.read_bytes()
            same_raw = (raw_recon == raw_bytes)
    except Exception as e:
        print("[verify/mem-nocmp] Skip raw compare due to error:", e)

    # Report
    report_lines = []
    report_lines.append("verify_from_mem (no compressed.hlz):")
    report_lines.append(f"  n_tok(meta): {n_tok}")
    report_lines.append(f"  bitstream(mem) length: {len(bitstream)}")
    report_lines.append(f"  payload_bits length: {len(token_payload_bits)}")
    report_lines.append(f"  .lz equals source.lz? {same_lz}")
    if same_raw is not None:
        report_lines.append(f"  raw equals input? {same_raw}")
    (outdir / "verify_from_mem.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if same_lz and (same_raw is True or same_raw is None):
        print("[verify/mem-nocmp] OK: recon_from_mem_nocmp.lz == source.lz", end="")
        if same_raw is True:
            print(" and matches input.raw 100%")
        else:
            print(" (raw not checked or no lz77_decompress)")
    else:
        print("[verify/mem-nocmp] NG: see verify_from_mem.txt for details")


def _emit_payload_hex_lines(outdir: Path, comp: bytes, enc_csv: Path, stem: str = "payload_lines"):
    """
    Write HEX lines of compressed token payload, với **mỗi dòng chứa các token nguyên vẹn**:

      - 512 bit đầu là payload gồm các bit **đúng như trong luồng nén** (theo thứ tự đọc từ compressed.hlz).
        * No token is split across lines.
        * Pad with trailing '0' to exactly 512 bits.
      - Next 9 bits: meaningful payload bit count before padding (0..512) for this line.
      - Final 7 bits: "is last line" flag (0000001 for last, 0000000 otherwise).

    Mỗi dòng có (512 + 9 + 7) = 528 bit = 66 byte. Ghi 1 dòng hex chữ thường cho mỗi dòng payload.
    Output file: {stem}.hex (each row is 132 hex chars).
    """
    if comp[:4] != b"HLZ1":
        raise ValueError("Bad magic: HLZ1")

    # Build token bitstrings exactly like _emit_tokens_only
    dec_trie = LitLenDecodeTrie_MSB(Path(enc_csv))
    all_bits = bytes_to_bits_msb(comp[5:])
    it = iter(all_bits)
    n_tok = bits_to_u32_msb(''.join(next(it) for _ in range(32)))

    token_bitstrs: List[str] = []
    for _ in range(n_tok):
        sym, _base_len, extra_bits, _meta, sym_bits = dec_trie.decode_symbol(it)
        parts = [sym_bits]
        if 257 <= sym <= 285:
            if extra_bits > 0:
                parts.append(''.join(next(it) for _ in range(extra_bits)))
            dcode = ''.join(next(it) for _ in range(5))
            parts.append(dcode)
            _base, deb = DIST_TABLE[int(dcode, 2)]
            if deb > 0:
                parts.append(''.join(next(it) for _ in range(deb)))
        token_bitstrs.append(''.join(parts))

    # Đóng gói: 512-bit payload (không cắt token), tiếp theo 9-bit used, cuối cùng 7-bit last flag.
    LINES: List[bytes] = []  # each is 66 bytes
    CAP = 512

    cur_bits: list[str] = []
    cur_used = 0

    def _flush(is_last: bool):
        nonlocal cur_bits, cur_used
        # Payload padded to 512
        payload = ''.join(cur_bits)
        used9 = format(cur_used, '09b')
        last7 = '0000001' if is_last else '0000000'
        payload_padded = (payload + ('0' * CAP))[:CAP]
        line_bits = payload_padded + used9 + last7  # 528 bits
        line_bytes = bits_to_bytes_msb(line_bits)
        # safety
        if len(line_bytes) != 66:
            line_bytes = (line_bytes + b'\x00' * 66)[:66]
        LINES.append(line_bytes)
        cur_bits = []
        cur_used = 0

    for idx, tb in enumerate(token_bitstrs):
        bl = len(tb)
        if bl > CAP:
            raise ValueError(f"Single token {bl}b exceeds 512-bit payload capacity.")
        if cur_used + bl > CAP:
            _flush(False)
        cur_bits.append(tb)
        cur_used += bl

    # final line (even if empty)
    _flush(True)

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{stem}.hex"
    
    # Also write a reference CSV to inspect each line
    csv_path = outdir / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8") as cf:
        cf.write("line_index,used_bits,is_last,payload_512b_hex,used9_bin,last7_bin,line_528b_hex\n")
        for i, row in enumerate(LINES):
            # Reconstruct fields for CSV:
            # Split row back into fields for clarity
            line_bits = bytes_to_bits_msb(row)
            payload_512 = line_bits[:512]
            used9 = line_bits[512:521]
            last7 = line_bits[521:528]
            payload_hex = bits_to_bytes_msb(payload_512).hex()
            cf.write(f"{i},{int(used9,2)},{1 if last7=='0000001' else 0},{payload_hex},{used9},{last7},{row.hex()}\n")

    with out_path.open("w", encoding="utf-8") as f:
        for row in LINES:
            f.write(row.hex() + "\n")

    # small sidecar (optional)
    meta_path = outdir / f"{stem}.meta.txt"
    with meta_path.open("w", encoding="utf-8") as fm:
        fm.write(f"lines={len(LINES)}  (each 528 bits = 66 bytes = 132 hex chars)\n")

    print(f"[payload-hex-lines] wrote -> {out_path.as_posix()} (lines={len(LINES)})")

# ===================== Stats & summary =====================
def _print_stats(raw_bytes: int, lz_bytes: int, hlz_bytes: int, prefix="[stats]"):
    if raw_bytes <= 0:
        print(prefix, "Raw size = 0 (bỏ qua thống kê)"); return
    ratio = hlz_bytes / raw_bytes; saving = 1.0 - ratio
    print(f"{prefix} raw={raw_bytes} B, lz={lz_bytes} B, hlz={hlz_bytes} B")
    print(f"{prefix} ratio=hlz/raw = {ratio:.4f}  ({saving*100:.2f}% tiết kiệm)")
def _write_summary(outdir: Path, row: dict):
    outdir.mkdir(parents=True, exist_ok=True); summ = outdir / "summary.txt"
    def _g(k, default=""): return row.get(k, default)
    line = (f"[{_g('timestamp', datetime.now().isoformat(timespec='seconds'))}] "
            f"input={_g('input')} raw={_g('raw_bytes')}B lz={_g('lz_bytes')}B hlz={_g('hlz_bytes')}B "
            f"ratio={_g('ratio_hlz_raw')} saving={_g('saving_percent')}% tokens={_g('tokens')} verify={_g('verify')}")
    with summ.open('a', encoding='utf-8') as f: f.write(line.strip() + "\n")

# ===================== CLI actions =====================
def cli_compress(args):
    enc_tbl_path = Path(args.encode_table)
    if not enc_tbl_path.exists():
        print("[error] Không tìm thấy encode-table MSB:", enc_tbl_path.as_posix())
        print("=> Hãy chạy 2_build_huffman_table_msb.py để sinh bảng."); sys.exit(1)
    raw_path = Path(args.inp); 
    if raw_path.is_dir():
        files = [f for f in sorted(raw_path.iterdir()) if f.is_file() and not f.name.startswith('.')]
        if not files: raise FileNotFoundError(f"Không tìm thấy file nào trong thư mục: {raw_path.as_posix()}")
        raw_path = files[0]
    elif not raw_path.exists():
        raise FileNotFoundError(f"Không thấy '{args.inp}'")
    print(f"[input] chọn file: {raw_path.as_posix()}")
    raw = raw_path.read_bytes(); _preview_chars("[preview] raw", raw, 20)
    lz_stream = _lz77.lz77_compress(raw, dict_bytes=None)
    tokens = lz_stream_to_tokens(lz_stream)
    enc_tbl = LitLenEncodeTableCSV_MSB(enc_tbl_path)
    comp = encode_with_huff_table_msb(tokens, enc_tbl)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir/"compressed.hlz").write_bytes(comp)
    (outdir/"source.lz").write_bytes(lz_stream)
    print("[compress] Wrote ->", (outdir/'compressed.hlz').as_posix(), f"(tokens={len(tokens)})")
    print("[compress] Saved LZ77 stream ->", (outdir/'source.lz').as_posix())
    try: _emit_tokens_only(outdir, comp, enc_tbl_path, stem="tokens_chunked")
    except Exception as e: print("[tokens-chunked/512] Bỏ qua do lỗi:", e)
    try: _emit_tokenbits_512_only(outdir, comp, enc_tbl_path, stem="tokenbits_512")
    except Exception as e: print("[tokenbits_512] Bỏ qua do lỗi:", e)
    # NEW: verify using tokens_chunked.mem + meta (no compressed.hlz)
    try:
        _verify_tokens_chunked_mem_nocmp(outdir, enc_tbl_path, raw_input_path=raw_path)
    except Exception as e:
        print("[verify/mem-nocmp] Verification error:", e)
    # NEW: payload HEX lines (512b payload + 9b used + 7b last)
    try:
        _emit_payload_hex_lines(outdir, comp, enc_tbl_path, stem="payload_lines")
    except Exception as e:
        print("[payload-hex-lines] Bỏ qua do lỗi:", e)
    # NEW(meta): write metadata for tokens_chunked (no dependency on compressed.hlz)
    try:
        import json as _json_for_meta
        _meta_obj = {"n_tok": int(len(tokens)), "format_version": 4}
        (outdir / "tokens_chunked.meta.json").write_text(_json_for_meta.dumps(_meta_obj) + "\n", encoding="utf-8")
        print("[meta] Wrote ->", (outdir/"tokens_chunked.meta.json").as_posix(), _meta_obj)
    except Exception as e:
        print("[meta] Failed to write tokens_chunked.meta.json:", e)
    preview_hlz_codebits_msb("[preview] hlz code_bits (MSB)", comp, enc_tbl_path, n=20)
    _print_stats(len(raw), len(lz_stream), len(comp))
    _write_summary(Path(args.outdir), {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'input': raw_path.as_posix(), 'raw_bytes': len(raw), 'lz_bytes': len(lz_stream),
        'hlz_bytes': len(comp), 'ratio_hlz_raw': f"{len(comp)/len(raw):.6f}" if len(raw)>0 else "",
        'saving_percent': f"{(1-len(comp)/len(raw))*100:.2f}" if len(raw)>0 else "",
        'tokens': len(tokens), 'verify': ''
    })
def cli_decompress(args):
    enc_tbl_path = Path(args.encode_table)
    if not enc_tbl_path.exists():
        print("[error] Không tìm thấy encode-table MSB:", enc_tbl_path.as_posix()); sys.exit(1)
    comp_path = Path(args.inp); comp = comp_path.read_bytes()
    dec_trie = LitLenDecodeTrie_MSB(enc_tbl_path)
    tokens = decode_with_trie_msb(comp, dec_trie)
    lz_stream = tokens_to_lz_stream(tokens)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir/"decompressed.lz").write_bytes(lz_stream)
    print("[decompress] Wrote ->", (outdir/'decompressed.lz').as_posix(), f"(tokens={len(tokens)})")

def _default_config_path():
    """Return best-guess path to config.ini without requiring --config.
    Search order:
      1) CWD/config.ini
      2) Script directory/config.ini
    """
    try:
        # 1) CWD
        cwd_cfg = Path.cwd() / "config.ini"
        if cwd_cfg.exists():
            return cwd_cfg.as_posix()
    except Exception:
        pass
    # 2) Script directory
    try:
        here = Path(__file__).resolve().parent / "config.ini"
        if here.exists():
            return here.as_posix()
    except Exception:
        pass
    # Fallback default name in CWD even if missing (load_config will return None)
    return "config.ini"


def apply_config_to_args(args):
    """Merge config into argparse Namespace. CLI values override config."""
    cfg_path = getattr(args, 'config', None) or _default_config_path()
    cfg = load_config(cfg_path)
    # Defaults
    enc = getattr(args, 'encode_table', None) or (cfg['encode_table'] if cfg else "huffman_out/huffman_litlen_encode_table.csv")
    outd = getattr(args, 'outdir', None) or (cfg['outdir'] if cfg else "deflate_out")
    # Determine input
    inp = getattr(args, 'inp', None)
    if not inp:
        if cfg and cfg['input_file']:
            inp = cfg['input_file']
    # Fallbacks honoring original defaults if still None
    if not inp:
        # Keep previous parser defaults if any
        inp = getattr(args, 'inp', None)
    # Write back
    setattr(args, 'encode_table', enc)
    setattr(args, 'outdir', outd)
    if inp:
        setattr(args, 'inp', inp)
    return args

def cli_full_pipeline():
    # Try config.ini first
    cfg = load_config(_default_config_path())
    if cfg:
        enc_tbl_path = Path(cfg['encode_table'] or "huffman_out/huffman_litlen_encode_table.csv")
        if not enc_tbl_path.exists():
            print("[error] Không tìm thấy encode-table MSB trong", enc_tbl_path.as_posix())
            print("=> Hãy chạy: python 2_build_huffman_table_msb.py"); sys.exit(1)
        inp_file = cfg['input_file']
        if not inp_file:
            raise FileNotFoundError("config.ini thiếu 'input_file' trong [paths]")
        raw_path = Path(inp_file)
        if not raw_path.exists():
            raise FileNotFoundError(f"Không thấy '{raw_path.as_posix()}'")
        outdir = cfg['outdir'] or "deflate_out"
        from types import SimpleNamespace
        args_c = SimpleNamespace(inp=raw_path.as_posix(), encode_table=enc_tbl_path.as_posix(), outdir=outdir)
        cli_compress(args_c)
        args_d = SimpleNamespace(inp=f"{outdir}/compressed.hlz", encode_table=enc_tbl_path.as_posix(), outdir=outdir)
        cli_decompress(args_d)
        return
    # Fallback to legacy behavior if no config.ini
    enc_tbl_path = Path("huffman_out/huffman_litlen_encode_table.csv")
    if not enc_tbl_path.exists():
        print("[error] Không tìm thấy encode-table MSB trong huffman_out/.")
        print("=> Hãy chạy: python 2_build_huffman_table_msb.py"); sys.exit(1)
    from types import SimpleNamespace
    raw_default = "data_set"
    rp = Path(raw_default)
    if rp.is_dir():
        files = [f for f in sorted(rp.iterdir()) if f.is_file() and not f.name.startswith('.')]
        if not files: raise FileNotFoundError(f"Không tìm thấy file nào trong thư mục: {rp.as_posix()}")
        picked = files[0]
    elif rp.exists():
        picked = rp
    else: raise FileNotFoundError(f"Không thấy '{raw_default}'")
    print(f"[input] chọn file: {picked.as_posix()}")
    args_c = SimpleNamespace(inp=picked.as_posix(), encode_table=enc_tbl_path.as_posix(), outdir="deflate_out")
    cli_compress(args_c)
    args_d = SimpleNamespace(inp="deflate_out/compressed.hlz", encode_table=enc_tbl_path.as_posix(), outdir="deflate_out")
    cli_decompress(args_d)
    src_lz = Path("deflate_out/source.lz").read_bytes()
    dec_lz = Path("deflate_out/decompressed.lz").read_bytes()
    ok = (src_lz == dec_lz)
    print("[verify] OK: decompressed.lz == source.lz" if ok else "[verify] NG: decompressed.lz khác source.lz")
    if Path("deflate_out/compressed.hlz").exists(): hlz_size = len(Path("deflate_out/compressed.hlz").read_bytes())
    else: hlz_size = ""
    raw_bytes = len(picked.read_bytes()) if picked.exists() else 0
    _write_summary(Path("deflate_out"), {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'input': picked.as_posix(),
        'raw_bytes': raw_bytes, 'lz_bytes': len(src_lz), 'hlz_bytes': hlz_size,
        'ratio_hlz_raw': f"{hlz_size/raw_bytes:.6f}" if (hlz_size!='' and raw_bytes>0) else '',
        'saving_percent': f"{(1-hlz_size/raw_bytes)*100:.2f}" if (hlz_size!='' and raw_bytes>0) else '',
        'tokens': '', 'verify': 'OK' if ok else 'NG'
    })

# ===================== CLI parser =====================
def build_parser():
    ap = argparse.ArgumentParser(description="3_deflate_csvbits_msb — LZ77 + Huffman MSB-first (CSV), token-only emit, preview, summary.")
    ap.add_argument("--config\", default=None, help=\"Tùy chọn: chỉ định file INI (mặc định: tự tìm config.ini)")
    sub = ap.add_subparsers(dest="cmd")
    p_c = sub.add_parser("compress", help="Nén: raw -> LZ77 -> Huffman (.hlz, distance fixed MSB)")
    p_c.add_argument("--in", dest="inp", default="inputs")
    p_c.add_argument("--encode-table", dest="encode_table", default="huffman_out/huffman_litlen_encode_table.csv")
    p_c.add_argument("--outdir", default="deflate_out"); p_c.set_defaults(func=cli_compress)
    p_d = sub.add_parser("decompress", help="Giải nén: .hlz -> token -> .lz")
    p_d.add_argument("--in", dest="inp", default="deflate_out/compressed.hlz")
    p_d.add_argument("--encode-table", dest="encode_table", default="huffman_out/huffman_litlen_encode_table.csv")
    p_d.add_argument("--outdir", default="deflate_out"); p_d.set_defaults(func=cli_decompress)
    return ap
def main():
    ap = build_parser(); argv = sys.argv[1:]
    if not argv: cli_full_pipeline(); return
    args = ap.parse_args(argv)
    args = apply_config_to_args(args)
    args.func(args)
if __name__=="__main__": main()
