#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lz77_with_dict.py — patched AUTO pipeline
----------------------------------------
- Preset dictionary only (matches reference the dictionary, not prior output).
- Token format:
    * LITERAL  : 0x00 | <len:1> | <len bytes>
    * MATCH    : 0x01 | <offset:2 big-endian> | <length:1>
      where offset=1 means the LAST byte of dictionary.
- Decompress copies from dictionary slice **left->right**.
- AUTO mode: for each file in ./inputs (except *.lz), do
    compress -> decompress -> compare -> write report.tsv
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple, List

INPUT_DIR  = Path("inputs")
OUTPUT_DIR = Path("lz77_out")
DICT_BIN   = Path("dicts_out/dictionary.bin")
DICT_HEX   = Path("dicts_out/dictionary.hex")

def load_dictionary(dict_path: Path | None = None) -> bytes:
    if dict_path is not None:
        p = Path(dict_path)
        if p.is_file():
            try:
                b = p.read_bytes()
                if b:
                    return b
            except Exception:
                pass
            return _read_dictionary_hex(p)
    if DICT_BIN.is_file():
        return DICT_BIN.read_bytes()
    if DICT_HEX.is_file():
        return _read_dictionary_hex(DICT_HEX)
    raise FileNotFoundError(f"Không tìm thấy dictionary. Hãy tạo bằng 1_dict_trainer.py -> {DICT_BIN} hoặc {DICT_HEX}")

def _read_dictionary_hex(hex_path: Path) -> bytes:
    text = hex_path.read_text(encoding="utf-8", errors="ignore")
    filtered = "".join(ch for ch in text if ch.isdigit() or ("a" <= ch.lower() <= "f"))
    if len(filtered) % 2 == 1:
        filtered = filtered[:-1]
    return bytes.fromhex(filtered)

TAG_LIT  = 0x00
TAG_MATCH= 0x01

def _u16_be(b0: int, b1: int) -> int:
    return (b0 << 8) | b1

def _split_u16_be(v: int) -> Tuple[int,int]:
    return ((v >> 8) & 0xFF, v & 0xFF)

def lz77_compress(data: bytes, dict_b: bytes, min_match: int = 3, max_match: int = 255) -> bytes:
    out = bytearray()
    i = 0
    n = len(data)
    lit = bytearray()
    while i < n:
        off, L = _find_best_match_in_dict(data, i, dict_b, min_match, max_match)
        if L >= min_match:
            if lit:
                _emit_literal(out, bytes(lit))
                lit.clear()
            offset = len(dict_b) - off
            if not (1 <= offset <= len(dict_b)):
                raise ValueError(f"Invalid offset computed: {offset}")
            _emit_match(out, offset, L)
            i += L
        else:
            lit.append(data[i])
            if len(lit) == 255:
                _emit_literal(out, bytes(lit))
                lit.clear()
            i += 1
    if lit:
        _emit_literal(out, bytes(lit))
    return bytes(out)

def _emit_literal(out: bytearray, run: bytes) -> None:
    j = 0
    while j < len(run):
        chunk = run[j:j+255]
        out.append(TAG_LIT)
        out.append(len(chunk))
        out.extend(chunk)
        j += len(chunk)

def _emit_match(out: bytearray, offset: int, length: int) -> None:
    out.append(TAG_MATCH)
    b0, b1 = _split_u16_be(offset)
    out.append(b0); out.append(b1)
    out.append(length & 0xFF)

def _find_best_match_in_dict(data: bytes, pos: int, dict_b: bytes, min_match: int, max_match: int) -> Tuple[int,int]:
    best_len = 0
    best_src = 0
    n = len(data)
    m = len(dict_b)
    max_len_possible = min(max_match, n - pos)
    if max_len_possible < min_match:
        return (0, 0)
    for s in range(0, m):
        if dict_b[s] != data[pos]:
            continue
        L = 1
        while L < max_len_possible and s + L < m and dict_b[s + L] == data[pos + L]:
            L += 1
        if L >= min_match and L > best_len:
            best_len = L
            best_src = s
            if best_len == max_len_possible:
                break
    return (best_src, best_len) if best_len >= min_match else (0, 0)

def lz77_decompress(stream: bytes, dict_b: bytes) -> bytes:
    i = 0
    out = bytearray()
    n = len(stream)
    D = len(dict_b)
    while i < n:
        tag = stream[i]; i += 1
        if tag == TAG_LIT:
            if i >= n:
                raise ValueError("Truncated literal header")
            ln = stream[i]; i += 1
            if i + ln > n:
                raise ValueError("Truncated literal data")
            out.extend(stream[i:i+ln])
            i += ln
        elif tag == TAG_MATCH:
            if i + 3 > n:
                raise ValueError("Truncated match header")
            b0 = stream[i]; b1 = stream[i+1]; ln = stream[i+2]
            i += 3
            off = _u16_be(b0, b1)
            if off == 0 or off > D:
                raise ValueError(f"Invalid offset: {off}")
            src = D - off
            if src + ln > D:
                raise ValueError(f"Match exceeds dictionary: src={src}, len={ln}, D={D}")
            out.extend(dict_b[src:src+ln])
        else:
            raise ValueError(f"Bad tag 0x{tag:02X} at pos {i-1}")
    return bytes(out)

def _ensure_dirs() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _compress_decompress_verify(in_path: Path, dict_b: bytes) -> tuple[Path, Path, bool, int]:
    raw = in_path.read_bytes()
    enc = lz77_compress(raw, dict_b)
    enc_path = OUTPUT_DIR / (in_path.name + ".lz")
    enc_path.parent.mkdir(parents=True, exist_ok=True)
    enc_path.write_bytes(enc)
    dec = lz77_decompress(enc, dict_b)
    dec_path = OUTPUT_DIR / (in_path.name + ".roundtrip")
    dec_path.write_bytes(dec)
    if dec == raw:
        return enc_path, dec_path, True, -1
    mm = -1
    for idx, (a, b) in enumerate(zip(raw, dec)):
        if a != b:
            mm = idx
            break
    if mm == -1 and len(raw) != len(dec):
        mm = min(len(raw), len(dec))
    return enc_path, dec_path, False, mm

def run_auto_mode(dict_b: bytes) -> int:
    _ensure_dirs()
    files = [p for p in INPUT_DIR.iterdir() if p.is_file() and not p.name.startswith(".")]
    if not files:
        print(f"Không tìm thấy file nào trong '{INPUT_DIR.as_posix()}'.")
        print("→ Hãy đặt các file cần kiểm tra round-trip vào thư mục này rồi chạy lại.")
        return 0
    src_files = [p for p in files if p.suffix.lower() != ".lz"]
    if not src_files:
        print("Không có file nguồn để round-trip (chỉ thấy .lz). Bỏ qua.")
        return 0
    report_lines = ["filename\tinput_bytes\tenc_bytes\tdec_bytes\tok\tfirst_mismatch\tenc_path\tdec_path"]
    n_ok = 0
    for p in sorted(src_files):
        try:
            enc_path, dec_path, is_ok, mismatch = _compress_decompress_verify(p, dict_b)
            raw_sz = p.stat().st_size
            enc_sz = enc_path.stat().st_size
            dec_sz = dec_path.stat().st_size
            report_lines.append(f"{p.name}\t{raw_sz}\t{enc_sz}\t{dec_sz}\t{int(is_ok)}\t{mismatch}\t{enc_path.as_posix()}\t{dec_path.as_posix()}")
            status = "OK" if is_ok else f"FAIL@{mismatch}"
            print(f"[{status}] {p.name}: raw={raw_sz}B, enc={enc_sz}B, dec={dec_sz}B")
            if is_ok:
                n_ok += 1
        except Exception as e:
            print(f"[ERR] {p.name}: {e}")
    (OUTPUT_DIR / "report.tsv").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nTổng kết: {n_ok}/{len(src_files)} round-trip khớp 100%. Báo cáo: {(OUTPUT_DIR/'report.tsv').as_posix()}")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser(description="LZ77 with preset dictionary — AUTO round-trip")
    ap.add_argument("--dict", type=str, default=None, help="Đường dẫn dictionary (BIN hoặc HEX). Nếu bỏ trống, ưu tiên BIN mặc định rồi HEX.")
    ap.add_argument("--mode", choices=["auto", "compress", "decompress"], default="auto")
    ap.add_argument("--input", type=str, default=None, help="File input cho chế độ compress/decompress")
    ap.add_argument("--output", type=str, default=None, help="Đường dẫn output khi chạy compress/decompress")
    args = ap.parse_args()
    dict_b = load_dictionary(Path(args.dict) if args.dict else None)
    if args.mode == "auto":
        return run_auto_mode(dict_b)
    elif args.mode == "compress":
        if not args.input:
            raise SystemExit("--mode compress yêu cầu --input")
        in_path = Path(args.input)
        data = in_path.read_bytes()
        enc = lz77_compress(data, dict_b)
        out_path = Path(args.output) if args.output else (Path(str(in_path)).with_suffix(in_path.suffix + ".lz"))
        out_path.write_bytes(enc)
        print(f"Đã nén -> {out_path.as_posix()}")
        return 0
    else:
        if not args.input:
            raise SystemExit("--mode decompress yêu cầu --input")
        in_path = Path(args.input)
        enc = in_path.read_bytes()
        dec = lz77_decompress(enc, dict_b)
        out_path = Path(args.output) if args.output else (Path(str(in_path)).with_suffix(in_path.suffix + ".roundtrip"))
        out_path.write_bytes(dec)
        print(f"Đã giải nén -> {out_path.as_posix()}")
        return 0

if __name__ == "__main__":
    raise SystemExit(main())
