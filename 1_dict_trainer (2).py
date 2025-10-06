#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1_dict_trainer.py — phiên bản tích hợp training dictionary kiểu LZ77
-------------------------------------------------------------------
Giữ nguyên CLI cũ (train, --pth, --dict-size, --line-bits, --pad-byte, --lowercase, ...)
nhưng thay thế phần sinh dictionary bằng pipeline LZ77-style:
  - Greedy parse để thu các match (offset,length)
  - Thu thập các đoạn (segment) thực sự được tham chiếu bởi match
  - Chấm điểm lợi ích xấp xỉ: score = freq * max(0, len(seg) - overhead)
  - Nhét vào dictionary cố định kích thước dict_size, loại lặp/con-tained

Gợi ý: với dữ liệu lớn, thay bộ tìm match bằng hash-chain để tăng tốc.
"""
from __future__ import annotations
import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

# =============================
# LZ77 greedy parser (đơn giản)
# =============================

def _find_longest_match(data: bytes, pos: int, window: int, min_match: int, max_match: int) -> Tuple[int,int]:
    """Tìm (offset, length) dài nhất trong cửa sổ [max(0,pos-window), pos).
    Trả về (0,0) nếu không có match >= min_match.
    
    Lưu ý: Cài đặt đơn giản O(window * max_match) dùng để minh hoạ.
    Với tập lớn, thay bằng hash-table/rolling-hash.
    """
    start = max(0, pos - window)
    best_len = 0
    best_off = 0
    if pos >= len(data):
        return (0, 0)
    for s in range(pos - 1, start - 1, -1):
        if data[s] != data[pos]:
            continue
        length = 1
        limit = min(max_match, len(data) - pos)
        while length < limit and data[s + length] == data[pos + length]:
            length += 1
        if length >= min_match and length > best_len:
            best_len = length
            best_off = pos - s
            if best_len == max_match:
                break
    return (best_off, best_len) if best_len >= min_match else (0, 0)

@dataclass
class Token:
    is_match: bool
    lit: bytes = b""
    offset: int = 0
    length: int = 0


def lz77_greedy_parse(data: bytes, window: int = 32768, min_match: int = 3, max_match: int = 64) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    n = len(data)
    lit_buf = bytearray()
    while i < n:
        off, L = _find_longest_match(data, i, window, min_match, max_match)
        if L:
            if lit_buf:
                tokens.append(Token(False, bytes(lit_buf)))
                lit_buf.clear()
            tokens.append(Token(True, b"", off, L))
            i += L
        else:
            lit_buf.append(data[i])
            i += 1
    if lit_buf:
        tokens.append(Token(False, bytes(lit_buf)))
    return tokens

# ============================================
# Thu thập chuỗi match & tính điểm lợi ích nén
# ============================================

def collect_match_segments(data: bytes, tokens: Iterable[Token]) -> Dict[bytes, int]:
    """Đếm tần suất xuất hiện *chuỗi match* trong dữ liệu.
    Với mỗi token match (offset,length), trích ra chuỗi data[pos:pos+length].
    Để biết pos cho từng token, mô phỏng lại quá trình đọc theo tokens.
    """
    seg_freq: Dict[bytes, int] = defaultdict(int)
    pos = 0
    for tk in tokens:
        if tk.is_match:
            seg = data[pos:pos + tk.length]
            if seg:
                seg_freq[bytes(seg)] += 1
            pos += tk.length
        else:
            pos += len(tk.lit)
    return seg_freq


def score_segments(seg_freq: Dict[bytes, int], overhead_per_match: int = 3) -> List[Tuple[int, bytes, int]]:
    """Trả về [(score, seg, freq)] sắp xếp giảm dần.
    score = freq * max(0, len(seg) - overhead_per_match)
    """
    scored: List[Tuple[int, bytes, int]] = []
    for seg, f in seg_freq.items():
        L = len(seg)
        gain = max(0, L - overhead_per_match) * f
        if gain > 0:
            scored.append((gain, seg, f))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

# ==========================
# Dựng dictionary cố định
# ==========================

def build_dictionary(scored: List[Tuple[int, bytes, int]], dict_size: int) -> bytes:
    out = bytearray()
    added = set()

    def _try_append(seg: bytes) -> bool:
        if not seg:
            return False
        if seg in added:
            return False
        # nếu seg đã là chuỗi con của những gì đã có trong out, bỏ qua
        if bytes(seg) in out:
            added.add(seg)
            return False
        remaining = dict_size - len(out)
        if remaining <= 0:
            return False
        if len(seg) <= remaining:
            out.extend(seg)
            added.add(seg)
            return True
        # cắt ngắn phần đuôi nếu còn >= 4 byte
        if remaining >= 4:
            out.extend(seg[:remaining])
            return True
        return False

    for score, seg, f in scored:
        if len(out) >= dict_size:
            break
        _try_append(seg)

    if len(out) < dict_size:
        out.extend(b"\x00" * (dict_size - len(out)))
    return bytes(out[:dict_size])

# ======================
# I/O helpers & xuất HEX
# ======================

def load_bytes(pth: str) -> bytes:
    with open(pth, 'rb') as f:
        return f.read()


def save_bin(pth: str, blob: bytes) -> None:
    with open(pth, 'wb') as f:
        f.write(blob)


def to_hex_lines(blob: bytes, line_bits: int = 512, lowercase: bool = False, pad_byte: int = 0x00) -> List[str]:
    line_bytes = max(1, line_bits // 8)
    fmt = (lambda s: s.lower()) if lowercase else (lambda s: s.upper())
    lines: List[str] = []
    for i in range(0, len(blob), line_bytes):
        chunk = bytearray(blob[i:i+line_bytes])
        if len(chunk) < line_bytes:
            chunk.extend(bytes([pad_byte]) * (line_bytes - len(chunk)))
        lines.append(fmt(bytes(chunk).hex()))
    return lines


def save_hex_views(out_dir: str, dict_bytes: bytes, line_bits: int = 512, lowercase: bool = False, pad_byte: int = 0x00) -> None:
    os.makedirs(out_dir, exist_ok=True)
    lines = to_hex_lines(dict_bytes, line_bits=line_bits, lowercase=lowercase, pad_byte=pad_byte)
    # full
    with open(os.path.join(out_dir, 'dictionary.hex'), 'w') as f:
        for ln in lines:
            f.write(ln + "\n")
    # odd / even
    with open(os.path.join(out_dir, 'dict_odd.hex'), 'w') as f:
        for idx, ln in enumerate(lines, 1):
            if idx % 2 == 1:
                f.write(ln + "\n")
    with open(os.path.join(out_dir, 'dict_even.hex'), 'w') as f:
        for idx, ln in enumerate(lines, 1):
            if idx % 2 == 0:
                f.write(ln + "\n")
    # bin
    save_bin(os.path.join(out_dir, 'dictionary.bin'), dict_bytes)

# ============
# Training API
# ============

def train_dictionary_lz77(data: bytes, dict_size: int = 4096, min_match: int = 3, max_match: int = 64, window: int = 32768, overhead: int = 3) -> bytes:
    if not data:
        return b"\x00" * dict_size
    tokens = lz77_greedy_parse(data, window=window, min_match=min_match, max_match=max_match)
    seg_freq = collect_match_segments(data, tokens)
    scored = score_segments(seg_freq, overhead_per_match=overhead)
    return build_dictionary(scored, dict_size)

# ==========
#   CLI
# ==========

def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--pth', type=str, default='data_set/yolov8s_weights.pth', help='Đường dẫn file dữ liệu huấn luyện')
    p.add_argument('--dict-size', type=int, default=4096, help='Kích thước dictionary (byte)')
    p.add_argument('--line-bits', type=int, default=512, help='Số bit mỗi dòng khi xuất HEX (mặc định 512 = 64B)')
    p.add_argument('--pad-byte', type=lambda x: int(x, 0), default=0x00, help='Byte đệm cho dòng HEX/cuối file, ví dụ 0x00')
    p.add_argument('--lowercase', action='store_true', help='Xuất HEX chữ thường')
    # Tham số LZ77
    p.add_argument('--min-match', type=int, default=3)
    p.add_argument('--max-match', type=int, default=64)
    p.add_argument('--window', type=int, default=32768)
    p.add_argument('--overhead', type=int, default=3, help='Ước lượng overhead mỗi match (byte)')
    p.add_argument('--out-dir', type=str, default='dicts_out')


def cmd_train(args: argparse.Namespace) -> None:
    data = load_bytes(args.pth)
    dict_bytes = train_dictionary_lz77(
        data,
        dict_size=args.dict_size,
        min_match=args.min_match,
        max_match=args.max_match,
        window=args.window,
        overhead=args.overhead,
    )
    save_hex_views(args.out_dir, dict_bytes, line_bits=args.line_bits, lowercase=args.lowercase, pad_byte=args.pad_byte)
    print(f"[OK] Saved: {args.out_dir}/dictionary.bin, dictionary.hex, dict_odd.hex, dict_even.hex")


def cmd_pipeline(args: argparse.Namespace) -> None:
    # Giữ hành vi cũ: gọi train (và có thể in thống kê nhẹ nếu cần)
    cmd_train(args)


def main():
    ap = argparse.ArgumentParser(description='Dictionary trainer (LZ77-integrated) — giữ nguyên CLI cũ')
    sub = ap.add_subparsers(dest='cmd')

    # Mặc định không subcommand: pipeline
    add_common_args(ap)

    # Subcommand: train (giữ nguyên tên)
    p_train = sub.add_parser('train', help='Train dictionary và xuất các view HEX/BIN')
    add_common_args(p_train)

    args = ap.parse_args()

    if args.cmd == 'train':
        cmd_train(args)
    else:
        # Không có subcommand: chạy pipeline mặc định
        cmd_pipeline(args)

if __name__ == '__main__':
    main()
