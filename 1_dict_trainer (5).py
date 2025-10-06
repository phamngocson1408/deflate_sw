#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1_dict_trainer.py — phiên bản tích hợp training dictionary kiểu LZ77/Zstd
-------------------------------------------------------------------
Giữ nguyên CLI cũ (train, --pth, --dict-size, --line-bits, --pad-byte, --lowercase, ...)
Bổ sung tuỳ chọn --algo 
  - lz77 : greedy parse thu match thực tế
  - zstd : phong cách COVER/FastCover (đếm k-mer, chọn đoạn tối đa hoá độ phủ)

"""
from __future__ import annotations
import argparse
import os
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

# =============================
# LZ77 greedy parser (đơn giản)
# =============================

def _find_longest_match(data: bytes, pos: int, window: int, min_match: int, max_match: int) -> Tuple[int,int]:
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
        if remaining >= 4:
            out.extend(seg[:remaining])
            return True
        return False

    for score, seg, f in scored:
        if len(out) >= dict_size:
            break
        _try_append(seg)

    if len(out) < dict_size:
        out.extend(b"\0x00" * (dict_size - len(out)))
    return bytes(out[:dict_size])

# ==============================
# Zstd-style (COVER/FastCover-ish)
# ==============================

def extract_kmers(data: bytes, k: int, step: int) -> Counter:
    cnt = Counter()
    n = len(data)
    if n < k:
        return cnt
    for i in range(0, n - k + 1, step):
        seg = bytes(data[i:i+k])
        cnt[seg] += 1
    return cnt


def zstd_candidate_segments(data: bytes, k_min: int = 16, k_max: int = 64, k_step: int = 8, sample_step: int = 3, max_candidates: int = 20000, overhead: int = 3) -> List[Tuple[int, bytes, int]]:
    """Sinh ứng viên theo k-mer nhiều độ dài; chấm điểm freq*(k-overhead)."""
    total = Counter()
    for k in range(k_min, k_max+1, k_step):
        total.update(extract_kmers(data, k, sample_step))
    # chấm điểm
    scored: List[Tuple[int, bytes, int]] = []
    for seg, f in total.items():
        L = len(seg)
        gain = max(0, L - overhead) * f
        if gain > 0:
            scored.append((gain, seg, f))
    scored.sort(key=lambda x: x[0], reverse=True)
    if len(scored) > max_candidates:
        scored = scored[:max_candidates]
    return scored


def prune_contained(scored: List[Tuple[int, bytes, int]]) -> List[Tuple[int, bytes, int]]:
    """Loại bớt chuỗi là *con* của chuỗi mạnh đã chọn trước đó (xấp xỉ)."""
    kept: List[Tuple[int, bytes, int]] = []
    blob = bytearray()
    for score, seg, f in scored:
        if seg in blob:
            continue
        kept.append((score, seg, f))
        blob.extend(seg)
    return kept


def train_dictionary_zstd(data: bytes, dict_size: int = 4096, k_min: int = 16, k_max: int = 64, k_step: int = 8, sample_step: int = 3, overhead: int = 3) -> bytes:
    # 1) lấy ứng viên kiểu COVER: tổng hợp k-mer đa độ dài, chấm điểm theo lợi ích
    scored = zstd_candidate_segments(data, k_min, k_max, k_step, sample_step, overhead=overhead)
    # 2) loại chuỗi con để giảm trùng lặp
    scored = prune_contained(scored)
    # 3) nhồi vào dictionary
    return build_dictionary(scored, dict_size)

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
    with open(os.path.join(out_dir, 'dictionary.hex'), 'w') as f:
        for ln in lines:
            f.write(ln + "/n")
    with open(os.path.join(out_dir, 'dict_odd.hex'), 'w') as f:
        for idx, ln in enumerate(lines, 1):
            if idx % 2 == 1:
                f.write(ln + "/n")
    with open(os.path.join(out_dir, 'dict_even.hex'), 'w') as f:
        for idx, ln in enumerate(lines, 1):
            if idx % 2 == 0:
                f.write(ln + "/n")
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

# =============================
# Zstandard official library mode
# =============================

# Yêu cầu: pip install zstandard
try:
    import zstandard as zstd
    HAS_ZSTD = True
except Exception:
    HAS_ZSTD = False


def make_sliding_samples_from_file(pth: str, chunk: int, step: int) -> List[bytes]:
    data = load_bytes(pth)
    n = len(data)
    if n == 0 or chunk <= 0 or step <= 0:
        return []
    samples: List[bytes] = []
    i = 0
    while i < n:
        samples.append(bytes(data[i:i+chunk]))
        i += step
    return samples


def train_dictionary_zstdlib(paths: List[str], dict_size: int, chunk: int = 131072, step: int = 65536, k: int | None = None, d: int | None = None, threads: int = -1, dict_id: int | None = None) -> bytes:
    if not HAS_ZSTD:
        raise RuntimeError("Chưa cài đặt thư viện 'zstandard'. Hãy chạy: pip install zstandard")
    # gom samples
    samples: List[bytes] = []
    for p in paths:
        samples.extend(make_sliding_samples_from_file(p, chunk=chunk, step=step))
    if not samples:
        return b"\0x00" * dict_size
    # zstd.train_dictionary trả về ZstdCompressionDict hoặc bytes? -> trả bytes qua .as_bytes()
    dict_obj = zstd.train_dictionary(dict_size, samples, k=k, d=d, threads=threads, dict_id=dict_id)
    return dict_obj.as_bytes() if hasattr(dict_obj, 'as_bytes') else bytes(dict_obj)


# ==========
#   CLI
# ==========

from glob import glob

def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--pth', type=str, default='data_set/yolov8s_weights.pth', help='Đường dẫn file dữ liệu huấn luyện (có thể dùng glob)')
    p.add_argument('--dict-size', type=int, default=4096, help='Kích thước dictionary (byte)')
    p.add_argument('--line-bits', type=int, default=512, help='Số bit mỗi dòng khi xuất HEX (mặc định 512 = 64B)')
    p.add_argument('--pad-byte', type=lambda x: int(x, 0), default=0x00, help='Byte đệm cho dòng HEX/cuối file, ví dụ 0x00')
    p.add_argument('--lowercase', action='store_true', help='Xuất HEX chữ thường')
    p.add_argument('--out-dir', type=str, default='dicts_out')
    # Tuỳ chọn thuật toán
    p.add_argument('--algo', choices=['zstdlib','zstd','lz77'], default='zstdlib', help='Chọn thuật toán huấn luyện dictionary (mặc định: zstandard official library)')
    # Tham số LZ77
    p.add_argument('--min-match', type=int, default=3)
    p.add_argument('--max-match', type=int, default=64)
    p.add_argument('--window', type=int, default=32768)
    p.add_argument('--overhead', type=int, default=3, help='Ước lượng overhead mỗi match/segment (byte)')
    # Tham số Zstd-style (tự cài đặt)
    p.add_argument('--k-min', type=int, default=16, help='Độ dài k-mer tối thiểu (zstd-style)')
    p.add_argument('--k-max', type=int, default=64, help='Độ dài k-mer tối đa (zstd-style)')
    p.add_argument('--k-step', type=int, default=8, help='Bước tăng k (zstd-style)')
    p.add_argument('--sample-step', type=int, default=3, help='Bước lấy mẫu k-mer (zstd-style)')
    # Tham số Zstandard official library
    p.add_argument('--chunk', type=int, default=131072, help='Kích thước mẫu (bytes) khi trích sliding-window từ file)')
    p.add_argument('--step', type=int, default=65536, help='Bước trượt khi tạo sample từ file')
    p.add_argument('--k', type=int, default=0, help='Tham số k (COVER). 0 = để lib tự chọn')
    p.add_argument('--d', type=int, default=0, help='Tham số d (COVER). 0 = để lib tự chọn')
    p.add_argument('--threads', type=int, default=-1, help='Số luồng khi train dict (zstd)')
    p.add_argument('--dict-id', type=int, default=0, help='DICTID cố định; 0 = để lib tạo')



def _expand_paths(pat: str) -> List[str]:
    hits = glob(pat)
    return hits if hits else ([pat] if os.path.exists(pat) else [])


def cmd_train(args: argparse.Namespace) -> None:
    paths = _expand_paths(args.pth)
    if args.algo == 'lz77':
        data = b''.join(load_bytes(p) for p in paths) if paths else load_bytes(args.pth)
        dict_bytes = train_dictionary_lz77(
            data,
            dict_size=args.dict_size,
            min_match=args.min_match,
            max_match=args.max_match,
            window=args.window,
            overhead=args.overhead,
        )
    elif args.algo == 'zstd':
        data = b''.join(load_bytes(p) for p in paths) if paths else load_bytes(args.pth)
        dict_bytes = train_dictionary_zstd(
            data,
            dict_size=args.dict_size,
            k_min=args.k_min,
            k_max=args.k_max,
            k_step=args.k_step,
            sample_step=args.sample_step,
            overhead=args.overhead,
        )
    else:  # zstdlib (official)
        dict_bytes = train_dictionary_zstdlib(
            paths if paths else [args.pth],
            dict_size=args.dict_size,
            chunk=args.chunk,
            step=args.step,
            k=(None if args.k == 0 else args.k),
            d=(None if args.d == 0 else args.d),
            threads=args.threads,
            dict_id=(None if args.dict_id == 0 else args.dict_id),
        )
    save_hex_views(args.out_dir, dict_bytes, line_bits=args.line_bits, lowercase=args.lowercase, pad_byte=args.pad_byte)
    print(f"[OK] Saved: {args.out_dir}/dictionary.bin, dictionary.hex, dict_odd.hex, dict_even.hex")


def cmd_pipeline(args: argparse.Namespace) -> None:
    cmd_train(args)


def main():
    ap = argparse.ArgumentParser(description='Dictionary trainer (zstd official / zstd-style / LZ77) — giữ nguyên CLI cũ + --algo')
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
        cmd_pipeline(args)

if __name__ == '__main__':
    main()
