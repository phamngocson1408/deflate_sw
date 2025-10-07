#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1_dict_trainer.py — patched (zstd official / zstd-style / LZ77)
- Fix b"\0x00" -> b"\x00"
- Fix "/n" -> "\n"
- zstandard train_dictionary: never pass None; use ints (0/-1) to mean auto
- Keep CLI compatible; default --algo zstdlib
"""
from __future__ import annotations
import argparse
import os
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
from glob import glob
from pathlib import Path

# =============================
# LZ77 greedy parser (simple)
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
# Collect match segments & score
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
# Build fixed-size dictionary
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
        out.extend(b"\x00" * (dict_size - len(out)))
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
    total = Counter()
    for k in range(k_min, k_max+1, k_step):
        total.update(extract_kmers(data, k, sample_step))
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
    kept: List[Tuple[int, bytes, int]] = []
    blob = bytearray()
    for score, seg, f in scored:
        if seg in blob:
            continue
        kept.append((score, seg, f))
        blob.extend(seg)
    return kept

def train_dictionary_zstd(data: bytes, dict_size: int = 4096, k_min: int = 16, k_max: int = 64, k_step: int = 8, sample_step: int = 3, overhead: int = 3) -> bytes:
    if not data:
        return b"\x00" * dict_size
    scored = zstd_candidate_segments(data, k_min, k_max, k_step, sample_step, overhead=overhead)
    scored = prune_contained(scored)
    return build_dictionary(scored, dict_size)

# =============================
# Zstandard official library
# =============================

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

def train_dictionary_zstdlib(paths: List[str], dict_size: int, chunk: int = 131072, step: int = 65536, k: int = 0, d: int = 0, threads: int = -1, dict_id: int = 0) -> bytes:
    if not HAS_ZSTD:
        raise RuntimeError("Chưa cài đặt thư viện 'zstandard'. Hãy chạy: pip install zstandard")
    # collect samples
    samples: List[bytes] = []
    for p in paths:
        samples.extend(make_sliding_samples_from_file(p, chunk=chunk, step=step))
    if not samples:
        return b"\x00" * dict_size
    # zstandard 0.23.0 requires integers (no None)
    dict_obj = zstd.train_dictionary(int(dict_size), samples, k=int(k), d=int(d), threads=int(threads), dict_id=int(dict_id))
    return dict_obj.as_bytes() if hasattr(dict_obj, "as_bytes") else bytes(dict_obj)

# ======================
# I/O helpers & HEX
# ======================

def length_to_deflate_code(L: int) -> int:
    """
    Map a match length (3..258) to DEFLATE length code (257..285).
    """
    if L < 3:
        raise ValueError("DEFLATE length must be >= 3")
    if L <= 10:
        return 257 + (L - 3)
    table = [
        (265, 11, 1, 2), (266, 13, 1, 2), (267, 15, 1, 2), (268, 17, 1, 2),
        (269, 19, 2, 4), (270, 23, 2, 4), (271, 27, 2, 4), (272, 31, 2, 4),
        (273, 35, 3, 8), (274, 43, 3, 8), (275, 51, 3, 8), (276, 59, 3, 8),
        (277, 67, 4, 16), (278, 83, 4, 16), (279, 99, 4, 16), (280, 115, 4, 16),
        (281, 131, 5, 32), (282, 163, 5, 32), (283, 195, 5, 32), (284, 227, 5, 31),
    ]
    for code, base, bits, span in table:
        if L >= base and L <= base + span - 1:
            return code
    if L == 258:
        return 285
    raise ValueError(f"Unsupported length for DEFLATE: {L}")

def load_dictionary_hex(path: str = "dicts_out/dictionary.hex") -> bytes:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không thấy {path}")
    out = bytearray()
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        s = line.strip()
        # Chỉ giữ ký tự hex để tránh BOM/ký tự lạ
        import re as _re
        s = _re.sub(r"[^0-9A-Fa-f]", "", s)
        if not s:
            continue
        if len(s) % 2 != 0:
            raise ValueError(f"Dòng {lineno} có số ký tự hex lẻ: {line!r}")
        out += bytes.fromhex(s)
    return bytes(out)

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
    with open(os.path.join(out_dir, 'dictionary.hex'), 'w', encoding='utf-8') as f:
        for ln in lines:
            f.write(ln + "\n")
    with open(os.path.join(out_dir, 'dict_odd.hex'), 'w', encoding='utf-8') as f:
        for idx, ln in enumerate(lines, 1):
            if idx % 2 == 1:
                f.write(ln + "\n")
    with open(os.path.join(out_dir, 'dict_even.hex'), 'w', encoding='utf-8') as f:
        for idx, ln in enumerate(lines, 1):
            if idx % 2 == 0:
                f.write(ln + "\n")
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

# ---------------------------
# LZ77 (dict-only) compressor
# ---------------------------

def _build_index(history: bytes, start_pos: int, end_pos: int) -> dict:
    idx = {}
    H = history
    if end_pos - start_pos < 3:
        return idx
    for i in range(start_pos, end_pos - 2):
        key = H[i:i+3]
        idx.setdefault(key, []).append(i)
    return idx

def _longest_match_in_dict(dict_bytes: bytes,
                           lookahead: bytes,
                           min_match: int,
                           max_match: int,
                           index):
    n = len(lookahead)
    if n < min_match or len(dict_bytes) < min_match:
        return (0, 0)
    best_len = 0
    best_off = 0
    if n >= 3 and index is not None:
        key = lookahead[:3]
        for j in reversed(index.get(key, [])):
            L = 0
            while L < n and L < max_match and (j + L) < len(dict_bytes) and dict_bytes[j + L] == lookahead[L]:
                L += 1
            if L >= min_match and L > best_len:
                best_len = L
                best_off = len(dict_bytes) - j
                if best_len == max_match:
                    break
    else:
        for j in range(0, len(dict_bytes) - min_match + 1):
            L = 0
            while L < n and L < max_match and (j + L) < len(dict_bytes) and dict_bytes[j + L] == lookahead[L]:
                L += 1
            if L >= min_match and L > best_len:
                best_len = L
                best_off = len(dict_bytes) - j
    return (best_off, best_len) if best_len >= min_match else (0, 0)

def lz77_compress(data: bytes,
                  dict_bytes: bytes = b"",
                  window_size: int = 65536,
                  min_match: int = 3,
                  max_match: int = 255) -> bytes:
    out = bytearray()
    if dict_bytes:
        start = max(0, len(dict_bytes) - window_size)
        index = _build_index(dict_bytes, start, len(dict_bytes))
    else:
        index = {}
    i = 0
    while i < len(data):
        lookahead = data[i:i+max_match]
        off, L = _longest_match_in_dict(dict_bytes, lookahead, min_match, max_match, index)
        if L >= min_match:
            out.append(0x01); out.append((off>>8)&0xFF); out.append(off&0xFF); out.append(L&0xFF)
            i += L
        else:
            out.append(0x00); out.append(data[i]); i += 1
    return bytes(out)

# ---------------------------
# Stats / CSV emit
# ---------------------------

def parse_stream_for_stats(stream: bytes):
    lit=Counter(); length=Counter(); distance=Counter(); i=0
    while i<len(stream):
        token=stream[i]; i+=1
        if token==0x00:
            if i>=len(stream): raise ValueError("Stream lỗi: thiếu literal byte")
            lit[stream[i]]+=1; i+=1
        elif token==0x01:
            if i+3>len(stream): raise ValueError("Stream lỗi: thiếu offset/length")
            off=(stream[i]<<8)|stream[i+1]; i+=2; L=stream[i]; i+=1
            length[L]+=1; distance[off]+=1
        else:
            raise ValueError(f"Token không hợp lệ: {token:#x}")
    return lit,length,distance

# ==========
#   CLI
# ==========

def _expand_paths(pat: str) -> List[str]:
    hits = glob(pat)
    return hits if hits else ([pat] if os.path.exists(pat) else [])

def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--pth', type=str, default='data_set/yolov8s_weights.pth', help='Đường dẫn file dữ liệu huấn luyện (có thể dùng glob)')
    p.add_argument('--dict-size', type=int, default=4096, help='Kích thước dictionary (byte)')
    p.add_argument('--line-bits', type=int, default=512, help='Số bit mỗi dòng khi xuất HEX (mặc định 512 = 64B)')
    p.add_argument('--pad-byte', type=lambda x: int(x, 0), default=0x00, help='Byte đệm cho dòng HEX/cuối file, ví dụ 0x00')
    p.add_argument('--lowercase', action='store_true', help='Xuất HEX chữ thường')
    p.add_argument('--out-dir', type=str, default='dicts_out')
    # Algorithm
    p.add_argument('--algo', choices=['zstdlib','zstd','lz77'], default='zstdlib', help='Chọn thuật toán huấn luyện dictionary (mặc định: zstandard official library)')
    # LZ77 params
    p.add_argument('--min-match', type=int, default=3)
    p.add_argument('--max-match', type=int, default=64)
    p.add_argument('--window', type=int, default=32768)
    p.add_argument('--overhead', type=int, default=3, help='Ước lượng overhead mỗi match/segment (byte)')
    # zstd-style params
    p.add_argument('--k-min', type=int, default=16, help='Độ dài k-mer tối thiểu (zstd-style)')
    p.add_argument('--k-max', type=int, default=64, help='Độ dài k-mer tối đa (zstd-style)')
    p.add_argument('--k-step', type=int, default=8, help='Bước tăng k (zstd-style)')
    p.add_argument('--sample-step', type=int, default=3, help='Bước lấy mẫu k-mer (zstd-style)')
    # zstandard official params
    p.add_argument('--chunk', type=int, default=131072, help='Kích thước mẫu (bytes) khi trích sliding-window từ file)')
    p.add_argument('--step', type=int, default=65536, help='Bước trượt khi tạo sample từ file')
    p.add_argument('--k', type=int, default=0, help='Tham số k (COVER). 0 = để lib tự chọn')
    p.add_argument('--d', type=int, default=0, help='Tham số d (COVER). 0 = để lib tự chọn')
    p.add_argument('--threads', type=int, default=-1, help='Số luồng khi train dict (zstd)')
    p.add_argument('--dict-id', type=int, default=0, help='DICTID cố định; 0 = để lib tạo')

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
            k=args.k,
            d=args.d,
            threads=args.threads,
            dict_id=args.dict_id,
        )
    save_hex_views(args.out_dir, dict_bytes, line_bits=args.line_bits, lowercase=args.lowercase, pad_byte=args.pad_byte)
    print(f"[OK] Saved: {args.out_dir}/dictionary.bin, dictionary.hex, dict_odd.hex, dict_even.hex")

def cli_stats(args):
    inp=Path(args.inp)
    if not inp.exists(): raise FileNotFoundError(f"Không thấy input: {inp}")
    dict_b=load_dictionary_hex(args.dict_path)
    raw=inp.read_bytes()
    stream=lz77_compress(raw, dict_b, window_size=args.window_size,
                         min_match=args.min_match, max_match=args.max_match)
    lit,length,distance=parse_stream_for_stats(stream)
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # separate CSVs
    (outdir/"literal_freq.csv").write_text(
        "value,count\n" + "\n".join(f"{v},{c}" for v,c in sorted(lit.items())), encoding="utf-8"
    )
    (outdir/"length_freq.csv").write_text(
        "length,count\n" + "\n".join(f"{L},{c}" for L,c in sorted(length.items())), encoding="utf-8"
    )

    (outdir/"distance_freq.csv").write_text(
        "distance,count\n" + "\n".join(f"{d},{c}" for d,c in sorted(distance.items())), encoding="utf-8"
    )

    # combined (sorted by count desc; include kind column)
    combined = []
    combined.extend([("literal", int(v), int(c)) for v, c in lit.items()])
    combined.extend([("length", int(length_to_deflate_code(int(L))), int(c)) for L, c in length.items()])
# removed: excluded 'distance' from combined_freq.csv
    combined.sort(key=lambda x: (-x[2], x[0], x[1]))
    (outdir / "combined_freq.csv").write_text(
        "kind,symbol,count\n" + "\n".join(f"{k},{s},{cnt}" for k,s,cnt in combined),
        encoding="utf-8"
    )

    # summary
    total_tokens=sum(lit.values())+sum(length.values())
    summary=[
        f"Input file: {inp.name}",
        f"Dictionary: {args.dict_path} (bytes: {len(dict_b)})",
        f"Compressed bytes: {len(stream)}",
        f"Total tokens: {total_tokens} (literal: {sum(lit.values())}, match: {sum(length.values())})",
    ]
    if length: summary.append("Top match lengths: " + ", ".join(f"{L}:{c}" for L,c in length.most_common(10)))
    if distance: summary.append("Top distances: " + ", ".join(f"{d}:{c}" for d,c in distance.most_common(10)))
    if lit:    summary.append("Top literal values: " + ", ".join(f"{v}:{c}" for v,c in lit.most_common(10)))
    (outdir/"summary.txt").write_text("\n".join(summary), encoding="utf-8")

def cmd_pipeline(args: argparse.Namespace) -> None:
    cmd_train(args)

def main():
    ap = argparse.ArgumentParser(description='Dictionary trainer (zstd official / zstd-style / LZ77) — patched')
    sub = ap.add_subparsers(dest='cmd')
    # default (no subcommand): pipeline
    add_common_args(ap)
    # train subcommand
    p_train = sub.add_parser('train', help='Train dictionary và xuất các view HEX/BIN')
    add_common_args(p_train)
    args = ap.parse_args()
    if args.cmd == 'train':
        cmd_train(args)
    else:
        cmd_pipeline(args)

    import types
    targs = types.SimpleNamespace(
            pth="data_set/yolov8s_weights.pth",
            dict_size=4096,
            line_bits=512,
            pad_byte=0x00,
            lowercase=False,
            even="dicts_out/dict_even.hex",
            odd="dicts_out/dict_odd.hex",
            out="dicts_out/dictionary.hex",
            bin="dicts_out/dictionary.bin",
        )

    sargs = types.SimpleNamespace(
            inp=targs.pth,
            dict_path=targs.out,
            outdir="stats_out",
            min_match=3,
            max_match=255,
            window_size=65536,
        )
    cli_stats(sargs)

if __name__ == '__main__':
    main()
