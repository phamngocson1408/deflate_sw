"""
1_dict_trainer.py
=====================

Script tích hợp **huấn luyện dictionary** và **thống kê nén LZ77**.

MẶC ĐỊNH (chạy không tham số):
- Đọc file `yolov8s_weights.pth`
- Sinh dictionary vào thư mục `dicts_out/`:
    * dicts_out/dictionary.hex
    * dicts_out/dict_even.hex
    * dicts_out/dict_odd.hex
    * dicts_out/dictionary.bin
- Nén file đầu vào với dictionary và xuất thống kê ra `stats_out/`:
    * stats_out/literal_freq.csv
    * stats_out/length_freq.csv
    * stats_out/summary.txt

TÍNH NĂNG
---------
1. Huấn luyện dictionary:
   - Đọc file nhị phân mẫu (ví dụ: yolov8s_weights.pth).
   - Sinh dictionary cố định kích thước (mặc định 4096 bytes).
   - Xuất ra các file dictionary trong `dicts_out/`.

2. Thống kê LZ77:
   - Nén dữ liệu dựa vào dictionary (chỉ tham chiếu trong dictionary).
   - Đếm tần suất literal (0..255) và match length (1..255).
   - Xuất thống kê ra CSV và summary.txt trong `stats_out/`.

CÁCH DÙNG
---------
1) Chạy mặc định (pipeline đầy đủ):
   python 1_dict_trainer.py

2) Chỉ huấn luyện dictionary:
   python 1_dict_trainer.py train --pth yolov8s_weights.pth

3) Chỉ thống kê:
   python 1_dict_trainer.py stats --in yolov8s_weights.pth --dict dicts_out/dictionary.hex --outdir stats_out
"""

import argparse
from pathlib import Path
from collections import Counter
from typing import List
import csv

# ---------------------------
# Helpers: I/O & HEX utilities
# ---------------------------
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


def load_bytes_from_file(path: str) -> bytes:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Không tìm thấy file đầu vào: {path}")
    return p.read_bytes()

def _bytes_to_hex_line(b: bytes, uppercase: bool = True) -> str:
    h = b.hex()
    if uppercase:
        h = h.upper()
    return h + "\n"

def _split_lines_fixed(data: bytes, line_bytes: int, pad_byte: int = 0x00) -> List[bytes]:
    lines = []
    for i in range(0, len(data), line_bytes):
        chunk = bytearray(data[i:i+line_bytes])
        if len(chunk) < line_bytes:
            chunk.extend([pad_byte] * (line_bytes - len(chunk)))
        lines.append(bytes(chunk))
    if not lines:
        lines.append(bytes([pad_byte] * line_bytes))
    return lines

def save_dict_to_even_odd_hex_files(
    dict_bytes: bytes,
    even_file: str = "dicts_out/dict_even.hex",
    odd_file: str = "dicts_out/dict_odd.hex",
    line_bits: int = 512,
    start_at_one: bool = True,
    pad_byte: int = 0x00,
    uppercase: bool = True,
):
    if line_bits % 8 != 0:
        raise ValueError("line_bits phải chia hết cho 8")
    line_bytes = line_bits // 8
    lines = _split_lines_fixed(dict_bytes, line_bytes, pad_byte=pad_byte)
    evens, odds = [], []
    for idx, line in enumerate(lines, start=1 if start_at_one else 0):
        (evens if idx % 2 == 1 else odds).append(line)
    Path(even_file).parent.mkdir(parents=True, exist_ok=True)
    Path(odd_file).parent.mkdir(parents=True, exist_ok=True)
    with open(even_file, "w", encoding="utf-8") as fo:
        for line in evens:
            fo.write(_bytes_to_hex_line(line, uppercase=uppercase))
    with open(odd_file, "w", encoding="utf-8") as fo:
        for line in odds:
            fo.write(_bytes_to_hex_line(line, uppercase=uppercase))

def save_dict_to_hex_file(
    dict_bytes: bytes,
    out_file: str = "dicts_out/dictionary.hex",
    line_bits: int = 512,
    pad_byte: int = 0x00,
    uppercase: bool = True,
) -> int:
    if line_bits % 8 != 0:
        raise ValueError("line_bits phải chia hết cho 8")
    line_bytes = line_bits // 8
    lines = _split_lines_fixed(dict_bytes, line_bytes, pad_byte=pad_byte)
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fo:
        for line in lines:
            fo.write(_bytes_to_hex_line(line, uppercase=uppercase))
    return len(lines)

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

# ---------------------------
# Dictionary "training" (simple baseline)
# ---------------------------
def train_dictionary_from_bytes(data: bytes, dict_size: int = 4096) -> bytes:
    if not data:
        return bytes([0] * dict_size)
    # 1) Top bytes
    byte_hist = Counter(data)
    top_bytes = [b for b, _ in byte_hist.most_common(256)]
    out = bytearray(top_bytes)
    # 2) Simple substrings harvest
    target = dict_size
    Ls = [8,7,6,5,4,3]
    added = set()
    for L in Ls:
        if len(out) >= target:
            break
        freq = Counter()
        step = max(1, L//2)
        for i in range(0, max(0, len(data)-L+1), step):
            frag = data[i:i+L]
            freq[frag] += 1
        for frag, _ in freq.most_common(1024):
            if len(out) + len(frag) > target:
                break
            if frag not in added:
                out += frag; added.add(frag)
        if len(out) >= target:
            break
    if len(out) < target:
        out.extend([0] * (target - len(out)))
    return bytes(out[:target])

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

# ---------------------------
# CLI Subcommands
# ---------------------------

def cli_train(args):
    data=load_bytes_from_file(args.pth)
    print(f"[train] Loaded {len(data)} bytes from {args.pth}")
    gdict=train_dictionary_from_bytes(data, dict_size=args.dict_size)
    print(f"[train] Dict bytes: {len(gdict)}")
    # ensure out dirs
    Path(args.even).parent.mkdir(parents=True, exist_ok=True)
    Path(args.odd).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.bin).parent.mkdir(parents=True, exist_ok=True)
    save_dict_to_even_odd_hex_files(
        gdict, even_file=args.even, odd_file=args.odd,
        line_bits=args.line_bits, start_at_one=True,
        pad_byte=args.pad_byte, uppercase=not args.lowercase
    )
    total=save_dict_to_hex_file(
        gdict, out_file=args.out, line_bits=args.line_bits,
        pad_byte=args.pad_byte, uppercase=not args.lowercase
    )
    print(f"[train] Wrote {args.out} with {total} lines")
    Path(args.bin).write_bytes(gdict)
    print(f"[train] Wrote {args.bin} ({len(gdict)} bytes)")

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

def build_parser():
    ap=argparse.ArgumentParser(description="Train dictionary & produce LZ77 stats (with combined literal/length frequency)")
    sub=ap.add_subparsers(dest="cmd")

    p_train=sub.add_parser("train", help="Huấn luyện dictionary từ file .pth/.bin")
    p_train.add_argument("--pth", default="data_set/yolov8s_weights.pth", help="File nhị phân đầu vào")
    p_train.add_argument("--dict-size", type=int, default=4096, help="Kích thước dictionary (bytes)")
    p_train.add_argument("--line-bits", type=int, default=512, help="Độ dài dòng .hex (bit)")
    p_train.add_argument("--pad-byte", type=lambda x:int(x,0), default=0x00, help="Padding cho dòng .hex")
    p_train.add_argument("--lowercase", action="store_true", help="Ghi hex thường (mặc định: in hoa)")
    p_train.add_argument("--even", default="dicts_out/dict_even.hex", help="Tên file even")
    p_train.add_argument("--odd", default="dicts_out/dict_odd.hex", help="Tên file odd")
    p_train.add_argument("--out", default="dicts_out/dictionary.hex", help="Tên file dictionary .hex")
    p_train.add_argument("--bin", default="dicts_out/dictionary.bin", help="Tên file dictionary .bin")
    p_train.set_defaults(func=cli_train)

    p_stats=sub.add_parser("stats", help="Sinh thống kê literal/length và combined CSV")
    p_stats.add_argument("--in", dest="inp", default="data_set/yolov8s_weights.pth", help="File nhị phân để nén & thống kê")
    p_stats.add_argument("--dict", dest="dict_path", default="dicts_out/dictionary.hex", help="File dictionary .hex")
    p_stats.add_argument("--outdir", default="stats_out", help="Thư mục output")
    p_stats.add_argument("--min-match", type=int, default=3)
    p_stats.add_argument("--max-match", type=int, default=255)
    p_stats.add_argument("--window-size", type=int, default=65536)
    p_stats.set_defaults(func=cli_stats)

    return ap

def main():
    ap=build_parser()
    import sys, types
    argv=sys.argv[1:]
    # Default: full pipeline when no args
    if not argv:
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
        cli_train(targs)
        sargs = types.SimpleNamespace(
            inp=targs.pth,
            dict_path=targs.out,
            outdir="stats_out",
            min_match=3,
            max_match=255,
            window_size=65536,
        )
        cli_stats(sargs)
        return
    # If args exist: default to "train" when subcommand missing
    if argv[0] not in ("train","stats"):
        argv=["train"]+argv
    args=ap.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()
