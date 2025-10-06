

def _preview_bytes(label: str, b: bytes, n: int = 10) -> None:
    try:
        shown = " ".join(f"{x:02X}" for x in b[:n])
    except Exception:
        bs = bytes(b)
        shown = " ".join(f"{x:02X}" for x in bs[:n])
    print(f"{label} (first {min(n, len(b))} bytes): {shown}")

def _printable_ascii(b: int) -> str:
    # Return readable single-char; replace non-printables with '.'
    return chr(b) if 32 <= b <= 126 else '.'

def _preview_chars(label: str, b: bytes, n: int = 10) -> None:
    s = ''.join(_printable_ascii(x) for x in b[:n])
    print(f"{label} (first {min(n, len(b))} chars): {s}")

def _preview_tokens(label: str, stream: bytes, n: int = 10) -> None:
    out = []
    i = 0
    count = 0
    while i < len(stream) and count < n:
        t = stream[i]
        i += 1
        if t == 0x00:
            if i >= len(stream):
                out.append("LIT<?>")
                break
            lit = stream[i]
            i += 1
            out.append(f"LIT '{_printable_ascii(lit)}'(0x{lit:02X})")
            count += 1
        elif t == 0x01:
            if i + 2 >= len(stream):
                out.append("MATCH<?>")
                break
            off = (stream[i] << 8) | stream[i+1]
            L = stream[i+2]
            i += 3
            out.append(f"MATCH off={off}, len={L}")
            count += 1
        else:
            # Unknown token type, bail to avoid desync
            out.append(f"?0x{t:02X}")
            break
    print(f"{label} (first {len(out)} symbols): " + ' | '.join(out))

# -*- coding: utf-8 -*-
"""
lz77_with_dict.py
-----------------
Nén/Giải nén LZ77 dùng dictionary tĩnh lấy từ `dicts_out/dictionary.hex`.

✅ TÍNH NĂNG CHÍNH
- Tự động đọc dictionary từ `dicts_out/dictionary.hex`.
- Nếu dictionary chưa tồn tại: báo cần chạy `1_dict_trainer.py` để tạo ra `dicts_out/dictionary.hex`.
- Chế độ mặc định (không truyền tham số): AUTO
  • Đọc tất cả file trong thư mục `lz77_in/`.
  • Với file có đuôi `.lz` → GIẢI NÉN sang `lz77_out/<tên>.dec`.
  • Các file khác → NÉN sang `lz77_out/<tên gốc>.lz`.
  • Tạo báo cáo tổng hợp `lz77_out/report.tsv` (tên, loại tác vụ, kích thước trước/sau).
- CLI vẫn hỗ trợ lệnh thủ công `compress` / `decompress` nếu cần.

📦 ĐỊNH DẠNG TOKEN (giữ nguyên như yêu cầu)
- Literal: [0x00][1 byte literal]
- Match  : [0x01][2 byte offset BE][1 byte length]
  • offset tính từ **cuối dictionary** (chỉ dùng dictionary làm history)
  • 1 = byte ngay trước

🚀 CÁCH DÙNG NHANH
1) Chế độ AUTO (không tham số):
    python lz77_with_dict.py
   - Lần đầu nên: đặt các file cần xử lý vào thư mục `lz77_in/` rồi chạy lại.
   - Kết quả sẽ xuất ở `lz77_out/`.

2) Nén thủ công:
    python lz77_with_dict.py compress <input_path> <output_path.lz>

3) Giải nén thủ công:
    python lz77_with_dict.py decompress <input_path.lz> <output_path>

❗ LƯU Ý
- Bắt buộc phải có `dicts_out/dictionary.hex`. Nếu chưa có, hãy chạy:
    python 1_dict_trainer.py
  để tạo ra file dictionary trước khi nén/giải nén.
"""

from pathlib import Path
from typing import Tuple, Optional, List
import argparse
import sys

# ---------------------------
# Cấu hình đường dẫn mặc định
# ---------------------------
DICT_PATH = Path("dicts_out/dictionary.hex")
INPUT_DIR = Path("inputs")
OUTPUT_DIR = Path("lz77_out")

def _ensure_dirs():
    """Đảm bảo tồn tại thư mục inputs/ và lz77_out/."""
    if not INPUT_DIR.exists():
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Đã tạo thư mục input: {INPUT_DIR.as_posix()}. Hãy đặt file vào đây rồi chạy lại.")
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _pick_any_file(for_mode: str) -> Path:
    """
    Chọn 'bất kỳ' file trong inputs/ phù hợp với chế độ.
    for_mode = 'compress' -> chọn file KHÔNG có đuôi .lz
    for_mode = 'decompress' -> chọn file CÓ đuôi .lz
    Trả về file đầu tiên theo thứ tự alphabet (ổn định). 
    """
    _ensure_dirs()
    all_files = [p for p in INPUT_DIR.iterdir() if p.is_file() and not p.name.startswith(".")]
    if for_mode == "compress":
        candidates = [p for p in all_files if p.suffix.lower() != ".lz"]
    elif for_mode == "decompress":
        candidates = [p for p in all_files if p.suffix.lower() == ".lz"]
    else:
        candidates = all_files

    if not candidates:
        raise FileNotFoundError(f"Không tìm thấy file phù hợp trong '{INPUT_DIR.as_posix()}' cho chế độ {for_mode}.")
    return sorted(candidates)[0]


# ---------------------------
# Utility: đọc dictionary.hex
# ---------------------------

def _parse_hex_line(line: str) -> bytes:
    s = line.strip().replace(" ", "")
    if not s:
        return b""
    if len(s) % 2 != 0:
        raise ValueError("Dòng hex có số ký tự lẻ: " + line)
    return bytes.fromhex(s)

def load_dictionary_hex(path: Path = DICT_PATH) -> bytes:
    """
    Đọc file dictionary.hex (mỗi dòng là chuỗi hex) -> bytes.
    Nếu không tồn tại, raise FileNotFoundError kèm hướng dẫn chạy 1_dict_trainer.py.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy '{path.as_posix()}'. "
            "Hãy chạy '1_dict_trainer.py' trước để tạo 'dicts_out/dictionary.hex'."
        )
    out = bytearray()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out += _parse_hex_line(line)
    return bytes(out)

# ------------------------------------
# Core: LZ77 compress với static dict
# ------------------------------------

def _build_index(history: bytes, start_pos: int, end_pos: int) -> dict:
    """
    Lập chỉ mục 3-byte anchors cho history[start_pos:end_pos].
    """
    idx = {}
    H = history
    if end_pos - start_pos < 3:
        return idx
    for i in range(start_pos, end_pos - 2):
        key = H[i:i+3]
        idx.setdefault(key, []).append(i)
    return idx

def _longest_match(history: bytes, cur_abs_pos: int, lookahead: bytes,
                   window_start: int, min_match: int, max_match: int,
                   index: Optional[dict]) -> Tuple[int, int]:
    """
    Tìm (offset, length) tốt nhất từ history (dictionary tĩnh).
    """
    n = len(lookahead)
    if n < min_match:
        return (0, 0)

    best_len = 0
    best_off = 0

    if n >= 3 and index is not None:
        key = lookahead[:3]
        cand_list: List[int] = index.get(key, [])
        for j in reversed(cand_list):  # ưu tiên offset nhỏ
            if j < window_start:
                continue
            L = 0
            while (
                L < n and (j + L) < cur_abs_pos and L < max_match
                and history[j + L] == lookahead[L]
            ):
                L += 1
            if L >= min_match and L > best_len:
                best_len = L
                best_off = cur_abs_pos - j
                if best_len == max_match:
                    break
    else:
        for j in range(max(window_start, 0), cur_abs_pos - min_match + 1):
            L = 0
            while (
                L < n and (j + L) < cur_abs_pos and L < max_match
                and history[j + L] == lookahead[L]
            ):
                L += 1
            if L >= min_match and L > best_len:
                best_len = L
                best_off = cur_abs_pos - j

    return (best_off, best_len) if best_len >= min_match else (0, 0)

def lz77_compress(data: bytes,
                  dict_bytes: Optional[bytes] = None,
                  window_size: int = 65536,
                  min_match: int = 3,
                  max_match: int = 255) -> bytes:
    """
    Nén 'data' với LZ77, CHỈ tìm match trong dict_bytes (dictionary tĩnh).
    Mặc định tự động đọc dict từ dicts_out/dictionary.hex.
    """
    if dict_bytes is None:
        dict_bytes = load_dictionary_hex()

    out = bytearray()
    history = dict_bytes  # chỉ dùng dictionary làm history
    index = _build_index(history, max(0, len(history) - window_size), len(history))

    i = 0
    while i < len(data):
        lookahead = data[i:i+max_match]
        off, L = _longest_match(
            history, len(history), lookahead,
            max(0, len(history) - window_size),
            min_match, max_match, index
        )
        if L >= min_match:
            # Match token
            out.append(0x01)
            out.append((off >> 8) & 0xFF)
            out.append(off & 0xFF)
            out.append(L & 0xFF)
            i += L
        else:
            # Literal
            out.append(0x00)
            out.append(data[i])
            i += 1

    return bytes(out)

# ------------------------------------
# Giải nén với cùng dictionary
# ------------------------------------

def lz77_decompress(stream: bytes, dict_bytes: Optional[bytes] = None) -> bytes:
    """
    Giải nén stream theo định dạng trên.
    Match chỉ được phép lấy từ dict_bytes (dictionary tĩnh).
    Mặc định tự động đọc dict từ dicts_out/dictionary.hex.
    """
    if dict_bytes is None:
        dict_bytes = load_dictionary_hex()

    out = bytearray()
    i = 0
    n = len(dict_bytes)

    while i < len(stream):
        token = stream[i]
        i += 1
        if token == 0x00:
            if i >= len(stream):
                raise ValueError("Stream lỗi: thiếu byte literal")
            out.append(stream[i])
            i += 1
        elif token == 0x01:
            if i + 2 >= len(stream):
                raise ValueError("Stream lỗi: thiếu tham số match")
            off = (stream[i] << 8) | stream[i+1]
            L = stream[i+2]
            i += 3

            if off == 0:
                raise ValueError("Offset = 0 không hợp lệ")
            src = n - off
            if src < 0 or src + L > n:
                raise ValueError("Offset/length vượt ngoài dictionary")

            out += dict_bytes[src:src+L]
        else:
            raise ValueError(f"Token không hợp lệ: {token:#x}")
    return bytes(out)

# ------------------------------------
# Helpers cho auto mode
# ------------------------------------

def _ensure_dirs():
    if not INPUT_DIR.exists():
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Đã tạo thư mục input: {INPUT_DIR.as_posix()}. Hãy đặt file vào đây rồi chạy lại.")
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _compress_file(in_path: Path, dict_b: bytes) -> Path:
    data = in_path.read_bytes()
    enc = lz77_compress(data, dict_b)
    _preview_chars('Uncompressed input', data, 10)
    _preview_tokens('Compressed output', enc, 10)
    out_path = OUTPUT_DIR / (in_path.name + ".lz")
    out_path.write_bytes(enc)
    return out_path

def _decompress_file(in_path: Path, dict_b: bytes) -> Path:
    stream = in_path.read_bytes()
    dec = lz77_decompress(stream, dict_b)
    out_path = OUTPUT_DIR / (in_path.stem + ".dec")
    out_path.write_bytes(dec)
    return out_path

def run_auto_mode(dict_b: bytes) -> int:
    _ensure_dirs()
    files = [p for p in INPUT_DIR.iterdir() if p.is_file() and not p.name.startswith(".")]
    if not files:
        print(f"Không tìm thấy file nào trong '{INPUT_DIR.as_posix()}'.")
        print("→ Hãy đặt các file nguồn (không phải .lz) vào thư mục này rồi chạy lại.")
        return 0

    # Chỉ chọn file KHÔNG phải .lz để đảm bảo trình tự: Nén -> Giải nén -> So sánh
    src_files = [p for p in files if p.suffix.lower() != ".lz"]
    if not src_files:
        print("Chỉ thấy file .lz trong inputs/. Bỏ qua vì chế độ tuần tự yêu cầu dữ liệu gốc.")
        return 0

    report_lines = ["filename\traw_bytes\tenc_bytes\tdec_bytes\tequal\tenc_path\tdec_path"]
    n_ok = 0
    for p in sorted(src_files):
        try:
            raw = p.read_bytes()
            # 1) Nén
            enc_path = _compress_file(p, dict_b)
            enc = enc_path.read_bytes()
            # 2) Giải nén (từ file vừa nén)
            dec_path = _decompress_file(enc_path, dict_b)
            dec = dec_path.read_bytes()
            # 3) So sánh
            equal = (dec == raw)

            report_lines.append(
                f"{p.name}\t{len(raw)}\t{len(enc)}\t{len(dec)}\t{int(equal)}\t"
                f"{enc_path.as_posix()}\t{dec_path.as_posix()}"
            )
            status = "OK" if equal else "MISMATCH"
            print(f"[{status}] {p.name} | raw={len(raw)}B, enc={len(enc)}B, dec={len(dec)}B")
            if equal:
                n_ok += 1
        except Exception as e:
            print(f"[ERR] {p.name}: {e}")

    (OUTPUT_DIR / "report.tsv").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nTổng kết: {n_ok}/{len(src_files)} file round-trip thành công. Báo cáo: {(OUTPUT_DIR / 'report.tsv').as_posix()}")
    return 0

# ------------------
# CLI
# ------------------
def main():
    parser = argparse.ArgumentParser(
        description="LZ77 với dictionary tĩnh từ dicts_out/dictionary.hex."
                    "Mặc định: AUTO mode xử lý toàn bộ file trong thư mục inputs/ và ghi kết quả vào lz77_out/."
    )
    sub = parser.add_subparsers(dest="cmd")

    p_enc = sub.add_parser("compress", help="Nén 1 file → .lz (dùng '*' để chọn bất kỳ file trong inputs/)")
    p_enc.add_argument("input", help="Đường dẫn file input hoặc '*' để chọn bất kỳ trong inputs/")
    p_enc.add_argument("output", nargs="?", help="Đường dẫn file output (.lz). Nếu input='*' sẽ tự đặt tên.")

    p_dec = sub.add_parser("decompress", help="Giải nén 1 file .lz → file out (dùng '*' để chọn bất kỳ .lz trong inputs/)")
    p_dec.add_argument("input", help="Đường dẫn file .lz hoặc '*' để chọn bất kỳ .lz trong inputs/")
    p_dec.add_argument("output", nargs="?", help="Đường dẫn file output. Nếu input='*' sẽ tự đặt tên.")

    # Không đặt required=True để cho phép mặc định AUTO khi không có subcommand
    args = parser.parse_args()

    # Kiểm tra dictionary
    try:
        dict_b = load_dictionary_hex()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    if args.cmd is None:
        # AUTO MODE
        return run_auto_mode(dict_b)

    if args.cmd == "compress":
        if args.input == "*":
            in_p = _pick_any_file("compress")
            out_p = Path(args.output) if args.output else OUTPUT_DIR / (in_p.name + ".lz")
        else:
            in_p = Path(args.input)
            out_p = Path(args.output) if args.output else OUTPUT_DIR / (in_p.name + ".lz")

        data = in_p.read_bytes()
        enc = lz77_compress(data, dict_b)
        _preview_chars('Uncompressed input', data, 10)
        _preview_tokens('Compressed output', enc, 10)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(enc)
        print(f"Đã nén {in_p.as_posix()} -> {out_p.as_posix()} (raw={len(data)}B, enc={len(enc)}B)")
        return

    if args.cmd == "decompress":
        if args.input == "*":
            in_p = _pick_any_file("decompress")
            out_p = Path(args.output) if args.output else OUTPUT_DIR / (in_p.stem + ".dec")
        else:
            in_p = Path(args.input)
            out_p = Path(args.output) if args.output else OUTPUT_DIR / (in_p.stem + ".dec")

        stream = in_p.read_bytes()
        dec = lz77_decompress(stream, dict_b)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(dec)
        print(f"Đã giải nén {in_p.as_posix()} -> {out_p.as_posix()} (out={len(dec)}B)")
        return

if __name__ == "__main__":
    main()