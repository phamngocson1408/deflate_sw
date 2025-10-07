
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_build_huffman_table.py (prefix-prepend fast table)

MÔ TẢ TÍNH NĂNG
- Sinh bảng mã Huffman canonical (LSB-first) cho 286 symbol DEFLATE (0..285), giới hạn độ dài mã tối đa 9 bit và đảm bảo ràng buộc Kraft.
- Tự động gán tần suất = 1 cho các symbol không xuất hiện trong input.
- Xuất 3 file CSV trong thư mục huffman_out/:
  1) huffman_litlen_encode_table.csv
     - Cột: symbol, code_bits_lsb, code_length, extra_bit, base_length
  2) huffman_litlen_decode_table.csv
     - Cột: code_bits_lsb, symbol, code_length, extra_bit, base_length
  3) huffman_litlen_fast_fulltable.csv (fast 9-bit, LSB-first)
     - Được xây dựng DỰA TRÊN decode table theo quy tắc:
       code_bits = prefix_lsb(9 - L) + code_bits_lsb
       (Là phép prepend: L bit CUỐI của code_bits LUÔN là code_bits_lsb.)
     - Đủ 1024 hàng (2^10), sort tăng dần theo code_bits.

CÁCH SỬ DỤNG
1) Chuẩn bị input tần suất:
   - Chạy script 1_dict_trainer.py để sinh file stats_out/combined_freq.csv.
2) Chạy script này:
   - python 2_build_huffman_table.py
3) Kết quả:
   - Xem các file CSV trong thư mục huffman_out/ như mô tả ở trên.

LƯU Ý
- Nếu KHÔNG tồn tại stats_out/combined_freq.csv, script sẽ in thông báo:
  "Hãy chạy script 1_dict_trainer.py trước để tạo file stats_out/combined_freq.csv."
  và dừng với FileNotFoundError.

Quy tắc fast table theo yêu cầu:
    code_bits = prefix_lsb(9 - L) + code_bits_lsb
"""

import os, csv
import pandas as pd
import numpy as np

INPUT_CSV = "/mnt/data/combined_freq.csv" if os.path.exists("/mnt/data/combined_freq.csv") else os.path.join("stats_out","combined_freq.csv")
OUTPUT_DIR = "huffman_out"

ENC_CSV = os.path.join(OUTPUT_DIR, "huffman_litlen_encode_table.csv")      # MSB-first
DEC_CSV = os.path.join(OUTPUT_DIR, "huffman_litlen_decode_table.csv")      # MSB-first
FAST_FULL_CSV = os.path.join(OUTPUT_DIR, "huffman_litlen_fast_fulltable.csv")  # key 9-bit MSB-first

MAX_LEN = 9
NUM_SYMBOLS = 286  # 0..285

def _detect_symbol_and_freq_cols(df: pd.DataFrame):
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    sym_candidates = [c for c in cols if "sym" in c or c in ("symbol","code","id")]
    sym_col = sym_candidates[0] if sym_candidates else cols[0]
    freq_candidates = [c for c in cols if ("freq" in c or "count" in c or c in ("f","p")) and c != sym_col]
    freq_col = freq_candidates[0] if freq_candidates else (cols[1] if len(cols) > 1 else cols[0])
    return sym_col, freq_col

def _to_int(x):
    try: return int(x)
    except: return None

def _to_float(x):
    try: return float(x)
    except: return None

def deflate_length_meta(symbol: int):
    table = {
        257: (0, 3),   258: (0, 4),   259: (0, 5),   260: (0, 6),
        261: (0, 7),   262: (0, 8),   263: (0, 9),   264: (0, 10),
        265: (1, 11),  266: (1, 13),  267: (1, 15),  268: (1, 17),
        269: (2, 19),  270: (2, 23),  271: (2, 27),  272: (2, 31),
        273: (3, 35),  274: (3, 43),  275: (3, 51),  276: (3, 59),
        277: (4, 67),  278: (4, 83),  279: (4, 99),  280: (4, 115),
        281: (5, 131), 282: (5, 163), 283: (5, 195), 284: (5, 227),
        285: (0, 258),
    }
    return table.get(symbol, ("", ""))

def assign_code_lengths(freq_series: pd.Series, max_len: int = MAX_LEN) -> np.ndarray:
    symbols = np.arange(NUM_SYMBOLS)
    freqs = np.asarray([freq_series.get(i, 1.0) for i in symbols], dtype=float)
    order = sorted(range(NUM_SYMBOLS), key=lambda i: (-freqs[i], i))
    lengths = np.full(NUM_SYMBOLS, max_len, dtype=int)
    TOTAL_CAP = 2 ** max_len
    slots_remaining = TOTAL_CAP - NUM_SYMBOLS
    for L in range(1, max_len):
        cost = (2 ** (max_len - L)) - 1
        if cost <= 0 or slots_remaining < cost:
            continue
        candidates = [i for i in order if lengths[i] == max_len]
        if not candidates:
            break
        max_promote = slots_remaining // cost
        num_to_promote = int(min(len(candidates), max_promote))
        if num_to_promote <= 0:
            continue
        for i in candidates[:num_to_promote]:
            lengths[i] = L
            slots_remaining -= cost
            if slots_remaining < cost:
                break
        if slots_remaining <= 0:
            break
    assert lengths.max() <= max_len
    return lengths



def build_rows_encode_msb(lengths: np.ndarray):
    """Canonical code assignment (MSB-first) using RFC 1951 derivation of next_code."""
    max_len = int(lengths.max()) if lengths.size else 0
    bl_count = {l: int(np.sum(lengths == l)) for l in range(0, max_len + 1)}
    bl_count.setdefault(0, 0)
    next_code = {}
    next_code[0] = 0
    if max_len >= 1:
        next_code[1] = 0
    for bits in range(2, max_len + 1):
        prev = next_code.get(bits - 1, 0) + bl_count.get(bits - 1, 0)
        next_code[bits] = prev << 1

    pairs = sorted([(int(lengths[s]), s) for s in range(NUM_SYMBOLS)])
    rows = []
    for L, s in pairs:
        if L <= 0:
            continue
        code_val = next_code[L]
        next_code[L] = code_val + 1
        bits_msb = format(code_val, "b").zfill(L)
        eb, bl = deflate_length_meta(s)
        rows.append({"symbol": s, "code_bits_msb": bits_msb, "code_length": L, "extra_bit": eb, "base_length": bl})
    rows.sort(key=lambda r: r["symbol"])
    return rows



def write_encode_csv(rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(ENC_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["symbol","code_bits_msb","code_length","extra_bit","base_length"])
        writer.writeheader()
        writer.writerows(rows)

def make_decode_rows(rows):
    dec_rows = [{
        "code_bits_msb": r["code_bits_msb"],
        "symbol": r["symbol"],
        "code_length": r["code_length"],
        "extra_bit": r["extra_bit"],
        "base_length": r["base_length"],
    } for r in rows]
    dec_rows.sort(key=lambda r: (int(r["code_length"]), r["code_bits_msb"]))
    return dec_rows

def write_decode_csv(dec_rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(DEC_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["code_bits_msb","symbol","code_length","extra_bit","base_length"])
        writer.writeheader()
        writer.writerows(dec_rows)

def build_fast_full_msb(dec_rows, max_len=9):
    """Xây FAST table 9-bit **MSB-first** bằng cách left-pad code và enumerate suffix."""
    out = []
    for r in dec_rows:
        L = int(r["code_length"])
        code_str = str(r["code_bits_msb"]).zfill(L)
        # integer value MSB-first
        code_val = int(code_str, 2)
        sym = int(r["symbol"])
        eb  = r["extra_bit"]
        bl  = r["base_length"]
        pad = max_len - L
        span = 1 << pad
        for s in range(span):
            bits10_val = (code_val << pad) | s
            bits10 = format(bits10_val, f"0{max_len}b")
            out.append({"key_bits_msb": bits10, "symbol": sym, "code_length": L, "base_length": bl, "extra_bit": eb})
    return out

# ---------------- Distance metadata & generic builders (không sinh code_bits) ----------------
NUM_DIST_SYMBOLS = 30  # 0..29

def deflate_distance_meta(symbol: int):
    base = [1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577]
    extra = [0,0,0,0, 1,1, 2,2, 3,3, 4,4, 5,5, 6,6, 7,7, 8,8, 9,9, 10,10, 11,11, 12,12, 13,13]
    if 0 <= symbol < NUM_DIST_SYMBOLS:
        return extra[symbol], base[symbol]
    return ("","")

def assign_code_lengths_generic(freqs, max_len=10):
    N = len(freqs)
    order = sorted(range(N), key=lambda i: (-float(freqs[i]), i))
    lengths = [max_len] * N
    TOTAL_CAP = 2 ** max_len
    slots_remaining = TOTAL_CAP - N
    for L in range(1, max_len):
        cost = (2 ** (max_len - L)) - 1
        if cost <= 0 or slots_remaining < cost:
            continue
        candidates = [i for i in order if lengths[i] == max_len]
        if not candidates:
            break
        max_promote = slots_remaining // cost
        num_to_promote = int(min(len(candidates), max_promote))
        if num_to_promote <= 0:
            continue
        for i in candidates[:num_to_promote]:
            lengths[i] = L
            slots_remaining -= cost
            if slots_remaining < cost:
                break
        if slots_remaining <= 0:
            break
    return lengths

def build_rows_encode_generic(lengths, meta_fn):
    N = len(lengths)
    max_len = max(lengths) if lengths else 0
    bl_count = {l: sum(1 for L in lengths if L == l) for l in range(1, max_len + 1)}
    first_code, code = {}, 0
    for l in range(1, max_len + 1):
        code <<= 1
        first_code[l] = code
        code += bl_count.get(l, 0)
    next_code = {l: first_code[l] for l in range(1, max_len + 1)}
    pairs = sorted([(lengths[s], s) for s in range(N)])
    rows = []
    for L, s in pairs:
        if L <= 0:
            continue
        msb_code = next_code[L]; next_code[L] += 1
        bits_msb = format(msb_code, "b").zfill(L)
        eb, bb = meta_fn(s)
        rows.append({"symbol": s, "code_bits_msb": bits_msb, "code_length": L, "extra_bit": eb, "base_distance": bb})
    rows.sort(key=lambda r: r["symbol"])
    return rows

DIST_DEC_CSV = os.path.join(OUTPUT_DIR, "huffman_distance_decode_table.csv")

def write_distance_decode_csv(rows):
    """Distance decode table WITHOUT code bits/length; columns: symbol, base_distance, extra_bit."""
    dec_rows = [{
        "symbol": r["symbol"],
        "base_distance": r["base_distance"],
        "extra_bit": r["extra_bit"],
    } for r in rows]
    dec_rows.sort(key=lambda r: int(r["symbol"]))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(DIST_DEC_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["symbol","base_distance","extra_bit"])
        writer.writeheader()
        writer.writerows(dec_rows)

def write_distance_decode_hex(dec_csv_path, out_path):
    """Write 32-bit hex per distance decode entry (sorted by symbol).
    Packing (LSB..MSB), NO SYMBOL FIELD:
      [3:0]   = extra_bit (4 bits)
      [18:4]  = base_distance (15 bits)
      [31:19] = 0
    """
    import pandas as _pd
    df = _pd.read_csv(dec_csv_path).sort_values("symbol")
    def _to_int_safe(v):
        try:
            return int(v)
        except:
            try:
                return int(float(v))
            except:
                return 0
    lines = []
    for r in df.itertuples(index=False):
        base = _to_int_safe(r.base_distance) & 0x7FFF
        xtra = _to_int_safe(r.extra_bit) & 0x0F
        word = (base << 4) | xtra
        lines.append(f"{word:08X}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def write_fast_fulltable_hex(fast_df, out_path):
    """Write 32-bit hex per fast LIT/LEN entry (sorted by key_bits_msb).
    Pack fields (LSB..MSB):
      [4:0]   = extra_bit  (5 bits)
      [13:5]  = base_length(9 bits)
      [17:14] = code_length(4 bits)
      [26:18] = symbol     (9 bits)
      [31:27] = 0
    """
    def _to_int_safe(v):
        if v is None: return 0
        s = str(v).strip()
        if s == "" or s.lower() == "nan": return 0
        try: return int(s)
        except ValueError:
            try: return int(float(s))
            except: return 0
    lines = []
    for r in fast_df.itertuples(index=False):
        sym  = _to_int_safe(r.symbol) & 0x1FF
        clen = _to_int_safe(r.code_length) & 0x0F
        base = _to_int_safe(r.base_length) & 0x1FF
        xtra = _to_int_safe(r.extra_bit) & 0x1F
        word = (sym<<18) | (clen<<14) | (base<<5) | xtra
        lines.append(f"{word:08X}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def main():
    if not os.path.exists(INPUT_CSV):
        msg = (
            f"Không tìm thấy input: {INPUT_CSV}\n"
            "Hãy chạy script 1_dict_trainer.py trước để tạo file stats_out/combined_freq.csv."
        )
        print(msg)
        raise FileNotFoundError(msg)

    df = pd.read_csv(INPUT_CSV)
    sym_col, freq_col = _detect_symbol_and_freq_cols(df)
    df["_symbol"] = df[sym_col].apply(_to_int)
    df = df.dropna(subset=["_symbol"]).copy()
    df["_symbol"] = df["_symbol"].astype(int)
    df = df[(df["_symbol"] >= 0) & (df["_symbol"] <= 285)].copy()
    df["_freq"] = df[freq_col].apply(_to_float).fillna(1.0)
    df.loc[df["_freq"] <= 0, "_freq"] = 1.0

    freq_map = dict(zip(df["_symbol"], df["_freq"]))
    freq_series = pd.Series({i: freq_map.get(i, 1.0) for i in range(NUM_SYMBOLS)})

    lengths = assign_code_lengths(freq_series, MAX_LEN)
    rows = build_rows_encode_msb(lengths)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_encode_csv(rows)

    dec_rows = make_decode_rows(rows)
    write_decode_csv(dec_rows)

    fast_rows = build_fast_full_msb(dec_rows, MAX_LEN)
    fast_df = pd.DataFrame(fast_rows)
    fast_df = fast_df.sort_values(by="key_bits_msb").reset_index(drop=True)
    # Reorder columns
    cols = [c for c in ["key_bits_msb","symbol","code_length","base_length","extra_bit"] if c in fast_df.columns]
    fast_df = fast_df[cols]
    fast_df.to_csv(FAST_FULL_CSV, index=False)

    # Emit HEX for Verilog ROM init (same packing as trước đây)
    FAST_HEX = os.path.join(OUTPUT_DIR, "huffman_litlen_fast_fulltable.hex")
    write_fast_fulltable_hex(fast_df, FAST_HEX)

    # Distance decode only (metadata)
    dist_freqs = [1.0] * 30
    dist_lengths = assign_code_lengths_generic(dist_freqs, MAX_LEN)
    dist_rows = build_rows_encode_generic(dist_lengths, deflate_distance_meta)
    # chỉ xuất meta (base_distance, extra_bit)
    write_distance_decode_csv(dist_rows)
    write_distance_decode_hex(os.path.join(OUTPUT_DIR, "huffman_distance_decode_table.csv"),
                              os.path.join(OUTPUT_DIR, "huffman_distance_decode_table.hex"))

    kraft = sum((2.0 ** (-int(r["code_length"]))) for r in rows)
    maxlen = max(int(r["code_length"]) for r in rows)
    print(f"Xuất: {ENC_CSV}")
    print(f"Xuất: {DEC_CSV}")
    print(f"Xuất: {FAST_FULL_CSV}")
    print(f"Xuất: {FAST_HEX}")
    print(f'Kraft sum = {kraft:.10f} (kỳ vọng = 1.0)')
    print(f"Max code length = {maxlen} (giới hạn = {MAX_LEN})")
    print(f"Tổng symbol = {len(rows)} (kỳ vọng = {NUM_SYMBOLS})")

if __name__ == "__main__":
    main()
