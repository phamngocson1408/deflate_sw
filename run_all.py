
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py — chạy lần lượt 3 script và in kết quả của từng bước (an toàn encoding trên Windows).
"""
import subprocess, sys, os, shlex, locale
from pathlib import Path
from datetime import datetime

SCRIPTS = [
    ("Dictionary Trainer", "1_dict_trainer_with_ini.py"),
    ("Build Huffman Table", "2_build_huffman_table.py"),
    ("Deflate (compress/decompress)", "3_deflate.py"),
]

def banner(title: str):
    line = "=" * 80
    print(f"\n{line}\n[ {title} ] — {datetime.now().isoformat(timespec='seconds')}\n{line}")

def run_step(title: str, script: str) -> int:
    banner(title)
    py = sys.executable
    cmd = [py, script]
    print("[cmd]", " ".join(shlex.quote(x) for x in cmd))

    if not Path(script).exists():
        print(f"[error] Không tìm thấy file: {script}")
        return 127

    # Đảm bảo IO của child là UTF-8 để tránh UnicodeDecodeError trên Windows cp1252
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")              # Python 3.7+ bật UTF-8 mode nếu có thể
    env.setdefault("PYTHONIOENCODING", "utf-8")    # ép stdout/stderr UTF-8

    # text=True dùng encoding mặc định (thường cp1252 trên Windows) -> chỉ định encoding='utf-8'
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",  # không vỡ stream nếu có byte lạ
        bufsize=1,
        env=env,
    )

    # Stream từng dòng
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    print(f"[exit] {script} -> returncode = {proc.returncode}")
    return proc.returncode

def main():
    cwd = Path.cwd()
    os.chdir(Path(__file__).resolve().parent)
    rc_all = 0
    for (title, script) in SCRIPTS:
        rc = run_step(title, script)
        if rc != 0:
            rc_all = rc
            print(f"[stop] Dừng pipeline do bước '{title}' lỗi (rc={rc}).")
            break
    os.chdir(cwd)
    sys.exit(rc_all)

if __name__ == "__main__":
    main()
