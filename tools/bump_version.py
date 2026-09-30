"""升版：把版本号改到一处，然后打标签。

版本号之前散落在 branding.APP_VERSION 和 SoundTrace.iss 两处，打标签时
很容易只改一边——结果是 release 叫 0.1.2、附件却叫 SoundTrace-Setup-0.1.1.exe。
这个脚本把"改版本"和"打标签"合成一步，顺带把 .iss 同步好。

用法:
    python tools/bump_version.py 0.1.3        # 改版本号并打本地标签
    python tools/bump_version.py 0.1.3 --push # 顺便推到远端
    python tools/bump_version.py --check      # 只看当前版本和是否一致

打完标签后到 GitHub Actions 手动触发构建，tag 填这里输出的版本号。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRANDING = ROOT / "branding.py"
ISS = ROOT / "SoundTrace.iss"
VER_RE = re.compile(r'APP_VERSION\s*=\s*"([^"]+)"')


def current() -> str:
    m = VER_RE.search(BRANDING.read_text(encoding="utf-8"))
    return m.group(1) if m else "?"


def iss_version() -> str:
    m = re.search(r'#define AppVersion "([^"]+)"', ISS.read_text(encoding="utf-8"))
    return m.group(1) if m else "?"


def sync_iss(version: str) -> bool:
    src = ISS.read_text(encoding="utf-8")
    out = re.sub(r'#define AppVersion "[^"]*"', f'#define AppVersion "{version}"', src)
    out = re.sub(r'installer/SoundTrace-Setup-[0-9.]+\.exe',
                 f'installer/SoundTrace-Setup-{version}.exe', out)
    if out != src:
        ISS.write_text(out, encoding="utf-8")
        return True
    return False


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return (r.stdout or r.stderr).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("version", nargs="?", help="新版本号，如 0.1.3")
    ap.add_argument("--push", action="store_true", help="推送改动和标签")
    ap.add_argument("--check", action="store_true", help="只检查一致性")
    args = ap.parse_args()

    cur = current()
    if args.check or not args.version:
        print(f"branding.APP_VERSION = {cur}")
        print(f"SoundTrace.iss       = {iss_version()}")
        ok = cur == iss_version()
        print("一致" if ok else "不一致 —— 跑 python tools/bump_version.py <版本号> 修正")
        return 0 if ok else 1

    if not re.fullmatch(r"\d+\.\d+\.\d+", args.version):
        raise SystemExit(f"版本号格式不对: {args.version}（应该是 x.y.z）")

    BRANDING.write_text(
        VER_RE.sub(f'APP_VERSION = "{args.version}"',
                   BRANDING.read_text(encoding="utf-8")),
        encoding="utf-8")
    changed = sync_iss(args.version)
    print(f"{cur} -> {args.version}")
    print(f"SoundTrace.iss {'已同步' if changed else '本来就是对的'}")

    git("add", "branding.py", "SoundTrace.iss")
    git("commit", "-m", f"版本 {args.version}")
    git("tag", "-f", f"v{args.version}")
    print(f"标签 v{args.version} 已打")

    if args.push:
        git("push", "origin", "main")
        git("push", "origin", f"v{args.version}")
        print("已推送")

    print()
    print("下一步：")
    print(f"  1. GitHub Actions 手动触发，tag 填 v{args.version}")
    print(f"  2. 确认 release {args.version} 上的附件名是 "
          f"SoundTrace-Setup-{args.version}.exe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
