"""从 PNG 生成 Windows 需要的多尺寸 .ico。

设计稿通常是 512 或 1024 的 PNG，Windows 图标要包含多个尺寸，
资源管理器、任务栏、标题栏各取不同的尺寸，缺一个就显示成拉伸的糊图。

用法:
    python tools/make_icon.py 图标.png                    # 输出 assets/SoundTrace.ico
    python tools/make_icon.py 图标.png --name MyIcon      # 改输出文件名
    python tools/make_icon.py --check                     # 只校验现有 .ico

注意：图标需要真透明通道。PNG 要是 RGB（没有 alpha），圆角会变成
黑方块——深色背景上不明显，浅色任务栏上就现原形了。
"""

from __future__ import annotations

import argparse
import struct
import sys

# 控制台编码：GitHub Actions 的 Windows runner 是 cp1252，编不了中文，
# 脚本里的中文提示会直接抛 UnicodeEncodeError。本地中文系统是 GBK 不会
# 暴露这个问题，所以必须在这里兜住。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ASSETS = ROOT / "assets"

# Windows 会用到的尺寸。16/24/32 是小图标，256 是资源管理器的大图标
SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
TARGET = 512   # 源图建议至少这么大


def make_ico(png: Path, out: Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("需要 Pillow: pip install pillow")

    im = Image.open(png).convert("RGBA")
    w, h = im.size
    if max(w, h) < 256:
        print(f"提示: 源图只有 {w}x{h}，放大到 256 可能发虚，建议用 >= {TARGET} 的图")

    # 有些导出的小图是 RGBA 但 alpha 全 255（没有真透明），圆角会变黑块
    alpha = im.getchannel("A")
    lo, hi = alpha.getextrema()
    if lo == 255:
        print("警告: 这张 PNG 没有透明通道，圆角会显示成黑色方块。"
              "请导出为带 alpha 的 RGBA PNG。")

    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, format="ICO", sizes=SIZES)

    # 同时留一份 PNG，README 和文档里用得上
    png_out = out.with_suffix(".png")
    im.save(png_out)

    print(f"已生成 {out.name}（{out.stat().st_size / 1024:.0f}KB，"
          f"{len(SIZES)} 个尺寸）")
    print(f"同时保存 {png_out.name}")


def check(ico: Path) -> bool:
    """校验 .ico 是否有效、尺寸是否齐全。"""
    if not ico.is_file():
        print(f"[FAIL] {ico.name} 不存在")
        return False

    data = ico.read_bytes()
    if data[:4] != b"\x00\x00\x01\x00":
        print(f"[FAIL] {ico.name} 不是有效的 ICO（文件头不对）")
        return False

    if len(data) < 6:
        print(f"[FAIL] {ico.name} 只有 {len(data)} 字节，不是完整 ICO")
        return False

    n = struct.unpack("<H", data[4:6])[0]
    # 文件头声明的图标数可能超过实际内容（截断/损坏），
    # 逐项解析时越界会直接崩，得按实际长度收敛
    max_entries = (len(data) - 6) // 16
    if n > max_entries:
        print(f"[FAIL] {ico.name} 头部声明 {n} 个图标，实际只放得下 {max_entries} 个")
        return False

    present = set()
    for i in range(n):
        off = 6 + i * 16
        w = data[off] or 256
        h = data[off + 1] or 256
        present.add((w, h))

    missing = [s for s in SIZES if s not in present]
    print(f"{'[ ok ]' if not missing else '[warn]'} {ico.name}: "
          f"{len(present)} 个尺寸，{ico.stat().st_size / 1024:.0f}KB")
    if missing:
        print(f"        缺少 {', '.join(str(w) for w, _ in missing)}")
        print("        建议重新生成: python tools/make_icon.py <源图.png>")
    return not missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("png", nargs="?", type=Path, help="源图 PNG")
    ap.add_argument("--name", default="SoundTrace", help="输出文件名（不含扩展名）")
    ap.add_argument("--check", action="store_true", help="只校验现有 ico")
    args = ap.parse_args()

    if args.check or not args.png:
        return 0 if check(ASSETS / f"{args.name}.ico") else 1

    if not args.png.is_file():
        raise SystemExit(f"找不到 {args.png}")
    make_ico(args.png, ASSETS / f"{args.name}.ico")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
