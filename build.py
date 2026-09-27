"""一键构建：exe + 安装包。

    python build.py            # 完整构建
    python build.py --exe      # 只出 exe（快，约 1.5 分钟）
    python build.py --clean    # 先清干净再构建

换图标的流程：
    1. 把新图标放成 assets/SoundTrace.ico（或用 tools/make_icon.py 从 PNG 生成）
    2. python build.py
    3. 装一遍看看任务栏和标题栏的显示效果

构建前会校验图标，避免"删了旧图结果安装包构建失败"这种绕远路的报错。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ISCC_CANDIDATES = [
    Path.home() / "AppData/Local/Programs/Inno Setup 7/ISCC.exe",
    Path("C:/Program Files (x86)/Inno Setup 7/ISCC.exe"),
    Path("C:/Program Files/Inno Setup 7/ISCC.exe"),
]


def human(n: int) -> str:
    return f"{n / 1e6:.1f}MB"


def run(cmd: list[str], title: str) -> bool:
    print(f"\n==> {title}")
    t0 = time.monotonic()
    r = subprocess.run(cmd, cwd=ROOT)
    dt = time.monotonic() - t0
    if r.returncode != 0:
        print(f"[FAIL] {title} 失败（{dt:.0f}s）")
        return False
    print(f"[ ok ] {title}  {dt:.0f}s")
    return True


def find_iscc() -> Path | None:
    for p in ISCC_CANDIDATES:
        if p.is_file():
            return p
    return None


def check_icon() -> bool:
    ico = ROOT / "assets" / "SoundTrace.ico"
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "make_icon.py"), "--check"],
        cwd=ROOT, capture_output=True, text=True)
    out = r.stdout.strip()
    print("\n==> 校验图标")
    print("    " + out.replace("\n", "\n    ") if out else "    (无输出)")
    if r.returncode != 0:
        print("    提示: 不影响出 exe，但安装包构建会失败。"
              "放好 assets/SoundTrace.ico 后重试。")
    return ico.is_file()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", action="store_true", help="只构建 exe")
    ap.add_argument("--clean", action="store_true", help="先删除 dist/ build/")
    args = ap.parse_args()

    if args.clean:
        for d in ("dist", "build"):
            shutil.rmtree(ROOT / d, ignore_errors=True)
            print(f"已清理 {d}/")

    check_icon()

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("\n[FAIL] 缺 PyInstaller: pip install pyinstaller")
        return 1

    if not run([sys.executable, "-m", "PyInstaller", "SoundTrace.spec",
                "--noconfirm", "--clean"], "构建 exe"):
        return 1

    exe = ROOT / "dist" / "SoundTrace" / "SoundTrace.exe"
    if exe.is_file():
        total = sum(p.stat().st_size for p in (ROOT / "dist" / "SoundTrace").rglob("*")
                    if p.is_file())
        print(f"\n[ ok ] exe      {exe}  ({human(exe.stat().st_size)})")
        print(f"        含依赖共 {human(total)}")
    else:
        print(f"\n[FAIL] 没找到产物 {exe}")
        return 1

    if args.exe:
        print("\n完成（--exe：跳过了安装包）")
        return 0

    iscc = find_iscc()
    if not iscc:
        print("\n[warn] 没找到 Inno Setup，跳过安装包。")
        print("       下载 https://jrsoftware.com/isdl.php 装好后重跑。")
        print("       或者用 --exe 只出 exe。")
        return 0

    if not run([str(iscc), "SoundTrace.iss"], "构建安装包"):
        return 1

    setups = sorted((ROOT / "installer").glob("*.exe"))
    if setups:
        p = setups[-1]
        print(f"\n[ ok ] 安装包   {p}  ({human(p.stat().st_size)})")
        print("\n下一步：")
        print("  1. 装一遍看图标在任务栏/标题栏/资源管理器里的显示效果")
        print("  2. 满意后传到 GitHub Releases（浏览器打开 release 页面拖进去）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
