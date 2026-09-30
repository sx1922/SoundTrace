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
import os
import re
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


def run_tests() -> bool:
    """打包前跑测试。

    0.1.0 发出去的包是坏的——`_check_assets` 方法被批量编辑整行覆盖，
    启动即崩，而当时没有任何自动化检查拦它。测试不过就不许出包。
    """
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                       cwd=ROOT, env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    return r.returncode == 0


def sync_version() -> None:
    """把 branding.APP_VERSION 同步进 SoundTrace.iss。

    版本号原本在 branding.py 和 SoundTrace.iss 里各写一份，改一边忘另一边
    就会出现"release 叫 0.1.1、附件却叫 0.1.0"这种错位。构建前统一一次。
    """
    sys.path.insert(0, str(ROOT))
    import branding
    ver = branding.APP_VERSION
    iss = ROOT / "SoundTrace.iss"
    src = iss.read_text(encoding="utf-8")
    out = re.sub(r'#define AppVersion "[^"]*"',
                 f'#define AppVersion "{ver}"', src)
    out = re.sub(r'installer/SoundTrace-Setup-[0-9.]+\.exe',
                 f'installer/SoundTrace-Setup-{ver}.exe', out)
    if out != src:
        iss.write_text(out, encoding="utf-8")
        print(f"版本号已同步为 {ver}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", action="store_true", help="只构建 exe")
    ap.add_argument("--clean", action="store_true", help="先删除 dist/ build/")
    ap.add_argument("--skip-tests", action="store_true", help="跳过测试（不建议）")
    args = ap.parse_args()

    if args.clean:
        for d in ("dist", "build"):
            shutil.rmtree(ROOT / d, ignore_errors=True)
            print(f"已清理 {d}/")

    if not args.skip_tests and not run_tests():
        print()
        print("[FAIL] 测试未通过，已中止构建。先把测试修好。")
        return 1

    sync_version()
    check_icon()

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("\n[FAIL] 缺 PyInstaller: pip install pyinstaller")
        return 1

    if not run([sys.executable, "-m", "PyInstaller", "SoundTrace.spec",
                "--noconfirm", "--clean"], "构建 exe"):
        return 1

    # 裁掉没用到的 Qt 模块和对应插件。必须成对删——Qt 启动时枚举加载
    # plugins/，只删 DLL 不删插件会让程序起不来。trim_dist 内部会做
    # 依赖自检，不安全就返回 -1 并跳过。
    sys.path.insert(0, str(ROOT / "tools"))
    from trim_dist import trim
    if trim() < 0:
        print("  裁剪自检未通过，跳过（不影响功能，只是包大一点）")

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
