"""解析 PE 文件的导入表，列出它依赖的 DLL。

只做静态分析，不加载也不执行任何东西——既能查清 Qt 到底还需要哪些
DLL，又不会在开发机上造成负担。

用法:
    python tools/pe_deps.py <dll路径>            看单个文件依赖谁
    python tools/pe_deps.py --dir dist/.../PySide6 --scan  扫整个目录
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path


def _read_headers(data: bytes):
    if data[:2] != b"MZ":
        raise ValueError("不是 PE 文件")
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    if data[e_lfanew : e_lfanew + 4] != b"PE\0\0":
        raise ValueError("PE 头无效")
    coff = e_lfanew + 4
    n_sections = struct.unpack_from("<H", data, coff + 2)[0]
    opt_size = struct.unpack_from("<H", data, coff + 16)[0]
    opt = coff + 20
    magic = struct.unpack_from("<H", data, opt)[0]
    pe32plus = magic == 0x20B

    dd_off = opt + (112 if pe32plus else 96)
    imp_rva, imp_size = struct.unpack_from("<II", data, dd_off + 8)

    sec_off = opt + opt_size
    sections = []
    for i in range(n_sections):
        s = sec_off + i * 40
        name = data[s : s + 8].rstrip(b"\0").decode("ascii", "replace")
        vsize, vaddr, rsize, raddr = struct.unpack_from("<IIII", data, s + 8)
        sections.append((name, vaddr, vsize, raddr, rsize))
    return sections, imp_rva, imp_size, pe32plus


def _rva_to_off(rva: int, sections) -> int | None:
    for _name, vaddr, vsize, raddr, rsize in sections:
        if vaddr <= rva < vaddr + max(vsize, rsize):
            return raddr + (rva - vaddr)
    return None


def imports(path: Path) -> list[str]:
    """返回该 PE 导入的 DLL 名（去重、保持顺序）。"""
    data = path.read_bytes()
    sections, imp_rva, _imp_size, pe32plus = _read_headers(data)
    if not imp_rva:
        return []
    off = _rva_to_off(imp_rva, sections)
    if off is None:
        return []

    names: list[str] = []
    # 导入描述符固定 20 字节，PE32 和 PE32+ 都一样。
    # （之前按 PE32+ 写成 8 字节，导致插件的导入表解析到一半就越界。）
    step = 20
    i = off
    while True:
        chunk = data[i : i + step]
        if len(chunk) < step:
            break
        # OriginalFirstThunk(4) TimeDateStamp(4) ForwarderChain(4)
        # Name RVA(4) FirstThunk(4)
        name_rva = struct.unpack_from("<I", chunk, 12)[0]
        if name_rva == 0:
            break
        n_off = _rva_to_off(name_rva, sections)
        if n_off is not None:
            end = data.find(b"\0", n_off, n_off + 260)
            if end > 0:
                name = data[n_off:end].decode("ascii", "replace")
                if name not in names:
                    names.append(name)
        i += step
    return names


def scan(dir_path: Path, only_local: bool = True) -> None:
    """扫目录，报告每个 PE 依赖了哪些不在目录里的 Qt6* / PySide 相关 DLL。"""
    dir_path = dir_path.resolve()
    present = {p.name.lower() for p in dir_path.glob("*.dll")}
    for f in sorted(dir_path.rglob("*.dll")):
        try:
            deps = imports(f)
        except Exception:
            continue
        missing = []
        for d in deps:
            dl = d.lower()
            if not (dl.startswith("qt6") or dl.startswith("pyside")):
                continue
            if only_local and dl not in present:
                missing.append(d)
        if missing:
            rel = f.relative_to(dir_path)
            print(f"{str(rel):<44} -> {', '.join(sorted(missing))}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--dir":
        root = Path(sys.argv[2])
        scan(root / "PySide6" if (root / "PySide6").is_dir() else root)
    else:
        for p in sys.argv[1:]:
            print(f"{Path(p).name}: {', '.join(imports(Path(p)))}")
