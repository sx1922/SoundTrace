"""platforms 路径解析的回归测试。

4b33c58 的 _as_vendor_root 只认 CPU 目录（vendor/whisper/Release）的形态，
CUDA 目录（vendor/whisper_cuda/Release）原样通过后被再拼一层 whisper/Release，
装了 CUDA 构建、device=auto 的机器模型加载必然失败。这里的用例用 tmp_path
构造两种目录形态，不依赖真模型，CI 上也能跑。
"""

from __future__ import annotations

from pathlib import Path

import pytest

import platforms


# -- _split_vendor：历史上一共传过这些形态 -----------------------------------


def test_split_vendor_accepts_all_historical_forms(tmp_path):
    vendor = tmp_path / "vendor"
    cases = [
        (vendor, (vendor, "whisper")),
        (vendor / "whisper", (vendor, "whisper")),
        (vendor / "whisper" / "Release", (vendor, "whisper")),
        (vendor / "whisper_cuda" / "Release", (vendor, "whisper_cuda")),
        (vendor / "whisper_cuda", (vendor, "whisper_cuda")),
    ]
    for given, expected in cases:
        assert platforms._split_vendor(given) == expected, f"输入 {given}"


# -- get_library_path（Windows 文件名形态）-----------------------------------


@pytest.mark.skipif(platforms.current().key != "windows",
                    reason="文件名拼法是 Windows 的 .dll 形态")
def test_get_library_path_resolves_cpu_dir(tmp_path):
    release = tmp_path / "vendor" / "whisper" / "Release"
    release.mkdir(parents=True)
    assert platforms.get_library_path(release, "whisper") == release / "whisper.dll"


@pytest.mark.skipif(platforms.current().key != "windows",
                    reason="文件名拼法是 Windows 的 .dll 形态")
def test_get_library_path_cuda_dir_not_double_joined(tmp_path):
    """GPU 回归的直接回归测试：CUDA 目录不得被拼上 whisper/Release。"""
    release = tmp_path / "vendor" / "whisper_cuda" / "Release"
    release.mkdir(parents=True)
    (release / "whisper.dll").write_bytes(b"x")

    p = platforms.get_library_path(release, "whisper")
    assert p == release / "whisper.dll"
    assert p.is_file(), f"解析出的库必须真实存在，实际 {p}"


@pytest.mark.skipif(platforms.current().key != "windows",
                    reason="文件名拼法是 Windows 的 .dll 形态")
def test_get_library_path_resolves_backends_under_cuda(tmp_path):
    """_load_backends 对 ggml / ggml-cuda 的解析同样走族推断。"""
    release = tmp_path / "vendor" / "whisper_cuda" / "Release"
    release.mkdir(parents=True)
    assert platforms.get_library_path(release, "ggml") == release / "ggml.dll"
    assert platforms.get_library_path(release, "ggml-cuda") == release / "ggml-cuda.dll"


# -- is_runtime_present：按族检查，CUDA 族曾永远 False ------------------------


def test_is_runtime_present_per_family(tmp_path):
    root = tmp_path / "vendor"
    cpu = root / "whisper" / "Release"
    cuda = root / "whisper_cuda" / "Release"
    cpu.mkdir(parents=True)
    cuda.mkdir(parents=True)

    plat = platforms.current()
    main = plat.whisper_path_name
    # macOS 的判定走 xcframework，平铺文件不算数
    expect = plat.key in ("windows", "linux")

    assert not platforms.is_runtime_present(root), "两族都没装时不该报就绪"
    assert not platforms.is_runtime_present(cuda), "CUDA 空目录不该报就绪"

    (cpu / main).write_bytes(b"x")
    assert platforms.is_runtime_present(root) is expect

    (cuda / main).write_bytes(b"x")
    # 传 CUDA 目录要推断出 whisper_cuda 族——回归点是这里曾被检查成
    # vendor/whisper_cuda/whisper/Release/...，永远 False，导致反复重下 700MB
    assert platforms.is_runtime_present(cuda) is expect


# -- 应用级链路：device.detect 给出的 vendor_dir 必须解析到真实库 --------------


def test_cpu_vendor_dir_from_detect_resolves_to_real_library():
    """事故复现场景：detect() 产出的 vendor_dir 经 get_library_path
    解析后文件必须存在。本机没装对应运行时就跳过。"""
    import device

    info = device.detect("cpu")
    if not info.vendor_dir.is_dir():
        pytest.skip(f"本机未装 CPU 运行时: {info.vendor_dir}")
    p = platforms.get_library_path(info.vendor_dir, "whisper")
    assert p is not None and p.is_file(), (
        f"detect 给的 vendor_dir 解析不到库: {p}")


def test_cuda_vendor_dir_from_detect_resolves_when_installed():
    """GPU 回归的端到端断言：detect('auto') 在 CUDA 机器上返回
    vendor/whisper_cuda/Release，解析结果必须指向真实存在的 whisper.dll。"""
    import device

    info = device.detect("auto")
    if not (info.use_gpu and info.vendor_dir.is_dir()):
        pytest.skip("本机未启用 CUDA 构建")
    p = platforms.get_library_path(info.vendor_dir, "whisper")
    assert p is not None and p.is_file(), (
        f"CUDA vendor_dir 解析不到库: {p}（GPU 路径回归）")
