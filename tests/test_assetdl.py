"""AssetFetcher 的模型名归一与就绪判断。

DownloadDialog 传的是 cfg.model（文件名 ggml-*.bin），MODELS 表的键却是
短名（small/base/...）。名字不归一的话 has_model 恒 False、_fetch_model
恒抛"未知模型"，程序内下载模型整条路径失效——0.2.0 安装后首跑即在此
报"下载模型失败"。这组用例锁死双向兼容。
"""

from __future__ import annotations

import pytest

import assetdl


@pytest.mark.parametrize("given, expected", [
    ("ggml-medium.bin", "medium"),
    ("ggml-small.bin", "small"),
    ("small", "small"),           # 本来就是短名，原样通过
    ("ggml-large-v3-turbo.bin", "large-v3-turbo"),
])
def test_model_param_accepts_both_forms(tmp_path, given, expected):
    f = assetdl.AssetFetcher(tmp_path, model=given)
    assert f.model == expected


def test_has_model_true_when_file_present(tmp_path):
    """机器上已有模型时必须判定就绪、跳过下载——名字不归一的话
    即使文件就在 models/ 里也会误判缺失。"""
    f = assetdl.AssetFetcher(tmp_path, model="ggml-medium.bin")
    p = tmp_path / "models" / "ggml-medium.bin"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    assert f.has_model() is True


def test_has_model_false_when_missing(tmp_path):
    f = assetdl.AssetFetcher(tmp_path, model="ggml-small.bin")
    assert f.has_model() is False


def test_unknown_model_still_raises(tmp_path):
    f = assetdl.AssetFetcher(tmp_path, model="nonexistent")
    with pytest.raises(RuntimeError, match="未知模型"):
        f._fetch_model()


def test_mirror_constant_registered():
    """镜像回退依赖这个常量，路径结构必须与主站一致。"""
    assert assetdl.HF_MIRROR.startswith("https://hf-mirror.com/")
    assert assetdl.HF_MIRROR.split("/resolve/")[0].endswith("ggerganov/whisper.cpp")