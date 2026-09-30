"""按内存推荐模型的逻辑。

背景：用户要"老电脑也能流畅跑"。实测 small 的推理内存是 base 的 2.4 倍
（701MB vs 287MB），而纯 CPU 上两者都跑得动（RTF 0.58 / 0.16）。所以取舍
只看内存装不装得下，推理速度不是瓶颈。
"""

import pytest

import recommend

HAVE = ["ggml-tiny.bin", "ggml-base.bin", "ggml-small.bin"]


@pytest.fixture(autouse=True)
def fake_mem(monkeypatch):
    monkeypatch.setattr(recommend, "available_mb", lambda: 8000)


def test_picks_small_when_memory_plenty():
    r = recommend.recommend(None, HAVE)
    assert r.model == "ggml-small.bin"
    assert r.tight is False


def test_picks_base_on_low_memory(monkeypatch):
    # small 需 796MB；留 500MB 余量后，可用 1200MB 时预算 700MB 装不下 small，
    # 但装得下 base（382MB）
    monkeypatch.setattr(recommend, "available_mb", lambda: 1200)
    r = recommend.recommend(None, HAVE)
    assert r.model == "ggml-base.bin", "内存紧时应该退到 base"
    assert r.tight is False


def test_degrade_order_is_high_to_low(monkeypatch):
    """降级必须从高精度往低走。反过来会永远停在 tiny。"""
    seen = []
    # 8000 -> small(796MB 够); 1100 -> base(预算 600，够 base 不够 small);
    # 800 -> tiny(预算 300，只够 tiny)
    for mem in (8000, 1100, 800):
        monkeypatch.setattr(recommend, "available_mb", lambda m=mem: m)
        seen.append(recommend.recommend(None, HAVE).model)
    assert seen == ["ggml-small.bin", "ggml-base.bin", "ggml-tiny.bin"], seen


def test_does_not_skip_base(monkeypatch):
    """tiny 总是装得下，中间不能直接跳过去——那会把 base 的精度白白丢掉。"""
    monkeypatch.setattr(recommend, "available_mb", lambda: 1100)
    r = recommend.recommend(None, HAVE)
    assert r.model != "ggml-tiny.bin", "base 装得下就不该选 tiny"


def test_picks_tiny_on_tiny_memory(monkeypatch):
    monkeypatch.setattr(recommend, "available_mb", lambda: 700)
    r = recommend.recommend(None, HAVE)
    assert r.model == "ggml-tiny.bin"


def test_respects_user_choice(monkeypatch):
    """用户明确选了就别自作主张改回来——哪怕内存看起来紧。"""
    monkeypatch.setattr(recommend, "available_mb", lambda: 1200)
    r = recommend.recommend("ggml-small.bin", HAVE)
    assert r.model == "ggml-small.bin"
    assert r.tight is True, "内存紧要标记出来，但不能偷偷换掉用户的选择"


def test_never_picks_medium():
    """medium 体积三倍且精度没有收益，不该被选中。"""
    have = HAVE + ["ggml-medium.bin", "ggml-large-v3-turbo.bin"]
    r = recommend.recommend(None, have)
    assert r.model in ("ggml-tiny.bin", "ggml-base.bin", "ggml-small.bin")


def test_no_models_yet():
    r = recommend.recommend(None, [])
    assert r.model == "ggml-small.bin", "没下载时给出默认建议"
    assert "还没有" in r.reason


def test_only_model_present_is_used(monkeypatch):
    monkeypatch.setattr(recommend, "available_mb", lambda: 8000)
    r = recommend.recommend(None, ["ggml-base.bin"])
    assert r.model == "ggml-base.bin", "只有 base 就用 base"


def test_estimate_includes_ui_overhead():
    r = recommend.recommend("ggml-base.bin", HAVE)
    assert r.estimated_mb > recommend.MODEL_PROFILE["ggml-base.bin"][0]


def test_describe_is_informative():
    for m in HAVE:
        assert "MB" in recommend.describe(m)
