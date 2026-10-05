"""幻觉/碎片过滤的测试。

重点不是"能不能拦住幻觉"，而是**不能误杀真实口语**。
旧版本把「订阅」「转发」「打赏」「music」放在子串表里，用户真说
"这个频道需要订阅"会被整句丢掉，而且只在状态栏多一个"过滤幻觉 1"，
用户根本察觉不到自己说的话没了。
"""

from __future__ import annotations

import pytest

from text_norm import is_hallucination, is_suspicious


# -- 必须拦住的真幻觉 -------------------------------------------------------


@pytest.mark.parametrize("text", [
    "谢谢观看",
    "感谢观看",
    "请不吝点赞",
    "字幕由某某整理",
    "字幕志愿者",
    "明镜与点点栏目",
    "订阅",
    "订阅，转发，打赏",        # 整段由幻觉词拼成
    "Thank you for watching",
    "Subtitles by Amara.org community",
    "（字幕：贝尔）",
    "（音乐）",
    "（笑）",          # 整段只有括号内容
    "。。。",           # 纯标点
    "   ",             # 空白
    "",
])
def test_real_hallucination_is_blocked(text):
    assert is_hallucination(text), f"漏掉了幻觉输出：{text!r}"


def test_please_subscribe_survives_as_documented_tradeoff():
    """刻意放行：宁可留一行垃圾，也不吞掉用户真说的话。

    "Please subscribe to my channel" 是 whisper 的经典幻觉，但
    "please subscribe to the newsletter" 完全是正常口述——两者在形状上
    无法区分。旧实现用子串匹配，两种都会删。

    权衡：删错的后果是**静默丢内容**，用户察觉不到自己说的话没了；
    留错的后果是正文里多一行看得见的垃圾。前者更糟，所以宁可放行。
    真正高频的中文幻觉（字幕由、谢谢观看、amara.org）仍然拦得住。
    """
    assert not is_hallucination("Please subscribe to my channel")
    assert not is_hallucination("please subscribe to the newsletter")


# -- 不能误杀的真实口语（本文件存在的主要理由）------------------------------


@pytest.mark.parametrize("text", [
    "这个频道需要订阅",           # 含"订阅"
    "大家记得转发给同事",
    "感谢大家的反馈",
    "谢谢你的帮助",
    "请大家支持一下这个方案",
    "我们不订阅那个会员了",
    "把这个需求打赏给他不合适",
    "我不太喜欢这首歌music很好听",   # 含英文 music
    "the music is playing in the background",
    "Thanks for watching the demo, it was helpful",
])
def test_legitimate_speech_survives(text):
    """这些是用户真可能说的话，删掉就是丢内容。"""
    assert not is_hallucination(text), f"误杀了真实口语：{text!r}"


def test_short_whole_segment_word_is_still_blocked():
    """整段就只有那一个词时仍要拦——这才是噪声幻觉的样子。"""
    assert is_hallucination("订阅")
    assert is_hallucination("订阅。")
    assert is_hallucination("请大家支持")
    assert is_hallucination("music")
    assert is_hallucination("Music")
    assert is_hallucination("谢谢")


def test_long_sentence_with_ambiguous_word_survives():
    """同一个词，长句里就不该删了。"""
    assert not is_hallucination("这个频道需要订阅才能看")
    assert not is_hallucination("请大家支持一下，我们下周继续")


# -- 碎片判定 ---------------------------------------------------------------


def test_suspicious_keeps_short_english_words():
    """语言可切换，只数中日韩会把英文短词全判成碎片。"""
    assert not is_suspicious("OK", 2000)
    assert not is_suspicious("Right", 2000)
    assert not is_suspicious("Sure", 1500)


def test_suspicious_still_blocks_noise_garbage():
    """仍要拦住"录了不短却几乎没出字"的情况。"""
    assert is_suspicious("1", 2000)
    assert is_suspicious("。", 2000)
    assert is_suspicious("?", 3000)
    assert is_suspicious("  ", 2000)


def test_suspicious_tradeoff_is_documented():
    """数了拉丁字母之后，"TR" 这种 1.2 秒残段的垃圾也会被放行。

    这是刻意的取舍：两字母的噪声碎片和用户真说的 "OK" / "No" 在形状上
    无法区分，删错的代价（丢掉用户的话）比漏放一个幻觉大。旧实现只数
    中日韩和数字，等于把英文短句全部判成碎片。
    """
    assert not is_suspicious("TR", 2000)   # 有意放行，见上面的说明


def test_suspicious_ignores_short_audio():
    """音频本来就短，只有一两个字是正常的。"""
    assert not is_suspicious("好", 500)
    assert not is_suspicious("嗯", 700)


def test_suspicious_keeps_normal_sentence():
    assert not is_suspicious("今天下午三点开会", 3000)
