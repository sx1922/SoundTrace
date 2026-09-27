"""中文文本后处理。

whisper 的中文输出有两个系统性问题，直接影响可读性：

1. **繁简混排**。同一个模型、同一个人说的话，不同段落会时简时繁
   （实测同一段音频里"讨论一下"是简体，"我們開個短會"是繁体）。
   原因是模型按段采样，落在不同 token 上。混排看着很乱。
2. **标点不统一**。中文该用全角，实际输出里半角逗号 `,` 和
   缺句末标点的情况都出现过。

这里统一转简体并规整标点。opencc 是纯 Python 实现，不需要编译；
万一没装，这里会退化成"只做标点规整"，不会让整个程序起不来。
"""

from __future__ import annotations

import re

# 半角 -> 全角。逐字符映射，注意引号要成对处理。
_HALF_TO_FULL = {
    ",": "，", ";": "；", ":": "：", "?": "？", "!": "！",
    "(": "（", ")": "）",
}

# 出现在中文语境里、但模型偏好的半角符号
_HALF_RE = re.compile("[" + re.escape("".join(_HALF_TO_FULL)) + "]")

# 这些符号不管前面是什么，统一转全角引号
_OPEN_QUOTES = '"“‘「『'
_CLOSE_QUOTES = '"”’」』'


class TextNormalizer:
    def __init__(self, simplify: bool = True, fix_punctuation: bool = True):
        self.simplify = simplify
        self.fix_punctuation = fix_punctuation
        self._cc = None
        if simplify:
            try:
                from opencc import OpenCC

                self._cc = OpenCC("t2s")
            except Exception:
                # 没装 opencc 就退化成只做标点处理，不影响主流程
                self._cc = None

    @property
    def simplify_available(self) -> bool:
        return self._cc is not None

    def __call__(self, text: str) -> str:
        return self.normalize(text)

    def normalize(self, text: str) -> str:
        if not text:
            return text

        if self._cc is not None:
            text = self._cc.convert(text)

        if self.fix_punctuation:
            text = self._fix_punct(text)

        return text.strip()

    def _fix_punct(self, text: str) -> str:
        if not _has_cjk(text):
            # 纯英文/数字段落保持原样，否则 'He said "hi", ok.' 会被
            # 错误地转成中文引号和全角逗号
            return text

        # 只在"相邻字符是中文"时替换，英文句子里的逗号保持半角。
        # 否则 "a, b" 会被错误地变成 "a，b"。
        out = []
        for i, ch in enumerate(text):
            if ch in _HALF_TO_FULL and self._near_cjk(text, i):
                out.append(_HALF_TO_FULL[ch])
            else:
                out.append(ch)
        return self._fix_quotes("".join(out))

    @staticmethod
    def _near_cjk(text: str, i: int) -> bool:
        """判断该不该按中文标点处理：任一侧紧邻中文字符即可。"""
        for j in (i - 1, i + 1):
            if 0 <= j < len(text):
                ch = text[j]
                if "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿":
                    return True
        return False

    @staticmethod
    def _fix_quotes(text: str) -> str:
        """成对地把直引号换成中文引号。"""
        out = []
        open_next = True
        for ch in text:
            if ch == '"':
                out.append("“" if open_next else "”")
                open_next = not open_next
            else:
                out.append(ch)
        return "".join(out)


def _has_cjk(text: str) -> bool:
    return any("㐀" <= c <= "鿿" for c in text)


# ---------------------------------------------------------------- 幻觉过滤
#
# 为什么要单独做一层：实测在安静房间录 12 秒，VAD 切出的两段都是噪声，
# whisper 对它们输出了 "(字幕:貝爾)"，而且 no_speech_prob = 0.000 ——
# 它非常确信那是人话。把 no_speech_thold 从 0.6 调到 0.3 毫无变化。
#
# 也就是说 whisper 自带的"无语音"判定在这种噪声上完全失效，
# 幻觉短语识别是唯一有效的防线。这层必须在繁简转换之后做，
# 否则匹配不到 "(字幕:貝爾)" 这种繁体输出。

# 中文幻觉短语，按原文匹配
_HALLUCINATION_CJK = (
    "谢谢观看", "感谢观看", "请不吝点赞", "订阅", "转发", "打赏", "明镜与点点栏目",
    "字幕由", "字幕志愿者", "请加入", "未经许可", "不得转载", "请大家支持",
)

# 英文幻觉短语，按小写后的文本匹配
_HALLUCINATION_EN = (
    "thank you for watching", "thanks for watching", "please subscribe",
    "thank you very much", "music", "applause", "clapping", "subtitles by",
    "amara.org", "bye.", "for more information,", "www.",
)

# 幻觉的典型外形：整段被括号包起来，或者带"字幕:""音乐:"这类标签
_HALLUCINATION_RES = (
    re.compile(r"[（(]\s*字幕\s*[:：]"),
    re.compile(r"[（(]\s*(音乐|音樂|歌曲|歌聲|歌声)\s*[)）]?"),
    re.compile(r"^\s*[（(][^）)]{0,24}[）)]\s*$"),
    re.compile(r"^[\s\W]*$"),   # 纯标点
)


def is_hallucination(text: str) -> bool:
    """识别 whisper 在非语音上编出来的内容。

    Args:
        text: 已完成繁简统一的识别结果。
    """
    t = text.strip()
    if not t:
        return True
    for rx in _HALLUCINATION_RES:
        if rx.search(t):
            return True
    lowered = t.lower()
    if any(p in lowered for p in _HALLUCINATION_EN):
        return True
    return any(p in t for p in _HALLUCINATION_CJK)


def is_suspicious(text: str, duration_ms: int = 0) -> bool:
    """判断一段文本是否像是模型在噪声上瞎猜的碎片。

    关键是要结合音频时长：单看字数会误伤真实短句（用户说"好"就只有
    一个字）。真正的特征是"录了不短，却几乎没出字"——whisper 遇到被
    截断的残段时会吐出这类垃圾（实测 1.2 秒的残段输出过单个 "TR"）。

    Args:
        text: 识别结果。
        duration_ms: 这段音频的时长。
    """
    t = text.strip()
    if not t:
        return True
    # 短音频本来就可能只有一两个字，不做判断
    if duration_ms < 800:
        return False
    cjk = sum(1 for c in t if "㐀" <= c <= "鿿")
    digits = sum(1 for c in t if c.isdigit())
    # 录了将近一秒以上，却只有零星几个字/数字，基本就是碎片
    return cjk + digits < 2 and len(t) < 6
