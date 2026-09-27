"""配置读写。首次运行自动生成 config.json。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import platforms
ROOT = platforms.app_root()
CONFIG_PATH = ROOT / "config.json"


@dataclass
class Config:
    # 模型文件名，放在 models/ 下
    model: str = "ggml-small.bin"
    language: str = "zh"
    # auto / cpu / gpu。auto 会在有 CUDA 构建时用 GPU
    device: str = "auto"

    # 麦克风，-1 表示系统默认输入设备
    input_device: int = -1
    # webrtcvad 灵敏度 0-3
    vad_aggressiveness: int = 2
    # 说话停顿多久算一句话结束（毫秒）。
    # 实测（tests/bench 六条中文样本）这是准确率最大的调节项：
    #   600ms  CER 36.5%（切 25 段）
    #   900ms  CER 32.4%（10 段）
    #  1200ms  CER 32.0%（10 段）← 默认
    #  1800ms+  平台期，再放宽没有收益
    # 段切得越碎，模型每段可用的上下文越少，错字越多。代价是断句后
    # 文本要晚 1.2 秒才定稿；预览仍每 1.2 秒刷新，体感差别不大。
    silence_ms: int = 1200
    # 多久跑一次增量识别（毫秒）
    live_interval_ms: int = 1200

    # 判定为"没有说话"的概率阈值，越大越不容易出幻觉
    no_speech_thold: float = 0.6
    # 低于此长度的语音段直接丢弃（毫秒）
    min_segment_ms: int = 500

    # 把识别结果统一转成简体。whisper 的中文输出会繁简混排
    simplify_chinese: bool = True
    # 规整标点：中文语境下的半角逗号等转全角，引号转中文引号
    fix_punctuation: bool = True

    # 界面
    # light / dark。深色下长时间盯着看比纯白舒服
    theme: str = "dark"
    window_w: int = 1000
    window_h: int = 720

    def model_path(self) -> Path:
        return ROOT / "models" / self.model

    def available_models(self) -> list[Path]:
        return sorted((ROOT / "models").glob("ggml-*.bin"))

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Config":
        cfg = cls()
        if not path.is_file():
            return cfg
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cfg
        known = {f.name for f in fields(cls)}
        for k, v in raw.items():
            if k in known:
                setattr(cfg, k, v)
        return cfg
