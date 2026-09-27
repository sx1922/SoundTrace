"""自绘的小控件。

电平表用 QProgressBar 不合适，两个原因：
1. 外观是"进度条"而不是"仪表"，语义不对
2. 更要命的是刻度——麦克风给的线性幅度在安静房间里只有 0.003~0.05，
   线性映射下指针几乎贴着左边，完全看不出"到底有没有在收音"。
   调试这个项目时就被这个坑过：一度以为环境很安静而调错了 VAD 门限。
   所以这里用 dB 刻度（-60 ~ 0 dBFS），安静环境也能看清相对关系。
"""

from __future__ import annotations

import math

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

import theme

DB_FLOOR = -60.0  # 表底；再低没有意义
DB_CEIL = 0.0


def amp_to_db(amp: float) -> float:
    if amp <= 1e-6:
        return DB_FLOOR
    db = 20.0 * math.log10(min(amp, 1.0))
    return max(DB_FLOOR, min(DB_CEIL, db))


def db_to_pos(db: float, width: int) -> float:
    frac = (db - DB_FLOOR) / (DB_CEIL - DB_FLOOR)
    return max(0.0, min(1.0, frac)) * width


class LevelMeter(QWidget):
    """横���电平表，带峰值保持。"""

    def __init__(self, parent: QWidget | None = None, width_hint: int = 170,
                 height_hint: int = 16) -> None:
        super().__init__(parent)
        self.setMinimumSize(width_hint, height_hint)
        self.setMaximumHeight(height_hint)
        self._db = DB_FLOOR
        self._peak_db = DB_FLOOR
        self._active = False
        self._theme = theme.LIGHT

    def set_theme(self, t) -> None:
        self._theme = t
        self.update()

    def set_level(self, amp: float) -> None:
        self._db = amp_to_db(amp)
        if self._db > self._peak_db:
            self._peak_db = self._db
        self.update()

    def reset(self) -> None:
        self._db = self._peak_db = DB_FLOOR
        self.update()

    # 峰值回落。定时器驱动，让"保持"看起来像真的电平表
    def decay(self) -> None:
        if self._peak_db > DB_FLOOR:
            self._peak_db -= 1.5
            if self._peak_db < self._db:
                self._peak_db = self._db
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()

        # 底槽
        p.fillRect(0, 0, w, h, QColor(self._theme.meter_track))

        bar_h = max(4, h - 8)
        y = (h - bar_h) // 2

        # 分段刻度：绿 -> 黄 -> 红
        seg_w = 4
        for x in range(0, w, seg_w):
            frac = x / max(w - 1, 1)
            if frac < 0.62:
                c = QColor(self._theme.success)
            elif frac < 0.85:
                c = QColor(self._theme.warning)
            else:
                c = QColor(self._theme.danger)
            p.fillRect(x, y, seg_w - 1, bar_h, c)

        # 当前电平：从左到右按 dB 位置画深色覆盖
        if self._db > DB_FLOOR:
            end = int(db_to_pos(self._db, w))
            p.fillRect(0, y, end, bar_h, QColor(0, 0, 0, 90))

        # 峰值保持线
        if self._peak_db > DB_FLOOR:
            x = int(db_to_pos(self._peak_db, w))
            p.setPen(QPen(QColor(self._theme.text), 2))
            p.drawLine(x, y - 1, x, y + bar_h + 1)

        # 刻度文字
        p.setPen(QColor(self._theme.text_muted))
        f = p.font()
        f.setPointSize(max(6, f.pointSize() - 3))
        p.setFont(f)
        for db, label in ((-40, "-40"), (-20, "-20")):
            x = int(db_to_pos(db, w))
            p.drawLine(x, h - 6, x, h - 3)
            p.drawText(x + 2, h - 1, label)
        p.end()
