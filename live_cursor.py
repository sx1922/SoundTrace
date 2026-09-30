"""录音时的"活插入点"指示。

口述场景里最容易失手的地方：回看前面改个字，就不知道接下来说的话
会插到哪里了。普通聊天软件不care这个，但听写工具必须知道。

做法：在正文末尾画一条会呼吸的高亮线，并标注"新内容会加在这里"。
用户往回翻时它还留在原位提示位置，滚回底部时自然消失。
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygon
from PySide6.QtWidgets import QTextEdit, QWidget


class LiveInsertionMark(QWidget):
    """正文框上覆盖的一层，负责画插入点。"""

    def __init__(self, editor: QTextEdit):
        super().__init__(editor)
        self._editor = editor
        self._pulse = 0.0
        self._visible = False
        self.setAttribute(Qt.WA_TransparentForMouseEvents)  # 不能挡住选中文本
        self.hide()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(60)

        editor.installEventFilter(self)
        editor.textChanged.connect(self._sync)

    # -- 显示控制 ----------------------------------------------------------

    def set_active(self, active: bool) -> None:
        if active == self._visible:
            return
        self._visible = active
        if active:
            self._pulse = 0.0
            self._timer.start()
            self.show()
            self.raise_()
        else:
            self._timer.stop()
            self.hide()
        self.update()

    def _tick(self) -> None:
        # 0..1 往复，亮度随之变化——不刺眼，但余光能注意到
        self._pulse = (self._pulse + 0.06) % 2.0
        if self._pulse > 1.0:
            self._pulse = 2.0 - self._pulse
        self.update()

    def _sync(self) -> None:
        self.updateGeometry()
        self.update()

    def eventFilter(self, obj, event):  # noqa: N802 - Qt 命名
        if obj is self._editor:
            self.updateGeometry()
            self.update()
        return False

    def resizeEvent(self, event):  # noqa: N802
        self.updateGeometry()

    def updateGeometry(self):  # noqa: N802 - 覆盖 QWidget 方法名
        r = self._editor.rect()
        self.setGeometry(r)

    def cursor_rect(self):
        """插入点在正文框内的坐标。

        取文档末尾的光标矩形。文本不足一屏时末尾就在可视区内，线会正常显示；
        用户翻到上面时线仍留在原位（滚动时 QTextEdit 会跟着重算），
        起到"新内容加在这里"的路标作用。
        """
        try:
            # QTextEdit 没有 cursorRectForPosition，要用 textCursor + cursorRect。
            # 之前写错了 API 又被 except 吞掉，指示线一直不显示却没人发现——
            # 这类错误宁可崩出来。
            cursor = self._editor.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            rect = self._editor.cursorRect(cursor)
            if rect.isNull():
                return None
            scroll = self._editor.contentsRect()
            # 末尾光标在可视区外时（用户翻到了别处），贴着可视区下沿画，
            # 保证路标始终可见，又不会跑进视口之外的空白
            x = min(max(rect.x(), scroll.x() + 6), scroll.right() - 12)
            y = rect.y()
            if y < scroll.y() + 2 or y > scroll.bottom() - 4:
                y = min(max(scroll.center().y(), scroll.y() + 2), scroll.bottom() - 4)
            return (x, y, 2, max(14, self._editor.fontMetrics().height()))
        except Exception:
            import traceback
            traceback.print_exc()
            return None

    # -- 绘制 --------------------------------------------------------------

    def paintEvent(self, event):  # noqa: N802
        if not self._visible:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        rect = self.cursor_rect()
        if rect is None:
            return
        x, y, w, h = rect

        accent = QColor(self.palette().highlight().color())
        alpha = int(140 + 90 * self._pulse)
        accent.setAlpha(alpha)

        # 一条会呼吸的竖线，锚在插入点左侧
        p.setPen(QPen(accent, 2))
        p.drawLine(x, y + 2, x, y + h - 2)

        # 小三角，指示"内容往这边长"
        p.setBrush(accent)
        p.setPen(Qt.NoPen)
        p.drawPolygon(_tri(x + 1, y + h // 2, 6))

        p.end()


def _tri(x: int, y: int, s: int) -> QPolygon:
    return QPolygon([QPoint(x, y - s), QPoint(x + s, y), QPoint(x, y + s)])


class EmptyStateOverlay(QWidget):
    """空状态引导。

    第一次打开是个空框，用户不知道该干什么、也不知道要先选设备还是
    直接开始。给三条最关键的操作提示。
    """

    def __init__(self, editor: QTextEdit, theme):
        super().__init__(editor)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._theme = theme
        self.hide()
        editor.textChanged.connect(lambda: self._sync(editor))

    def set_theme(self, theme) -> None:
        self._theme = theme
        self.update()

    def _sync(self, editor: QTextEdit) -> None:
        empty = not editor.toPlainText().strip()
        self.setVisible(empty)
        self.updateGeometry()
        if empty:
            self.raise_()

    def updateGeometry(self):  # noqa: N802
        self.setGeometry(self.parentWidget().rect())

    def resizeEvent(self, event):  # noqa: N802
        self.updateGeometry()

    def showEvent(self, event):  # noqa: N802
        self.updateGeometry()
        self.raise_()

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setPen(QColor(self._theme.text_muted))
        f = p.font()
        f.setPointSize(f.pointSize() + 1)
        p.setFont(f)
        r = self.rect()
        lines = [
            "点「开始录音」直接说话",
            "也可以按 Ctrl+R",
            "",
            "停顿一会儿就会自动断句，文字会逐段落下来",
        ]
        y = r.center().y() - len(lines) * 9
        for line in lines:
            p.drawText(r.left(), y, line)
            y += 22
        p.end()
