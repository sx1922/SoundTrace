"""界面主题：浅色 / 深色两套，配色和圆角集中在这里。

用 QPalette 而不是纯 QSS：下拉框、滚动条、SpinBox 这些原生控件的
内部绘制不完全受 QSS 控制，只有先切到 Fusion 风格并设置调色板，
它们才会跟着变。QSS 负责补上圆角、边框和具体细节。

配色不是纯黑纯白——纯黑背景在长时间盯着看时对比过强，眼睛更累；
这里用略偏灰的深色，和普通编辑器的做法一致。
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

# 圆角尺度：统一到这几个值，不要在别处写魔法数字
RADIUS_SM = 6    # 输入框、下拉框
RADIUS_MD = 8    # 按钮
RADIUS_LG = 12   # 卡片、对话框
RADIUS_PILL = 999  # 圆点、指示器


@dataclass(frozen=True)
class Theme:
    name: str
    bg: str            # 窗口背景
    surface: str       # 控件表面
    surface_alt: str   # 次级表面（输入框、列表项）
    border: str
    text: str
    text_muted: str
    accent: str
    accent_text: str
    danger: str
    success: str
    warning: str
    selection: str
    meter_track: str   # 电平表底槽


LIGHT = Theme(
    name="light",
    bg="#F5F6F8",
    surface="#FFFFFF",
    surface_alt="#ECEEF1",
    border="#D5D8DD",
    text="#1F2328",
    text_muted="#6B7280",
    accent="#3B7DD8",
    accent_text="#FFFFFF",
    danger="#D64541",
    success="#3A9A4E",
    warning="#C08A1E",
    selection="#CFE0F7",
    meter_track="#DFE2E6",
)

DARK = Theme(
    name="dark",
    bg="#1C1E22",
    surface="#25282D",
    surface_alt="#2E3238",
    border="#3A3F46",
    text="#E4E7EC",
    text_muted="#98A0AB",
    accent="#5B93E6",
    accent_text="#10131A",
    danger="#E0574F",
    success="#5FB86B",
    warning="#D6A63C",
    selection="#31465E",
    meter_track="#31353B",
)

THEMES = {"light": LIGHT, "dark": DARK}


def build_palette(t: Theme) -> QPalette:
    p = QPalette()
    p.setColor(QPalette.Window, QColor(t.bg))
    p.setColor(QPalette.WindowText, QColor(t.text))
    p.setColor(QPalette.Base, QColor(t.surface))
    p.setColor(QPalette.AlternateBase, QColor(t.surface_alt))
    p.setColor(QPalette.Text, QColor(t.text))
    p.setColor(QPalette.Button, QColor(t.surface))
    p.setColor(QPalette.ButtonText, QColor(t.text))
    p.setColor(QPalette.Highlight, QColor(t.selection))
    p.setColor(QPalette.HighlightedText, QColor(t.text))
    p.setColor(QPalette.ToolTipBase, QColor(t.surface_alt))
    p.setColor(QPalette.ToolTipText, QColor(t.text))
    p.setColor(QPalette.PlaceholderText, QColor(t.text_muted))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(t.text_muted))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(t.text_muted))
    return p


def stylesheet(t: Theme) -> str:
    """QSS。圆角、边框、焦点态都在这里。"""
    return f"""
QWidget {{
    background: {t.bg};
    color: {t.text};
    font-size: 13px;
}}
QMainWindow, QDialog {{ background: {t.bg}; }}

/* ---------- 输入类 ---------- */
QComboBox, QSpinBox, QLineEdit, QTextEdit, QPlainTextEdit {{
    background: {t.surface};
    border: 1px solid {t.border};
    border-radius: {RADIUS_SM}px;
    padding: 4px 8px;
    selection-background-color: {t.selection};
    selection-color: {t.text};
}}
QComboBox:focus, QSpinBox:focus, QTextEdit:focus, QLineEdit:focus {{
    border: 1px solid {t.accent};
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {t.text_muted};
    width: 0; height: 0;
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background: {t.surface};
    border: 1px solid {t.border};
    border-radius: {RADIUS_SM}px;
    padding: 4px;
    outline: none;
    selection-background-color: {t.selection};
}}
QComboBox QAbstractItemView::item {{
    padding: 5px 8px;
    border-radius: 4px;
    min-height: 22px;
}}
QSpinBox::up-button, QSpinBox::down-button {{ width: 16px; }}

/* ---------- 按钮 ---------- */
QPushButton {{
    background: {t.surface};
    border: 1px solid {t.border};
    border-radius: {RADIUS_MD}px;
    padding: 7px 16px;
}}
QPushButton:hover {{ background: {t.surface_alt}; border-color: {t.text_muted}; }}
QPushButton:pressed {{ background: {t.border}; }}
QPushButton:disabled {{ color: {t.text_muted}; border-color: {t.border}; }}
QPushButton:focus {{ border: 1px solid {t.accent}; }}

/* ---------- 滑块 ---------- */
QSlider::groove:horizontal {{
    height: 4px;
    background: {t.surface_alt};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {t.accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {t.surface};
    border: 1px solid {t.border};
    width: 14px; height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ border-color: {t.accent}; }}

/* ---------- 进度条 ---------- */
QProgressBar {{
    background: {t.surface_alt};
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
}}
QProgressBar::chunk {{ background: {t.accent}; border-radius: 3px; }}

/* ---------- 滚动条 ---------- */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {t.border};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {t.text_muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {t.border};
    border-radius: 5px;
    min-width: 30px;
}}

/* ---------- 分组框 ---------- */
QGroupBox {{
    border: 1px solid {t.border};
    border-radius: {RADIUS_LG}px;
    margin-top: 12px;
    padding: 12px 10px 10px 10px;
    background: {t.surface};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 5px;
    color: {t.text_muted};
}}

/* ---------- 对话框里的按钮 ---------- */
QDialog QPushButton {{ min-width: 76px; }}

/* ---------- 提示气泡 ---------- */
QToolTip {{
    background: {t.surface_alt};
    color: {t.text};
    border: 1px solid {t.border};
    border-radius: {RADIUS_SM}px;
    padding: 5px 7px;
}}
"""


def apply(theme_name: str) -> Theme:
    """把主题应用到整个 QApplication。返回实际生效的 Theme。"""
    t = THEMES.get(theme_name, LIGHT)
    app = QApplication.instance()
    if app is None:
        return t
    # Fusion 才会完全听 QPalette 的，Windows 默认样式会忽略一部分
    app.setStyle("Fusion")
    app.setPalette(build_palette(t))
    app.setStyleSheet(stylesheet(t))
    return t
