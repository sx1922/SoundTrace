"""项目标识。

名称集中在这里定义，改名时只动这一个文件。中文名「声迹」是注释里
保留的备选：两个字，分别对应"声音"和"痕迹"，和 SoundTrace 是同一个意思。
用中文名的话要注意目录名和 exe 名换成拼音或英文，中文路径在打包
单文件时容易出编码问题。
"""

APP_NAME = "SoundTrace"
APP_NAME_CN = "声迹"
APP_TAGLINE = "本地实时语音转文字"
APP_VERSION = "0.1.1"

# 窗口标题：录音中会在前面加 [● 录音中]
WINDOW_TITLE = f"{APP_NAME} · {APP_NAME_CN}"


def banner() -> str:
    return f"{APP_NAME} {APP_VERSION} — {APP_TAGLINE}"
