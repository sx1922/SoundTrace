# SoundTrace · 声迹

对着麦克风说话，边说边出字。识别全程在本地跑，**不联网、不上传音频**。

底层是 [whisper.cpp](https://github.com/ggml-org/whisper.cpp)，通过 ctypes 直接调用其动态库，
**不需要装任何编译器**。

---
## 安装

**Windows**：下载 `SoundTrace-Setup-0.1.0.exe` 运行安装（装到用户目录，不需要管理员权限）。
首次启动会弹窗下载 whisper.cpp 运行时和模型（约 500MB，只需一次），之后完全离线。

**macOS**：`git clone` 后双击 `run.command`，或终端里 `./run.command`。

**Linux**：`./run.sh` 会先跑一遍环境自检再启动。

> 跨平台状态：**Windows 完整验证过**（含打包与安装流程）。
> macOS / Linux 的代码路径按 whisper.cpp 官方 release 的资产命名写的，
> 但没有实机验证过。第一次在新系统上跑，请先执行 `python tools/selfcheck.py`，
> 它会逐项告诉你缺什么、怎么补。

## 从源码运行

```bash
git clone https://github.com/sx1922/SoundTrace.git
cd SoundTrace

# Windows
run.bat
# macOS
./run.command
# Linux
./run.sh
```

手动执行：

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python tools/fetch_assets.py        # 下载 whisper.cpp 运行时 + 模型
python app.py
```

## 打包

```bash
pip install pyinstaller
pyinstaller SoundTrace.spec                       # -> dist/SoundTrace/
iscc SoundTrace.iss                               # -> installer/*.exe
```

安装包只含程序本体（约 160MB，其中大头是 Qt），模型不进包，改为首次运行下载。

## 快速开始

```bat
run.bat
```

首次运行会自动建虚拟环境、装依赖、下载模型（约 500MB），之后启动就快了。

手动执行的话：

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
python tools\fetch_assets.py        # 下载 whisper.cpp 运行时 + 模型
python app.py
```

> **不要手工编辑 `run.bat`。** 它必须是纯 ASCII + CRLF 换行。
> `cmd.exe` 按系统 OEM 代码页（简体中文机器是 GBK）读取批处理，
> 文件里出现非 ASCII 字符或 LF 换行会导致解析错乱——表现为双击后窗口
> 闪一下就没了，或者冒出 `'xx' 不是内部或外部命令` 这类莫名其妙的信息。
> 改的时候用 `open('run.bat','wb')` 写 `.encode('ascii')` 的内容。
> 中文提示信息请放在 Python 那边，`.py` 文件不受影响。

---

## 界面说明

| 位置 | 作用 |
|---|---|
| 麦克风下拉框 | 选输入设备，录音时锁定不可改 |
| 语言 | 中文 / English / 自动检测 |
| 灵敏度 0–3 | 越往右越严格，噪声越不容易被当成说话。环境嘈杂就调右 |
| 模型 | 切换后立即重载，几秒钟。录音中不可改 |
| 设置… | 断句、预览间隔、繁简标点、**主题（深色/浅色）** |
| 开始录音 | `Ctrl+R` 也可以 |
| 上方文本框 | **已确认正文**，断句后定稿，不会再变。**可以直接改错字**，改动会进导出的字幕 |
| 灰色斜体行 | **实时预览**，临时结果，会被不断覆盖 |
| 电平表 | 输入电平，dBFS 刻度（-60~0），红线是峰值保持 |
| ● 状态 | "静音 / 说话中 / 停顿中"，和电平表一起看更容易判断 |
| 速度 / 计时 | `1.0x` 表示比实时快一倍；右边是本次录音时长 |

`Ctrl+Shift+C` 复制全文，`Ctrl+S` 导出 txt 或 srt（带时间轴的字幕）。

> `Ctrl+C` 保留给文本框的原生行为，用来复制选中的片段——
> 正文框是可编辑的，绑走标准复制键会没法正常改稿。

参数（灵敏度、语言、模型）只在**下次录音**时生效，改动后状态栏会提示。
录音中这些控件是锁定的，避免产生"改了以为有用"的错觉。

---

## 它是怎么工作的

```
麦克风 → 环形缓冲 → VAD 分段 → whisper 识别 → 繁简/标点规整 → 定稿/预览
```

五件事值得知道：

**为什么要 VAD。** Whisper 对纯静音会"脑补"出内容，经典输出是 "Thank you for watching!"。
不过 VAD 的话，房间里一点背景音就会让这些假文字不断冒进正文。

**幻觉有三层防线，但只有一层真正管用。** 实测数据（安静房间录 12 秒，
VAD 切出 2 段，两段都是噪声）：

- `webrtcvad` 判定为语音，把噪声放行了
- **`no_speech_prob` 返回 0.000** —— whisper 非常确信那是人话，
  把 `no_speech_thold` 从 0.6 调到 0.3 毫无变化
- whisper 的实际输出是 `(字幕:貝爾)`

也就是说 whisper 自带的"无语音"判定在这种噪声上完全失效，**幻觉短语识别
是唯一有效的防线**。这层刻意放在繁简转换之后执行，否则匹配不到 `貝爾`
这种繁体输出。过滤后房间噪声 0 段进入正文（拦下 4 次），
真实语音 5 段全部正确、零误拦。

**断句阈值是准确率最大的调节项。** Whisper 对短片段的上下文不足，
段切得越碎错字越多。用 `tests/bench` 的六条中文样本实测（字符错误率）：

| 停顿阈值 | 切段数 | 字符错误率 |
|---|---|---|
| 600 ms | 25 | 36.5% |
| 900 ms | 10 | 32.4% |
| **1200 ms（默认）** | 10 | **32.0%** |
| 1800 ms | 7 | 32.4% |
| 3000 ms | 7 | 32.4% |
| 完全不切 | 6 | 32.0% |

900–1200ms 是甜点，之后进入平台期。**完全没有收益是不切段**——所以
分段必须保留（它是幻觉防线），只把阈值调宽。副作用是断句后文本晚
1.2 秒才定稿，但预览仍是每 1.2 秒刷新，体感差别不大。

同样的原因，段变长之后模型自己会输出正确的中文标点，之前的碎片式输出
反而是标点全无。

**灵敏度滑块的实际影响。** 用房间噪声和中文语音各测一遍：

| 灵敏度 | 噪声误判占比 | 语音覆盖率 |
|---|---|---|
| 0 | 44.5% | 97.2% |
| 1 | 44.5% | 97.2% |
| 2（默认） | 39.8% | 97.2% |
| 3 | 29.2% | 97.2% |

四档对语音检出率**没有任何影响**，噪声误判差异明显。环境嘈杂可以往右调。

**预览和正文为什么分开。** Whisper 没有流式接口，只能整段识别。这里的做法是
"假流式"：VAD 切出一段后送识别，同时每 1.2 秒对当前这段重算一次并覆盖显示（预览），
等这段说完再算一次定稿（正文）。预览可以错，正文必须准。

**实时性能的关键在调用次数，不在音频长度。** Whisper 的编码器恒定按 30 秒的
mel 窗口计算，你喂 2 秒还是 30 秒成本几乎一样。所以短句多 = 调用多 = 慢。
工具会自己判断：如果上一次识别耗时已经接近预览间隔，就自动放弃预览，优先保证正文出字。

**中文输出做了后处理。** Whisper 的中文有两个系统性问题，不处理很难看：

- **繁简混排**。同一个模型、同一个人说话，不同段落会时简时繁（实测同一段音频里
  "讨论一下"是简体，"我們開個短會"是繁体），因为模型按段采样落在不同 token 上。
  默认用 opencc 统一转简体。
- **标点不统一**。中文该用全角，实际输出里有半角逗号 `,` 和缺句末标点的情况。
  只在中文语境下替换，英文句子的 `He said "hi", ok.` 保持原样。

两个都能在 `config.json` 里关掉（`simplify_chinese` / `fix_punctuation`）。

**碎片过滤。** Whisper 遇到被截断的残段会吐出极短的垃圾（实测 1.2 秒的残段输出过
单个 "TR"）。判断依据是"录了将近一秒却几乎没出字"，而不是单纯看字数——否则用户
说一句"好"就会被误删。

---

## 性能

实测环境：AMD Ryzen 5 5500（12 逻辑核）+ RTX 4060 8GB，whisper.cpp b5130，ggml-small 模型。
下表是 11 秒英文音频的实测结果：

| 设备 | 识别耗时 | 实时率 | 体验 |
|---|---|---|---|
| CPU（11 线程） | 3.81 秒 | 0.35（约 3 倍实时） | 能用，但预览经常被抑制 |
| GPU（RTX 4060） | 0.17 秒 | 0.015（约 67 倍实时） | 流畅，预览稳定刷新 |

差了 22 倍。**强烈建议用 GPU**——CPU 上短片段的实时率会退化到 1.1 以上
（2 秒的片段要花 2.3 秒），预览基本失效，只剩断句后才出字。

装法见下面 GPU 章节。只想快速试试、不想下 700MB，那 CPU 也够用，
把 `config.json` 的 `model` 换成 `ggml-base.bin` 会明显变快。

### 换模型

同一套测试集（断句阈值 1200ms，GPU）实测：

| 模型 | 体积 | 字符错误率 | 错字数 |
|---|---|---|---|
| base | 142 MB | 35.62% | 78 / 219 |
| **small（默认）** | 487 MB | **31.96%** | 70 / 219 |
| medium | 1.5 GB | 31.96% | 70 / 219 |

**medium 不值得换**：体积是 small 的三倍、推理更慢，错误率却一模一样。
base → small 有约 11% 的相对改善，所以 small 确实是甜点。
（这是合成语音上的结果，真实带口音、有背景音的语料可能不同。）

```bat
python tools\fetch_assets.py --model medium   # 1.5GB，实测无收益，别下
python tools\fetch_assets.py --all            # 全部下下来自己试
```

下完在界面的「模型」下拉里直接切即可，不用改 `config.json`。

---

## GPU 加速

whisper.cpp 的 CPU 版和 CUDA 版是**两套不同的 DLL**，同一个目录里只能放一套。
本工具把它们装在两个位置，按需切换。

```bat
python tools\fetch_assets.py --cuda
```

这会额外下载约 700MB 的 CUDA 构建到 `vendor\whisper_cuda\`，
并自动补上 CUDA 运行时 DLL（构建包本身不含 cudart，缺了它驱动再新也起不来）。

装好后 `config.json` 里的 `device` 保持 `"auto"` 即可，程序会自动探测并使用 GPU。
状态栏会显示当前实际跑在 CPU 还是 GPU，以及探测结论。
如果回退到 CPU，状态栏会写明具体原因（比如"CUDA 构建缺少 CUDA 运行时"）。

没装 CUDA 构建但检测到 N 卡时，状态栏会提示可以运行上面的命令来启用。

---

## 配置

`config.json` 首次运行自动生成：

| 字段 | 默认 | 说明 |
|---|---|---|
| `model` | `ggml-small.bin` | 模型文件名，放在 `models/` 下 |
| `language` | `zh` | `zh` / `en` / `auto` |
| `device` | `auto` | `auto` / `cpu` / `gpu` |
| `vad_aggressiveness` | `2` | 0 最松，3 最严，界面上的滑块就是它 |
| `silence_ms` | `600` | 停顿多久算一句话说完，调大则段落更长、更准但出字更慢 |
| `live_interval_ms` | `1200` | 多久刷一次预览 |
| `no_speech_thold` | `0.6` | 无语音概率阈值，调大更不容易出幻觉 |
| `min_segment_ms` | `500` | 短于此长度的段直接丢弃 |
| `simplify_chinese` | `true` | 统一转简体（whisper 会繁简混排） |
| `fix_punctuation` | `true` | 中文语境下标点转全角、引号转中文引号 |

界面上的麦克风选择和灵敏度滑块改动会自动存盘。断句阈值、预览间隔、
无语音阈值、繁简与标点开关在**「设置…」对话框**里（右下角按钮），
默认不用手改这个文件。

主题在「设置…」里切换，默认深色——长时间盯着看比纯白舒服。
配色不是纯黑，是略偏灰的深色，理由和编辑器里把主题调暗一样。

配置里除主题外的项只在下次录音时生效：VAD 参数是在开始录音那一刻构造的，构造到一半
再改就没有意义，界面会禁用相关控件。

---

## 单独调试用的脚本

```bat
python tools\test_bridge.py tests\jfk.wav    :: 验证 ctypes 绑定
python tools\test_vad.py                     :: 验证分段 + 静音幻觉防线
python tools\realtime_test.py                :: 麦克风全链路，终端逐行打印
python tools\realtime_test.py --file tests\jfk.wav   :: 同上但用文件
python tools\make_zh_sample.py               :: 生成中文测试音频
python tools\make_bench.py                   :: 生成带标准答案的测试集
python tools\make_bench.py --eval --model small       :: 算字符错误率
```

排查问题建议按这个顺序：先 `test_bridge.py`（绑定对不对），
再 `test_vad.py`（分得对不对），最后 `realtime_test.py`（串起来通不通）。

`make_bench.py` 是**量化验证**用的：用系统 TTS 合成六条不同风格的中文语音
（会议、数字、混排、疑问、长段落、短句）并保存标准答案，然后跑完整管线
算字符错误率。改任何影响识别的参数前后都跑一遍，比看几条输出靠谱得多。
（早期版本直接调引擎测，绕过了繁简统一，测出 41.5% 的 CER，一半是繁体
字造成的假错误——评测必须走和正式程序一样的管线。）

---

## 常见问题

**打开就报缺文件。** 跑 `python tools\fetch_assets.py`。

**提示找不到麦克风。** Windows 隐私设置里没给 Python 录音权限：
设置 → 隐私和安全性 → 麦克风 → 允许桌面应用访问。

**电平表长期贴着表底。** 说明麦克风基本没在收音：换输入设备，
或者调高 Windows 的输入音量（设置 → 系统 → 声音 → 输入）。
表底是 -60dBFS，正常说话应该在 -40 以上。

**一直识别不出来。** 先看界面上的●状态灯是否变成"说话中"。一直是"静音"说明 VAD
没触发，把灵敏度滑块往左调（更松），同时确认对着正确的麦克风说话。
状态栏也会显示实际选中的设备。

**正文里偶尔冒出奇怪的话。** 调小 `no_speech_thold` 之外，也可以把 `silence_ms`
调小让断句更频繁，或者把灵敏度往右调。

**文字没有标点。** Whisper 中文输出本身标点就少。想要标点得另接一个标点恢复模型，
本工具没做。

---

## 改名与打包

名称集中定义在 `branding.py`：`APP_NAME` / `APP_NAME_CN` / `APP_VERSION` /
`WINDOW_TITLE`。窗口标题、界面抬头、run.bat 横幅、README 都从这里取，
要改名字只动这一个文件。

打包成 exe 时建议用英文目录名（当前就是），中文路径在 PyInstaller 
单文件模式下容易出编码问题。

## 项目结构

```
branding.py         名称与版本号，改名只动这个文件
theme.py           浅色/深色两套配色与圆角
settings.py        设置对话框
widgets.py         自绘控件（电平表）
app.py             PySide6 界面与信号接线
audio.py           设备枚举、录音、环形缓冲
vad.py             webrtcvad 封装与分段状态机
whisper_bridge.py  ctypes 封装 whisper.dll
transcribe.py      推理线程、任务队列、预览预算
device.py          GPU/CPU 探测
config.py          配置读写
tools/             资源下载与调试脚本
vendor/            whisper.cpp 运行时
models/            GGUF 权重
```

### 升级 whisper.cpp 的注意事项

**切换模型必须"先加载新的、再释放旧的"。** 顺序反了会让同进程内后续
所有模型加载崩溃（`0xc000001d` 非法指令）。原因是 `whisper_free()` 之后
ggml 的全局后端注册表就废了，没法重建。先建后放则两个 context 可以共存，
实测来回切三次都正常。`ModelLoader.run()` 里这个顺序不能调换。

**CUDA 上下文不能被并发操作。** 释放模型必须发生在加载模型的那个线程上。
如果靠 Python 的 GC 在别的线程释放，会撞出 `access violation`。
所以整个程序只用一个后台加载线程，加载和释放都在它上面做。

`whisper_bridge.py` 里的结构体定义是照着 `vendor/whisper.h` 一行一行对出来的。
whisper.cpp 改过初始化接口（`whisper_init_params` 已改名 `whisper_context_params`，
`whisper_full_params` 末尾还嵌了 `whisper_vad_params`），升级后**必须重新核对头文件**，
否则 ctypes 会静默读到错位的字段，表现为识别结果乱码或直接崩溃。

另外有个坑：ggml 的计算后端是独立 DLL，它只在**宿主进程 exe 目录**里找后端。
正常用 `whisper-cli.exe` 没问题，但从 Python 加载时宿主是 `python.exe`，
扫描不到任何后端，建 context 时会直接 `GGML_ASSERT(device)` 崩掉。
`WhisperEngine._load_backends()` 里显式调 `ggml_backend_load_all_from_path(vendor_dir)`
就是为解决这个，改动时别删掉。

CUDA 还有一个更隐蔽的坑：ggml 内部用 `LoadLibrary` 加载 `ggml-cuda.dll` 时，
解析不到同目录下的 `cublas64_12.dll` / `cudart64_12.dll`，会**静默失败**——
release 版把日志静音了，外部只表现为 whisper 打印一行 `no GPU found`，
然后悄悄退回 CPU，很容易误以为是驱动问题。所以 `_load_backends()` 里会先用
ctypes 按绝对路径预加载 `ggml-cuda.dll`，把依赖先拉进进程，注册才拿得到 CUDA。
判断是否真的启用了 GPU，看后端数：2（CUDA+CPU）还是 1（只有 CPU）。
