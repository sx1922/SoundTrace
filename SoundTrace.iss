; SoundTrace 安装脚本（Inno Setup）
;
; 打包命令:
;   pyinstaller SoundTrace.spec
;   iscc SoundTrace.iss
; 产物:
;   installer/SoundTrace-Setup-0.1.0.exe
;
; 安装包只含程序本体（约 160MB，其中大头是 Qt）。
; whisper.cpp 运行时和 GGUF 模型不进安装包，改为首次运行下载：
;   · 安装包小得多，装得快
;   · 模型可以按用户需求换（base / small / medium）
;   · 避免 700MB 的安装包在网速慢的地方直接失败
; 首次运行会弹出下载对话框，下载完就完全离线了。

#define AppName "SoundTrace"
#define AppNameCN "声迹"
#define AppVersion "0.1.0"
#define AppPublisher "SoundTrace"
#define AppExeName "SoundTrace.exe"
#define SourceDir "dist\SoundTrace"

[Setup]
AppId={{8E3C1A42-5B7D-4F19-9A6C-2D0F7B3E8A11}
AppName={#AppName} ({#AppNameCN})
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultGroupName=SoundTrace
DisableProgramGroupPage=yes
LicenseFile=assets\LICENSE.txt
OutputDir=installer
OutputBaseFilename=SoundTrace-Setup-{#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
; 装到用户目录，免 UAC 提权。小工具这样体验更好，也不会在无人值守
; 安装时卡在提权对话框上（静默模式下会直接退出，exit=2）。
; 想改成装到 Program Files，把下面两行换成 admin / {autopf} 并删掉
; PrivilegesRequiredOverridesAllowed。
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\SoundTrace
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#AppExeName}
ChangesAssociations=no
WizardSizePercent=120

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
chinesesimplified.LaunchApp=启动 {#AppName}
chinesesimplified.LaunchAppDesc=安装完成后立即启动

[Tasks]
Name: "desktopicon"; Description: "{cm:LaunchApp}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\{#AppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs
; 源码工具链跟着装，自检和换模型时用得上
Source: "tools\*.py"; DestDir: "{app}\tools"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
; 图标用通配符而不是硬引用：换图标时先删旧的也不会让构建失败
Source: "assets\*.ico"; DestDir: "{app}"; Flags: ignoreversion
; 注意：models\ 和 vendor\ 不在这里，首运行下载

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\卸载 {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchApp}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; 用户下载的模型和运行时默认保留，体积大且重装还能用
; 如果想彻底清干净，把下一行打开：
; Type: filesandordirs; Name: "{app}\models"
; Type: filesandordirs; Name: "{app}\vendor"
Type: files; Name: "{app}\config.json"
