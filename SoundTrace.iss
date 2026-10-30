; SoundTrace 安装脚本（Inno Setup）
;
; 打包命令:
;   pyinstaller SoundTrace.spec
;   iscc SoundTrace.iss
; 产物:
;   installer/SoundTrace-Setup-0.2.2.exe
;
; 安装包只含程序本体（约 31MB，其中大头是 Qt）。
; whisper.cpp 运行时和 GGUF 模型不进安装包，改为首次运行下载：
;   · 安装包小得多，装得快
;   · 模型可以按用户需求换（base / small / medium）
;   · 避免 700MB 的安装包在网速慢的地方直接失败
; 首次运行会弹出下载对话框，下载完就完全离线了。

#define AppName "SoundTrace"
#define AppNameCN "声迹"
; 版本号由 build.py 从 branding.py 同步进来，避免两处各写一份
#define AppVersion "0.2.2"
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
; 升级时旧版本可能还在运行（比如停在下载失败的对话框上不退），
; 旧版卸载器会因文件占用删不掉 exe，表现为"检测到已安装旧版、
; 然后无法安装"。除了这里的声明，[Code] 里会在安装/卸载前先结束实例。
CloseApplications=yes

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
; 源码工具链不再随包安装。原来装 tools\*.py + requirements.txt 是无用功：
; 安装目录里没有 Python 解释器，而项目模块在 {app}\_internal 而不是 {app}，
; 所有工具的 sys.path.insert(0, ROOT) 都指错了地方，一个也跑不起来。
; 真要自检或换模型，请用源码仓库跑 tools\fetch_assets.py。
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
; 自检功能写在同目录的日志，卸载时一并清掉
Type: files; Name: "{app}\soundtrace-selftest.log"

[Code]
// 安装前先结束可能还在运行的实例，避免旧版卸载器因文件占用失败
// 导致升级中断。进程不存在时 taskkill 返回非零，忽略即可。
function InitializeSetup(): Boolean;
var
  R: Integer;
begin
  Exec(ExpandConstant('{cmd}'),
       '/C taskkill /IM {#AppExeName} /F >nul 2>&1',
       '', SW_HIDE, ewWaitUntilTerminated, R);
  Result := True;
end;

// 从控制面板 / 系统设置里卸载时同样先结束实例
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  R: Integer;
begin
  if CurUninstallStep = usUninstall then
    Exec(ExpandConstant('{cmd}'),
         '/C taskkill /IM {#AppExeName} /F >nul 2>&1',
         '', SW_HIDE, ewWaitUntilTerminated, R);
end;
