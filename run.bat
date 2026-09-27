@echo off
rem ============================================================
rem  Local realtime speech-to-text
rem  Keep this file pure ASCII with CRLF endings: cmd.exe reads
rem  .bat in the OEM codepage and mis-parses LF-only files.
rem ============================================================
setlocal
cd /d "%~dp0"

set "PY="
set "VENV_PY=.venv\Scripts\python.exe"

rem --- 1. an existing venv always wins -----------------------------
if exist "%VENV_PY%" (
    set "PY=%VENV_PY%"
    goto :check_deps
)

rem --- 2. fast path: reuse any interpreter that already has deps --
rem     (avoids a needless venv rebuild on a machine that is set up)
for %%C in (py python python3) do (
    if not defined PY (
        %%C -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
        if not errorlevel 1 (
            %%C -c "import sounddevice, webrtcvad, PySide6" >nul 2>&1
            if not errorlevel 1 set "PY=%%C"
        )
    )
)
if defined PY goto :run

rem --- 3. otherwise find an interpreter and build a venv -----------
set "BASE="
for %%C in (py python python3) do (
    if not defined BASE (
        %%C -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
        if not errorlevel 1 set "BASE=%%C"
    )
)
if not defined BASE (
    echo [ERROR] Python 3.10+ not found on PATH.
    echo         Install it from https://www.python.org/downloads/
    echo         and tick "Add python.exe to PATH" during setup.
    goto :fail
)

echo [1/3] Creating virtual environment ...
%BASE% -m venv .venv
if errorlevel 1 goto :fail
set "PY=%VENV_PY%"
echo [2/3] Installing dependencies ...
"%PY%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :fail

:check_deps
%PY% -c "import sounddevice, webrtcvad, PySide6" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Dependencies are missing.
    echo         Run:  %PY% -m pip install -r requirements.txt
    goto :fail
)

rem --- assets ------------------------------------------------------
if exist "models\ggml-small.bin" goto :run
if exist "vendor\whisper\Release\whisper.dll" goto :run
echo [3/3] First run: downloading runtime and model, about 500 MB ...
%PY% tools\fetch_assets.py
if errorlevel 1 goto :fail

:run
echo [OK] SoundTrace 0.1.0
echo [OK] Starting ...
echo.
%PY% app.py
if errorlevel 1 goto :fail
endlocal
exit /b 0

:fail
echo.
echo [FAILED] See the messages above.
endlocal
pause
exit /b 1
