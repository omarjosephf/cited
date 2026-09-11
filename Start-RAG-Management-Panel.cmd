@echo off
setlocal
title Cited RAG Management Panel

cd /d "%~dp0"
set "PANEL_URL=http://127.0.0.1:8765/"
set "PYTHONPATH=%~dp0src"

if defined DOC_ASSISTANT_PYTHON set "PYTHON_EXE=%DOC_ASSISTANT_PYTHON%"
if not defined PYTHON_EXE set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" goto missing_python

"%PYTHON_EXE%" -c "import fastapi, uvicorn, pypdf, docx" 1>nul 2>nul
if errorlevel 1 goto missing_dependencies

if /I not "%CITED_PANEL_NO_BROWSER%"=="1" (
  start "" powershell.exe -NoProfile -WindowStyle Hidden -Command "$panelUrl = 'http://127.0.0.1:8765/'; for ($attempt = 0; $attempt -lt 60; $attempt++) { try { $response = Invoke-WebRequest -UseBasicParsing -Uri $panelUrl -TimeoutSec 1; if ($response.StatusCode -eq 200) { Start-Process $panelUrl; break } } catch {}; Start-Sleep -Milliseconds 500 }"
)

echo Starting the local, read-only Cited RAG Management Panel...
echo Keep this window open. Press Ctrl+C here when the demonstration is finished.
echo.
"%PYTHON_EXE%" -m assistant.cli inspect
set "PANEL_EXIT=%ERRORLEVEL%"
echo.
if "%PANEL_EXIT%"=="0" (
  echo Panel stopped.
) else (
  echo The panel stopped with exit code %PANEL_EXIT%.
)
pause
exit /b %PANEL_EXIT%

:missing_python
echo The project virtual environment was not found:
echo   %PYTHON_EXE%
echo.
echo Complete the one-time setup in README.md, then double-click this file again.
pause
exit /b 1

:missing_dependencies
echo The virtual environment exists but the panel dependencies are not installed.
echo.
echo From this folder, run the README.md setup command:
echo   .venv\Scripts\python.exe -m pip install -e ".[api,dev]"
pause
exit /b 1
