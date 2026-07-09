@echo off
:: ============================================================
:: restart_ollama.bat
:: Kills and restarts Ollama to free RAM before night run.
:: Run this if Ollama is stalling on later stocks.
:: ============================================================

echo [OLLAMA] Stopping Ollama to free RAM...
taskkill /f /im ollama.exe 2>nul
if %errorlevel%==0 (
    echo [OLLAMA] Stopped successfully
) else (
    echo [OLLAMA] Was not running
)

:: Wait 5 seconds for process to fully exit
timeout /t 5 /nobreak >nul

echo [OLLAMA] Starting fresh Ollama instance...
start "" "ollama" serve

:: Wait for Ollama to be ready
timeout /t 8 /nobreak >nul

echo [OLLAMA] Ready. RAM cleared.
echo.
echo You can now run: python night_runner.py
