@echo off
setlocal
title Beyond Possession
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0launchers\start_windows.ps1" %*
set "result=%errorlevel%"
if not "%result%"=="0" if not defined CI pause
exit /b %result%
