@echo off
setlocal enabledelayedexpansion

REM run_analyses.bat
REM Runs hap40_interactome.jl and hap40_differential_interactome.jl sequentially,
REM logs all output to a file, and optionally shuts down the PC after completion.
REM
REM Usage:  Double-click, or run from repo root: examples\run_analyses.bat
REM         Pass --shutdown to shut down the PC after completion:
REM         examples\run_analyses.bat --shutdown

set THREADS=30
set EXAMPLES_DIR=%~dp0
set SHUTDOWN=0

REM Parse command-line arguments
:parse_args
if "%~1"=="--shutdown" (
    set SHUTDOWN=1
    shift
    goto parse_args
)
if not "%~1"=="" (
    shift
    goto parse_args
)

REM Repo root is one level up from examples/
for %%i in ("%EXAMPLES_DIR%..") do set REPO_ROOT=%%~fi

REM Timestamp via wmic (locale-safe)
for /f "tokens=2 delims==" %%a in ('wmic os get localdatetime /value') do set DT=%%a
set LOGFILE=%REPO_ROOT%\analysis_log_%DT:~0,8%_%DT:~8,4%.txt

echo ========================================
echo BayesInteractomics Analysis Run
echo ========================================
echo Date:    %date% %time%
echo Threads: %THREADS%
echo Repo:    %REPO_ROOT%
echo Log:     %LOGFILE%
echo.

(
    echo ========================================
    echo BayesInteractomics Analysis Run
    echo ========================================
    echo Date:    %date% %time%
    echo Threads: %THREADS%
    echo Repo:    %REPO_ROOT%
    echo ========================================
) > "%LOGFILE%"

REM ===== Script 1: HAP40 Interactome =====
set SCRIPT1=%EXAMPLES_DIR%hap40_interactome.jl
set TEMPOUT1=%TEMP%\julia_output_1_%RANDOM%.txt

echo.
echo ======================================================================
echo [1/2] HAP40 Interactome Analysis + Docking Request Generation
echo Script: %SCRIPT1%
echo Started: %date% %time%
echo ======================================================================
echo.

echo. >> "%LOGFILE%"
echo ====================================================================== >> "%LOGFILE%"
echo [1/2] HAP40 Interactome Analysis + Docking Request Generation >> "%LOGFILE%"
echo Started: %date% %time% >> "%LOGFILE%"
echo ====================================================================== >> "%LOGFILE%"

julia --threads=%THREADS% --project="%EXAMPLES_DIR%" "%SCRIPT1%" > "%TEMPOUT1%" 2>&1
set EXIT1=!ERRORLEVEL!

REM Show output on screen and append to log
if exist "%TEMPOUT1%" (
    type "%TEMPOUT1%"
    type "%TEMPOUT1%" >> "%LOGFILE%"
    del "%TEMPOUT1%"
)

if !EXIT1!==0 (
    set STATUS1=OK
) else (
    set STATUS1=FAILED - exit code !EXIT1!
)

echo.
echo [1/2] Finished: !STATUS1!
echo [1/2] Finished: !STATUS1! >> "%LOGFILE%"
echo.

REM ===== Script 2: Differential Interactome =====
set SCRIPT2=%EXAMPLES_DIR%hap40_differential_interactome.jl
set TEMPOUT2=%TEMP%\julia_output_2_%RANDOM%.txt

echo ======================================================================
echo [2/2] HAP40 Differential Interactome Analysis
echo Script: %SCRIPT2%
echo Started: %date% %time%
echo ======================================================================
echo.

echo. >> "%LOGFILE%"
echo ====================================================================== >> "%LOGFILE%"
echo [2/2] HAP40 Differential Interactome Analysis >> "%LOGFILE%"
echo Started: %date% %time% >> "%LOGFILE%"
echo ====================================================================== >> "%LOGFILE%"

julia --threads=%THREADS% --project="%EXAMPLES_DIR%" "%SCRIPT2%" > "%TEMPOUT2%" 2>&1
set EXIT2=!ERRORLEVEL!

REM Show output on screen and append to log
if exist "%TEMPOUT2%" (
    type "%TEMPOUT2%"
    type "%TEMPOUT2%" >> "%LOGFILE%"
    del "%TEMPOUT2%"
)

if !EXIT2!==0 (
    set STATUS2=OK
) else (
    set STATUS2=FAILED - exit code !EXIT2!
)

echo.
echo [2/2] Finished: !STATUS2!
echo [2/2] Finished: !STATUS2! >> "%LOGFILE%"

REM ===== Summary =====
echo.
echo ========================================
echo FINAL SUMMARY
echo ========================================
echo hap40_interactome.jl:              !STATUS1!
echo hap40_differential_interactome.jl: !STATUS2!
echo Log: %LOGFILE%
echo ========================================

(
    echo.
    echo ========================================
    echo FINAL SUMMARY
    echo ========================================
    echo hap40_interactome.jl:              !STATUS1!
    echo hap40_differential_interactome.jl: !STATUS2!
    echo ========================================
) >> "%LOGFILE%"

echo.
if !SHUTDOWN!==1 (
    echo Shutting down in 5 minutes... (Ctrl+C to cancel)
    timeout /t 300 /nobreak
    shutdown /s /t 0
) else (
    echo Shutdown skipped. Pass --shutdown to enable automatic shutdown.
)
