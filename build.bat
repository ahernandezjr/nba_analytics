@echo off
SETLOCAL

REM Clean up old build artifacts
echo Cleaning up old build artifacts...
rmdir /s /q build dist nba_analytics.egg-info
if %errorlevel% neq 0 (
    echo Failed to clean up old build artifacts. Exiting.
    exit /b %errorlevel%
)

REM Build the package
echo Building the package...
python -m build
if %errorlevel% neq 0 (
    echo Failed to build the package. Exiting.
    exit /b %errorlevel%
)

echo Build process completed successfully.
ENDLOCAL