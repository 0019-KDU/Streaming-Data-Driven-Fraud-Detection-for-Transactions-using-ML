@echo off
REM ============================================================================
REM Quick Upload Script - Upload IEEE-CIS Dataset to DigitalOcean Droplet
REM Run this from your LOCAL Windows machine
REM ============================================================================

echo ========================================
echo Upload IEEE-CIS Dataset to Droplet
echo ========================================
echo.

REM Get droplet IP
set /p DROPLET_IP="Enter your DigitalOcean Droplet IP: "
echo.

REM Check if SCP is available
where scp >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: SCP not found!
    echo.
    echo Install OpenSSH Client:
    echo   1. Settings ^> Apps ^> Optional Features
    echo   2. Add Feature ^> OpenSSH Client
    echo   3. Install
    echo.
    pause
    exit /b 1
)

REM Get dataset path
echo Where is your IEEE-CIS dataset?
echo Example: D:\datasets\ieee_cis
echo.
set /p DATASET_PATH="Enter full path to IEEE-CIS folder: "
echo.

if not exist "%DATASET_PATH%" (
    echo ERROR: Path not found: %DATASET_PATH%
    pause
    exit /b 1
)

REM Upload dataset
echo Uploading dataset to droplet...
echo This may take 10-30 minutes depending on your internet speed.
echo.
scp -r "%DATASET_PATH%" root@%DROPLET_IP%:/opt/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src/data/

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo SUCCESS! Dataset uploaded
    echo ========================================
    echo.
    echo Next steps:
    echo   1. SSH into droplet: ssh root@%DROPLET_IP%
    echo   2. Trigger training:
    echo      docker exec airflow-scheduler airflow dags trigger ieee_cis_training_dag
    echo   3. Monitor: http://%DROPLET_IP%:8080
    echo.
) else (
    echo.
    echo ERROR: Upload failed
    echo Make sure you can SSH into the droplet: ssh root@%DROPLET_IP%
    echo.
)

pause
