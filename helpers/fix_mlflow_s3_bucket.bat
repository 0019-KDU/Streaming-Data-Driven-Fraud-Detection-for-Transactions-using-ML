@echo off
REM ============================================================================
REM Fix MLflow S3 Bucket Issue - Create/Verify MinIO Bucket
REM ============================================================================
REM This script fixes the "NoSuchBucket" error by ensuring the mlflow bucket exists

echo ========================================
echo MLflow S3 Bucket Fix
echo ========================================
echo.

cd ..\src

echo Step 1: Checking if MinIO container is running...
docker ps | findstr minio
if %errorlevel% neq 0 (
    echo ERROR: MinIO container is not running!
    echo Please start MinIO first: docker compose up -d minio
    pause
    exit /b 1
)
echo MinIO is running.
echo.

echo Step 2: Creating mlflow bucket (if it doesn't exist)...
echo This will use the mc (MinIO Client) container to create the bucket.
echo.

REM Run MinIO Client to create bucket
docker compose run --rm mc sh -c "mc alias set minio http://minio:9000 minio minio123 && mc mb minio/mlflow --ignore-existing && mc ls minio/"

if %errorlevel% neq 0 (
    echo.
    echo WARNING: Failed to create bucket using mc container.
    echo Trying alternative method with docker exec...
    echo.
    
    REM Alternative: Use docker exec on minio container directly
    docker exec minio sh -c "mc alias set local http://localhost:9000 minio minio123 && mc mb local/mlflow --ignore-existing && mc ls local/"
)

echo.
echo Step 3: Verifying bucket exists...
docker compose run --rm mc sh -c "mc alias set minio http://minio:9000 minio minio123 && mc ls minio/"

echo.
echo ========================================
echo Fix Complete!
echo ========================================
echo.
echo The mlflow bucket should now exist in MinIO.
echo You can verify by opening: http://localhost:9001
echo   Username: minio
echo   Password: minio123
echo.
echo Now restart the Airflow training job to upload artifacts successfully.
echo.
pause
