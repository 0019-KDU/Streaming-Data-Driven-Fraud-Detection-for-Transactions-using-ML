@echo off
REM Test real IEEE-CIS transaction with curl

echo ====================================
echo Testing Real IEEE-CIS Transaction
echo ====================================
echo.
echo Transaction Details:
echo   ID: 3970494
echo   Amount: $92.00
echo   Card: Mastercard Debit
echo   Email: gmail.com
echo.
echo Sending to: http://localhost:8000/api/v1/transactions/submit
echo.

curl -X POST "http://localhost:8000/api/v1/transactions/submit" ^
     -H "Content-Type: application/json" ^
     -d @test_transaction_payload.json

echo.
echo.
echo ====================================
echo Check Results:
echo   1. Dashboard: http://localhost:8501
echo   2. Spark logs: docker logs fraud-inference-spark --tail 50
echo ====================================
