#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- MLflow Database
    CREATE DATABASE mlflow;
    CREATE USER mlflow WITH PASSWORD 'mlflow';
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;
    
    -- Notification Service Database
    CREATE DATABASE "notify-data";
    CREATE USER postgres WITH PASSWORD '12345q';
    GRANT ALL PRIVILEGES ON DATABASE "notify-data" TO postgres;
EOSQL