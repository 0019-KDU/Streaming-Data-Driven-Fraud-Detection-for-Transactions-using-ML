#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create MLflow Database
    CREATE DATABASE mlflow;
    CREATE USER mlflow WITH PASSWORD 'mlflow';
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;

    -- Create Notification Service Database with Table
    CREATE DATABASE "notify-data";
    GRANT ALL PRIVILEGES ON DATABASE "notify-data" TO postgres;
    
    \c "notify-data"
    CREATE TABLE IF NOT EXISTS users (
        user_id INT PRIMARY KEY,
        email VARCHAR(255) NOT NULL UNIQUE
    );
EOSQL