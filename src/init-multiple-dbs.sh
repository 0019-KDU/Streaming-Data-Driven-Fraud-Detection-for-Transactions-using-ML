#!/bin/bash
# Runs once, when the Postgres volume is first created.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER mlflow WITH PASSWORD 'mlflow';
    CREATE DATABASE mlflow;
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;

    CREATE USER fraud WITH PASSWORD 'fraud';
    CREATE DATABASE fraud OWNER fraud;
EOSQL

psql -v ON_ERROR_STOP=1 --username fraud --dbname fraud -f /sql/decision_log.sql
