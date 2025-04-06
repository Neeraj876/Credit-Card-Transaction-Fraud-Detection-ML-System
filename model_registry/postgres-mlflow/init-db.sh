#!/bin/bash
set -e

# Check if mlflow_db already exists
DB_EXISTS=$(psql -tAc "SELECT 1 FROM pg_database WHERE datname='mlflow_db'" --username "$POSTGRES_USER")

if [ "$DB_EXISTS" != "1" ]; then
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE mlflow_db;
  EOSQL
fi