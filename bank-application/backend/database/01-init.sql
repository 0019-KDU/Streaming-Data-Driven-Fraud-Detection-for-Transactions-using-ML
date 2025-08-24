-- Create the bank_management database if it doesn't exist
CREATE DATABASE IF NOT EXISTS bank_management;

-- Create admin user for the application
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_catalog.pg_user 
        WHERE usename = 'bank_admin'
    ) THEN
        CREATE USER bank_admin WITH PASSWORD 'secure_password';
        GRANT ALL PRIVILEGES ON DATABASE bank_management TO bank_admin;
    END IF;
END
$$;
