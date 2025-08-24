-- Create database if not exists
CREATE DATABASE bank_management;

-- Create user if not exists
DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE  rolname = 'postgres') THEN
      
      CREATE ROLE postgres LOGIN PASSWORD '12345q';
   END IF;
END
$do$;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE bank_management TO postgres;
