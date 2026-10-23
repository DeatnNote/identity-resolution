-- Warehouse tables (Parquet exports; in production these are BigQuery tables). Dates cast to DATE.
CREATE OR REPLACE VIEW users          AS SELECT * REPLACE (CAST(signup_date AS DATE) AS signup_date) FROM read_parquet('data/users.parquet');
CREATE OR REPLACE VIEW assignments    AS SELECT * REPLACE (CAST(assigned_date AS DATE) AS assigned_date) FROM read_parquet('data/assignments.parquet');   -- server-side
CREATE OR REPLACE VIEW exposures      AS SELECT * REPLACE (CAST(exposed_date AS DATE) AS exposed_date) FROM read_parquet('data/exposures.parquet');     -- client-logged
CREATE OR REPLACE VIEW daily_activity AS SELECT * REPLACE (CAST(date AS DATE) AS date) FROM read_parquet('data/daily_activity.parquet');
CREATE OR REPLACE VIEW subscriptions  AS SELECT * REPLACE (CAST(started_date AS DATE) AS started_date, CAST(ended_date AS DATE) AS ended_date) FROM read_parquet('data/subscriptions.parquet');
CREATE OR REPLACE VIEW invoices       AS SELECT * REPLACE (CAST(created_date AS DATE) AS created_date) FROM read_parquet('data/invoices.parquet');
-- data is complete through the day before this date
CREATE OR REPLACE MACRO cutoff_date() AS DATE '2026-03-30';
