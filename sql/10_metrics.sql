-- Metric definitions: the single source of truth for product, growth and finance.
-- Mirrors gel/metrics.py (tests/test_metrics.py checks they agree).

CREATE OR REPLACE VIEW user_day AS
SELECT a.user_id, a.date, a.generations, a.paywall_views,
       date_diff('day', u.signup_date, a.date) AS days_since_signup
FROM daily_activity a JOIN users u USING (user_id);

CREATE OR REPLACE VIEW paid_invoices AS
SELECT i.*, date_diff('day', u.signup_date, i.created_date) AS days_since_signup
FROM invoices i JOIN users u USING (user_id)
WHERE i.status = 'paid';

-- one row per user whose first 28 days are fully observed
CREATE OR REPLACE VIEW user_metrics AS
WITH act AS (
  SELECT user_id,
         SUM(generations) FILTER (WHERE days_since_signup BETWEEN 0 AND 2)  AS gens_3d,
         SUM(generations) FILTER (WHERE days_since_signup BETWEEN 0 AND 6)  AS gens_7d,
         MAX(CASE WHEN days_since_signup = 1 THEN 1 ELSE 0 END)             AS d1_retained,
         MAX(CASE WHEN days_since_signup BETWEEN 21 AND 27 THEN 1 ELSE 0 END) AS w4_retained,
         COUNT(*) FILTER (WHERE days_since_signup BETWEEN 0 AND 6)          AS active_days_7d
  FROM user_day GROUP BY user_id),
rev AS (
  SELECT user_id,
         SUM(amount_cents) FILTER (WHERE days_since_signup BETWEEN 0 AND 27) / 100.0 AS revenue_28d,
         MAX(CASE WHEN billing_reason = 'subscription_create' AND days_since_signup BETWEEN 0 AND 27
                  THEN 1 ELSE 0 END) AS converted_28d
  FROM paid_invoices GROUP BY user_id)
SELECT u.user_id, u.signup_date, u.channel, u.platform, u.country_tier,
       date_trunc('week', u.signup_date) AS signup_week,
       CASE WHEN COALESCE(a.gens_3d, 0) >= 7 THEN 1 ELSE 0 END AS activated,  -- threshold from results/study.json
       COALESCE(a.gens_3d, 0)        AS gens_3d,
       COALESCE(a.d1_retained, 0)    AS d1_retained,
       COALESCE(a.w4_retained, 0)    AS w4_retained,
       COALESCE(a.gens_7d, 0)        AS generations_7d,
       COALESCE(a.active_days_7d, 0) AS active_days_7d,
       COALESCE(r.converted_28d, 0)  AS converted_28d,
       COALESCE(r.revenue_28d, 0)    AS revenue_28d
FROM users u
LEFT JOIN act a USING (user_id)
LEFT JOIN rev r USING (user_id)
WHERE u.signup_date + INTERVAL 28 DAY <= cutoff_date();

-- price test: outcomes in the 28 days from first paywall view in the test, plus a pre-period covariate
CREATE OR REPLACE VIEW price_metrics AS
WITH x AS (
  SELECT user_id, variant, assigned_date AS exposed_date FROM assignments
  WHERE experiment = 'paywall_price' AND assigned_date + INTERVAL 28 DAY <= cutoff_date()),
act AS (
  SELECT x.user_id,
         MAX(CASE WHEN date_diff('day', x.exposed_date, a.date) BETWEEN 21 AND 27 THEN 1 ELSE 0 END) AS w4_retained,
         SUM(a.generations) FILTER (WHERE date_diff('day', x.exposed_date, a.date) BETWEEN -14 AND -1) AS pre_generations_14d,
         COUNT(*) FILTER (WHERE date_diff('day', x.exposed_date, a.date) BETWEEN -14 AND -1) AS pre_active_days_14d
  FROM x JOIN daily_activity a USING (user_id) GROUP BY x.user_id),
rev AS (
  SELECT x.user_id,
         SUM(i.amount_cents) / 100.0 AS revenue_28d,
         MAX(CASE WHEN i.billing_reason = 'subscription_create' THEN 1 ELSE 0 END) AS converted_28d
  FROM x JOIN invoices i USING (user_id)
  WHERE i.status = 'paid' AND date_diff('day', x.exposed_date, i.created_date) BETWEEN 0 AND 27
  GROUP BY x.user_id)
SELECT x.user_id, x.variant, x.exposed_date, u.country_tier, u.platform, u.channel,
       COALESCE(r.converted_28d, 0) AS converted_28d, COALESCE(r.revenue_28d, 0) AS revenue_28d,
       COALESCE(a.w4_retained, 0) AS w4_retained,
       COALESCE(a.pre_generations_14d, 0) AS pre_generations_14d,
       COALESCE(a.pre_active_days_14d, 0) AS pre_active_days_14d
FROM x JOIN users u USING (user_id)
LEFT JOIN act a USING (user_id) LEFT JOIN rev r USING (user_id);
