-- Growth funnel, retention cohorts, resurrection, revenue. Each query is a dashboard tile.

-- name: funnel_by_signup_week
SELECT signup_week, COUNT(*) AS signups,
       AVG(activated) AS activation_rate, AVG(d1_retained) AS d1_retention,
       AVG(w4_retained) AS w4_retention, AVG(converted_28d) AS paid_conversion_28d,
       SUM(revenue_28d) / COUNT(*) AS revenue_per_signup_28d
FROM user_metrics GROUP BY 1 ORDER BY 1;

-- name: funnel_by_channel
SELECT channel, COUNT(*) AS signups, AVG(activated) AS activation_rate, AVG(w4_retained) AS w4_retention,
       AVG(converted_28d) AS paid_conversion_28d, SUM(revenue_28d) / COUNT(*) AS revenue_per_signup_28d
FROM user_metrics GROUP BY 1 ORDER BY revenue_per_signup_28d DESC;

-- name: retention_curve_by_channel
-- share of a channel's signups active N days after signup (only fully observed days)
WITH base AS (SELECT user_id, channel, signup_date FROM users),
d AS (SELECT range AS n FROM range(0, 57))
SELECT b.channel, d.n AS days_since_signup,
       COUNT(DISTINCT a.user_id)::DOUBLE / COUNT(DISTINCT b.user_id) AS active_share
FROM base b CROSS JOIN d
LEFT JOIN daily_activity a ON a.user_id = b.user_id AND a.date = b.signup_date + d.n::INTEGER
WHERE b.signup_date + d.n::INTEGER < cutoff_date()
GROUP BY 1, 2 ORDER BY 1, 2;

-- name: weekly_cohort_retention
-- active in calendar week k after signup week (cohort triangle)
WITH w AS (
  SELECT u.user_id, date_trunc('week', u.signup_date) AS cohort,
         date_diff('week', date_trunc('week', u.signup_date), date_trunc('week', a.date)) AS week_n
  FROM users u JOIN daily_activity a USING (user_id)),
size AS (SELECT date_trunc('week', signup_date) AS cohort, COUNT(*) AS n FROM users GROUP BY 1)
SELECT w.cohort, w.week_n, COUNT(DISTINCT w.user_id)::DOUBLE / ANY_VALUE(s.n) AS retained
FROM w JOIN size s USING (cohort)
WHERE w.cohort + INTERVAL (w.week_n + 1) WEEK <= cutoff_date()
GROUP BY 1, 2 ORDER BY 1, 2;

-- name: resurrection
-- a user is "dormant" after 14 days without activity; resurrected = active again after that
WITH g AS (
  SELECT user_id, date,
         date_diff('day', LAG(date) OVER (PARTITION BY user_id ORDER BY date), date) AS gap
  FROM daily_activity)
SELECT date_trunc('week', date) AS week,
       COUNT(*) FILTER (WHERE gap > 14) AS resurrected_users,
       COUNT(DISTINCT user_id) AS weekly_active_users,
       COUNT(*) FILTER (WHERE gap > 14)::DOUBLE / COUNT(DISTINCT user_id) AS resurrected_share_of_wau
FROM g GROUP BY 1 ORDER BY 1;

-- name: mrr_reconciliation
-- finance check: MRR from the subscription table vs. MRR implied by the last 30 days of paid invoices
WITH active_subs AS (
  SELECT SUM(price) AS mrr_subscriptions FROM subscriptions
  WHERE started_date < cutoff_date() AND (ended_date IS NULL OR ended_date >= cutoff_date())),
inv AS (
  SELECT SUM(amount_cents) / 100.0 AS mrr_invoices FROM invoices
  WHERE status = 'paid' AND created_date >= cutoff_date() - INTERVAL 30 DAY)
SELECT mrr_subscriptions, mrr_invoices, mrr_invoices - mrr_subscriptions AS gap FROM active_subs, inv;
