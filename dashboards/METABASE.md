# Dashboard spec (Metabase-ready SQL)

Each tile is a named query in `sql/20_growth.sql`, written against the metric views in
`sql/10_metrics.sql`. In Metabase: create a native SQL question per query (BigQuery dialect needs
`DATE_DIFF(a, b, DAY)` instead of DuckDB's `date_diff('day', b, a)`), then add them to one
dashboard with a date filter on `signup_week`.

**Not built in Metabase here**; the queries are tested against DuckDB (`tests/test_sql.py`).

| Tile | Query | Chart | Question it answers |
|---|---|---|---|
| Funnel by signup week | `funnel_by_signup_week` | table + line per rate | Is activation / W4 / paid conversion moving week over week? |
| Funnel by channel | `funnel_by_channel` | bar | Which acquisition channels bring users who stay and pay? |
| Retention curves | `retention_curve_by_channel` | line, one per channel | Where does each channel's curve flatten? |
| Cohort triangle | `weekly_cohort_retention` | pivot heat map | Are newer cohorts retaining better? |
| Resurrection | `resurrection` | bar + share line (two charts) | How many weekly actives are coming back after 14+ dormant days? |
| MRR reconciliation | `mrr_reconciliation` | number + gap | Do product (subscriptions) and finance (invoices) agree on MRR? |
