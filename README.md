# identity-resolution-lab

An end-to-end, production-shaped **customer identity resolution** pipeline: messy records from
several restaurant systems (point-of-sale, online ordering, loyalty) are standardised, blocked,
matched and clustered into one identity per person, then published behind a PII-safe data
contract and an MCP server. Orchestration (Airflow), experiment tracking and a model registry
(MLflow), a Spark blocking job with a key-skew fix, data-quality gates, drift checks, Terraform
for AWS and CI are included.

> **All data is simulated.** About 49.5K records for 20K synthetic people per run, with nicknames, typos,
> email/phone variants, missing fields, people who move and households that share surname, address
> and phone. No real customer data is used, and nothing has been deployed to AWS.

## Pipeline

```
ingest → validate → standardize → block → score → cluster → publish ──► identities.parquet ──► MCP server
            │                                                              (hashed + masked PII only)
            └─ DataQualityError stops the run (not retried)
```

| Step | What it does | Code |
|---|---|---|
| validate | Schema, unique IDs, zip format, null-rate and source checks; critical failures stop the run | `irl/quality.py` |
| standardize | Nickname → canonical name, gmail dots / +tags, phone digits, address abbreviations, Soundex | `irl/standardize.py` |
| block | Union of 4 blocking rules (Soundex+initial+zip, email, phone, card-last-4+zip) | `irl/blocking.py` |
| score | **Champion:** Fellegi-Sunter match weights (unsupervised EM, Splink-style training sessions, *u* from random pairs). **Challenger:** LightGBM on comparison features | `irl/match.py` |
| cluster | Connected components, then split components with cannot-link conflicts (different birth years, clearly different first names) or implausible size | `irl/cluster.py` |
| publish | Identity table with salted hashes for exact lookup and masked strings for display; no raw email/phone/address/card | `irl/pipeline.py` |

## Results (3 seeds, people held out from training; full tables in [RESULTS.md](RESULTS.md))

| | Pair F1 | B³ F1 (constrained clustering) | Over-merged clusters |
|---|---:|---:|---:|
| Champion: Fellegi-Sunter (no labels) | 0.910 ± 0.003 | 0.955 ± 0.001 | 110 ± 6 |
| Challenger: LightGBM (~2,000 labelled pairs) | 0.962 ± 0.003 | 0.981 ± 0.001 | 122 ± 11 |

* **Blocking:** the 4-rule union keeps 98.1% of true pairs with 110K candidate pairs; adding the
  skew-prone zip+surname rule raises that to 99.0% but costs 664K pairs (6×).
* **Clustering:** cannot-link constraints cut over-merged clusters by about 4× (463 → 110 for the
  champion) and the largest cluster from 33 to 10 records, for a small recall loss.
* **Champion/challenger gate:** the challenger is better on F1 in every seed but merges slightly more
  distinct people, so the gate (F1 gain ≥ 0.005 **and** no increase in over-merges) kept the champion.
  The challenger is registered in MLflow under the `challenger` alias; `champion` moves only when the gate passes.
* **Calibration finding:** Fellegi-Sunter ranks pairs well (AP 0.961) but its posterior is badly calibrated
  (Brier 0.124) because household members agree on several fields at once, breaking its independence
  assumption. About half of its false positives are same-household pairs (6% for LightGBM). So the champion
  uses the match weight with a threshold chosen on reviewed pairs.
* **Drift:** a later batch with a new `kiosk` source triggers PSI alerts (source 3.5, name format 3.5) and two
  warning-level expectations; it is flagged, not blocked.
* **Spark skew:** at 250K records the naive self-join's largest task is 6.5× the median. Salting oversized
  blocks across 8 tasks keeps all 15.7M pairs and cuts the largest task from 4.0M to 1.5M rows; capping
  block size removes skew but drops this rule's pair completeness from 0.92 to 0.72. On 2 local cores,
  wall-clock time did not improve.

## Run it

```bash
pip install -r requirements.txt
pytest -q                                   # 14 tests
python -m irl.study && python -m irl.report # 3-seed study -> results/, mlflow.db, RESULTS.md (~1 min)
python -m pipelines.run_local 2026-10-01 2026-10-02   # local orchestrator, two dates = backfill
python -m mcp_server.demo_client someone@example.com  # starts the MCP server over stdio and calls it
mlflow ui --backend-store-uri sqlite:///mlflow.db     # runs, metrics, registered model + aliases
pip install pyspark && python spark/blocking_job.py --people 100000   # needs Java 17+
```

**Airflow** (`dags/identity_resolution_dag.py`): one task per step, retries with exponential backoff,
data-quality failures marked non-retryable, `catchup=True` for backfills, `max_active_runs=1`, a run
timeout and a failure callback. Checked with `airflow dags test identity_resolution 2026-10-02` on
Airflow 3.3.2 (all 7 tasks succeeded).

**MCP server** (`mcp_server/server.py`): read-only tools `lookup_by_email`, `lookup_by_phone`,
`lookup_by_name_zip`, `get_identity`, `match_stats`. Lookups normalise input the same way the
pipeline does and query hashes; responses contain masked fields only (tested).

**Terraform** (`infra/terraform/`): S3 lake (encrypted, versioned, public access blocked, raw PII
prefix expires after 30 days), Secrets Manager hash salt, EMR Serverless Spark application,
least-privilege job role, Glue catalog table for the identity output. Passes `fmt` and `validate`
(OpenTofu 1.8.5, AWS provider 5.72.1); **not applied**.

**CI** (`.github/workflows/ci.yml`): lint + tests, a Spark smoke test, Terraform fmt/validate.

## Limitations

* Simulated data; real POS and loyalty data will have failure modes this generator does not have.
* The challenger's labels come from simulation truth, standing in for perfect clerical review.
* Everything runs on one machine; the Spark timings are local and say nothing about EMR cost.
* The MCP server reads a local Parquet file; in AWS it would query the Glue/Athena table.
