"""Run the pipeline for one or more dates without Airflow:  python -m pipelines.run_local 2026-10-01 [2026-10-02 ...]"""
import json
import sys

from irl.pipeline import run_local

if __name__ == "__main__":
    dates = sys.argv[1:] or ["2026-10-01"]
    for ds in dates:                         # several dates = a backfill
        t = run_local(ds)
        print(json.dumps({"run": ds, "seconds": t, "total": round(sum(t.values()), 1)}))
