import json
import os

import pandas as pd
import pytest

from irl import pipeline
from irl.generate import drifted_batch, generate
from irl.quality import DataQualityError, drift_report, run_expectations


def test_clean_batch_passes_and_broken_batch_fails():
    raw, _ = generate(500, 0)
    run_expectations(raw)
    bad = pd.concat([raw, raw.head(3)])                       # duplicated record ids
    with pytest.raises(DataQualityError):
        run_expectations(bad)
    with pytest.raises(DataQualityError):
        run_expectations(raw.drop(columns=["zip"]))


def test_drift_report_flags_new_source():
    raw, people = generate(2000, 0)
    rep = drift_report(raw, drifted_batch(people, 3))
    assert rep["source"]["alert"]


def test_pipeline_end_to_end_and_data_contract(tmp_path):
    ds, root = "2026-01-01", str(tmp_path)
    for name, fn in pipeline.TASKS:
        if name == "ingest":
            fn(ds, root=root, n_people=1500)
        elif name == "cluster":
            fn(ds, root=root, threshold=13.5)
        else:
            fn(ds, root=root)
    out = pd.read_parquet(os.path.join(root, f"run={ds}", "identities.parquet"))
    raw = pd.read_parquet(os.path.join(root, f"run={ds}", "raw.parquet"))
    assert len(out) == len(raw) and out["identity_id"].notna().all()
    # data contract: no raw email or phone anywhere in the published table, no simulation truth
    published = set(out.astype(str).to_numpy().ravel())
    assert not (set(raw["email"].dropna()) & published)
    assert not (set(raw["phone"].dropna()) & published)
    assert "person_id" not in out.columns and "household_of" not in out.columns
    stats = json.load(open(os.path.join(root, f"run={ds}", "match_stats.json")))
    assert stats["records"] == len(out)


def test_data_quality_failure_is_not_retried(tmp_path, monkeypatch):
    calls = {"n": 0}

    def bad_validate(ds, root):
        calls["n"] += 1
        raise DataQualityError("record_id_unique")

    monkeypatch.setattr(pipeline, "TASKS", [("validate", bad_validate)])
    with pytest.raises(DataQualityError):
        pipeline.run_local("2026-01-01", root=str(tmp_path), retries=3, retry_delay=0, log=lambda *_: None)
    assert calls["n"] == 1


def test_transient_failure_is_retried(tmp_path, monkeypatch):
    calls = {"n": 0}

    def flaky(ds, root):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("S3 timeout")

    monkeypatch.setattr(pipeline, "TASKS", [("flaky", flaky)])
    pipeline.run_local("2026-01-01", root=str(tmp_path), retries=3, retry_delay=0, log=lambda *_: None)
    assert calls["n"] == 3
