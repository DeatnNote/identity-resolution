"""
The batch pipeline as plain, idempotent task functions. Each task reads its inputs from and
writes its outputs to a run directory (data/run=<date>/), so a failed task can be retried and
a past date re-run (backfill) without touching other runs. The Airflow DAG in dags/ and the local
runner below call exactly these functions.

  ingest -> validate -> standardize -> block -> score -> cluster -> publish

Matching uses the champion Fellegi-Sunter model (unsupervised, no labels needed in production).
"""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from .blocking import PRODUCTION_RULES, candidate_pairs, pairs_for_rule
from .cluster import constrained_components
from .generate import generate
from .match import FellegiSunter, compare, levels, random_pairs
from .quality import run_expectations
from .standardize import standardize

SALT = os.environ.get("IRL_HASH_SALT", "local-dev-salt")    # in AWS: from Secrets Manager
PII_COLUMNS = ["full_name", "email", "phone", "address", "card_last4", "birth_year"]


def run_dir(ds, root="data"):
    d = os.path.join(root, f"run={ds}")
    os.makedirs(d, exist_ok=True)
    return d


def _seed(ds):
    return int(ds.replace("-", "")) % 10_000


# ---------------------------------------------------------------- tasks
def ingest(ds, root="data", n_people=20000):
    """Stand-in for landing raw source extracts (POS, online, loyalty) in object storage."""
    raw, _ = generate(n_people, _seed(ds))
    path = os.path.join(run_dir(ds, root), "raw.parquet")
    raw.to_parquet(path, index=False)
    return path


def validate(ds, root="data"):
    raw = pd.read_parquet(os.path.join(run_dir(ds, root), "raw.parquet"))
    res = run_expectations(raw)                            # raises -> task fails, nothing downstream runs
    with open(os.path.join(run_dir(ds, root), "dq.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    return res


def standardize_task(ds, root="data"):
    raw = pd.read_parquet(os.path.join(run_dir(ds, root), "raw.parquet"))
    path = os.path.join(run_dir(ds, root), "std.parquet")
    standardize(raw).to_parquet(path, index=False)
    return path


def block(ds, root="data"):
    d = pd.read_parquet(os.path.join(run_dir(ds, root), "std.parquet"))
    cp = candidate_pairs(d, PRODUCTION_RULES)
    path = os.path.join(run_dir(ds, root), "pairs.parquet")
    cp.to_parquet(path, index=False)
    return path


def score(ds, root="data"):
    rd = run_dir(ds, root)
    d = pd.read_parquet(os.path.join(rd, "std.parquet"))
    cp = pd.read_parquet(os.path.join(rd, "pairs.parquet"))
    lv = levels(compare(d, cp))
    rand = levels(compare(d, random_pairs(len(d), 200000, _seed(ds))))
    sessions = {r: levels(compare(d, pairs_for_rule(d, r))) for r in ("email", "phone", "soundex+initial+zip")}
    fs = FellegiSunter().fit_sessions(sessions, rand, lv)
    fs.set_prior(len(d), len(pairs_for_rule(d, "email")))
    cp["match_weight"] = fs.match_weight(lv)
    cp.to_parquet(os.path.join(rd, "scored.parquet"), index=False)
    fs.table().to_csv(os.path.join(rd, "fs_parameters.csv"), index=False)
    return os.path.join(rd, "scored.parquet")


def cluster(ds, root="data", threshold=None):
    rd = run_dir(ds, root)
    d = pd.read_parquet(os.path.join(rd, "std.parquet"))
    sc = pd.read_parquet(os.path.join(rd, "scored.parquet"))
    if threshold is None:     # chosen offline on reviewed pairs (see results/); stored with the model
        threshold = float(json.load(open("results/seed_0.json"))["champion_fellegi_sunter"]["threshold"]) \
            if os.path.exists("results/seed_0.json") else 13.5
    s = sc["match_weight"].to_numpy()
    pct = pd.Series(s).rank(pct=True).to_numpy()
    labels = constrained_components(d, sc, pct, float((s < threshold).mean()))
    _, ident = np.unique(labels, return_inverse=True)
    out = pd.DataFrame({"record_id": d["record_id"], "identity_id": [f"ID{v:07d}" for v in ident]})
    out.to_parquet(os.path.join(rd, "clusters.parquet"), index=False)
    return os.path.join(rd, "clusters.parquet")


# ---------------------------------------------------------------- publish: the data contract
def _h(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else \
        hashlib.sha256((SALT + str(x)).encode()).hexdigest()[:24]


def mask_email(e):
    if not isinstance(e, str) or "@" not in e:
        return None
    local, dom = e.split("@", 1)
    return f"{local[0]}***@{dom}"


def mask_phone(p):
    return f"***-***-{p[-4:]}" if isinstance(p, str) and len(p) >= 4 else None


def publish(ds, root="data"):
    """
    Identity table consumers may read. Raw PII never leaves this task: email and phone appear
    only as salted hashes (for exact lookup) and masked strings (for display). Simulation truth
    columns are dropped.
    """
    rd = run_dir(ds, root)
    d = pd.read_parquet(os.path.join(rd, "std.parquet"))
    cl = pd.read_parquet(os.path.join(rd, "clusters.parquet"))
    t = d.merge(cl, on="record_id")
    out = pd.DataFrame({
        "record_id": t["record_id"], "identity_id": t["identity_id"], "source": t["source"],
        "name_display": [(f"{f[0].upper()}. " if isinstance(f, str) else "") + (l.title() if isinstance(l, str) else "")
                         for f, l in zip(t["first_name"], t["last_name"])],
        "name_key": t["first_name"].fillna("") + "|" + t["last_name"].fillna(""),
        "zip5": t["zip5"],
        "email_hash": t["email_clean"].map(_h), "email_masked": t["email_clean"].map(mask_email),
        "phone_hash": t["phone_clean"].map(_h), "phone_masked": t["phone_clean"].map(mask_phone),
    })
    out.to_parquet(os.path.join(rd, "identities.parquet"), index=False)
    sizes = out.groupby("identity_id").size()
    stats = {"run": ds, "records": int(len(out)), "identities": int(sizes.size),
             "multi_record_identities": int((sizes > 1).sum()), "largest_identity": int(sizes.max()),
             "records_by_source": out["source"].value_counts().to_dict(),
             "identity_size_distribution": sizes.value_counts().sort_index().to_dict()}
    with open(os.path.join(rd, "match_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2, default=int)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "LATEST"), "w") as fh:
        fh.write(ds)
    return stats


TASKS = [("ingest", ingest), ("validate", validate), ("standardize", standardize_task), ("block", block),
         ("score", score), ("cluster", cluster), ("publish", publish)]


def run_local(ds, root="data", retries=2, retry_delay=1.0, log=print):
    """Minimal local orchestrator: runs tasks in order with retries and timing."""
    timings = {}
    for name, fn in TASKS:
        for attempt in range(retries + 1):
            t0 = time.time()
            try:
                fn(ds, root=root)
                timings[name] = round(time.time() - t0, 2)
                log(f"[{ds}] {name:<12} ok   {timings[name]:>6.2f}s")
                break
            except Exception as e:
                # data-quality failures are not transient: do not retry them
                from .quality import DataQualityError
                if isinstance(e, DataQualityError) or attempt == retries:
                    log(f"[{ds}] {name:<12} FAILED: {e}")
                    raise
                log(f"[{ds}] {name:<12} retry {attempt + 1} after: {e}")
                time.sleep(retry_delay)
    return timings
