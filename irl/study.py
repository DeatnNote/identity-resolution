"""
End-to-end study (3 seeds). Results -> results/*.json, MLflow runs -> mlruns/ (local).

  python -m irl.study
"""
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score

from .blocking import PRODUCTION_RULES, candidate_pairs, evaluate_blocking, pairs_for_rule
from .cluster import bcubed, connected_components, constrained_components
from .generate import drifted_batch, generate
from .match import FellegiSunter, LGBMMatcher, compare, levels, random_pairs
from .quality import DataQualityError, drift_report, expectations, run_expectations
from .standardize import standardize

warnings.filterwarnings("ignore")
OUT = "results"
SEEDS = [0, 1, 2]
N_REVIEW = 3000          # clerically reviewed pairs available for training / threshold choice


def best_threshold(y, s):
    qs = np.unique(np.quantile(s, np.linspace(0.05, 0.99, 120)))
    f1 = [f1_score(y, s >= q) for q in qs]
    return float(qs[int(np.argmax(f1))])


def pct(s):
    return pd.Series(s).rank(method="average", pct=True).to_numpy()


def run_seed(seed, log_mlflow=True):
    t0 = time.time()
    raw, people = generate(20000, seed)
    dq = run_expectations(raw)                               # raises on critical failures
    d = standardize(raw)
    pid = d["person_id"].to_numpy()
    hh = d["household_of"].to_numpy()

    blocking, total_true = evaluate_blocking(d)
    cp = candidate_pairs(d, PRODUCTION_RULES)
    f = compare(d, cp)
    lv = levels(f)
    y = (pid[f.i_l] == pid[f.i_r]).astype(int)
    same_hh = (hh[f.i_l] == hh[f.i_r]) & (y == 0)

    # person split: reviewed pairs come only from group A; evaluation only on group B
    rng = np.random.default_rng(seed)
    group_a = rng.random(people.shape[0]) < 0.5
    in_a = group_a[pid]
    aa = in_a[f.i_l] & in_a[f.i_r]
    bb = ~in_a[f.i_l] & ~in_a[f.i_r]

    # champion: unsupervised Fellegi-Sunter (Splink-style sessions)
    rand = levels(compare(d, random_pairs(len(d), 200000, seed)))
    sessions = {r: levels(compare(d, pairs_for_rule(d, r))) for r in ("email", "phone", "soundex+initial+zip")}
    fs = FellegiSunter().fit_sessions(sessions, rand, lv)
    fs.set_prior(len(d), len(pairs_for_rule(d, "email")))
    s_fs = fs.match_weight(lv)

    # reviewed sample from group A, stratified across the champion's score range
    idx_a = np.where(aa)[0]
    dec = pd.qcut(pd.Series(s_fs[idx_a]).rank(method="first"), 10, labels=False)
    review = pd.Series(idx_a).groupby(dec.to_numpy()).apply(
        lambda s: s.sample(min(len(s), N_REVIEW // 10), random_state=seed)).to_numpy()
    rng.shuffle(review)
    tr, va = review[: int(0.67 * len(review))], review[int(0.67 * len(review)):]

    lgbm = LGBMMatcher(seed).fit(f.iloc[tr], y[tr])
    s_lg = lgbm.predict_proba(f)
    th_fs = best_threshold(y[va], s_fs[va])
    th_lg = best_threshold(y[va], s_lg[va])

    res = {"seed": seed, "n_records": len(d), "n_people": int(people.shape[0]), "true_pairs": total_true,
           "candidate_pairs": len(cp), "review_pairs": int(len(review)),
           "data_quality": dq, "blocking": blocking.to_dict(orient="records")}
    ev = bb
    for name, s, th in (("champion_fellegi_sunter", s_fs, th_fs), ("challenger_lightgbm", s_lg, th_lg)):
        pred = s >= th
        fp = pred & ev & (y == 0)
        r = {"threshold": th, "pair_ap": average_precision_score(y[ev], s[ev]),
             "pair_precision": precision_score(y[ev], pred[ev]), "pair_recall": recall_score(y[ev], pred[ev]),
             "pair_f1": f1_score(y[ev], pred[ev]),
             "false_positives": int(fp.sum()), "fp_same_household_share": float(same_hh[fp].mean()) if fp.sum() else 0.0}
        if name == "challenger_lightgbm":
            r["brier"] = brier_score_loss(y[ev], s[ev])
        else:
            p = fs.predict_proba(lv)
            r["brier_uncalibrated_posterior"] = brier_score_loss(y[ev], p[ev])
        # clustering on all records; B-cubed on group-B records only
        recs_b = ~in_a
        for cname, labels in (
            ("connected_components", connected_components(len(d), cp, s, th)),
            # constrained clustering works on score percentiles so a "stricter threshold" step means
            # the same thing for both models (LightGBM probabilities saturate near 1.0)
            ("constrained_components", constrained_components(d, cp, pct(s), float((s < th).mean()))),
        ):
            r[cname] = bcubed(labels[recs_b], pid[recs_b])
        res[name] = r
    res["fs_parameters"] = fs.table().to_dict(orient="records")
    res["fs_prior_lambda"] = fs.lam
    # drift: a later batch with a new 'kiosk' source
    drift = drifted_batch(people, seed + 100)
    res["drift_psi"] = drift_report(raw, drift)
    res["drift_expectations"] = expectations(drift, baseline_null_rates=raw.isna().mean())
    try:
        run_expectations(drift)
        res["drift_batch_blocked"] = False
    except DataQualityError as e:
        res["drift_batch_blocked"] = str(e)
    res["seconds"] = time.time() - t0

    from .tracking import log_seed, promotion_decision
    res["promotion"] = promotion_decision(res["champion_fellegi_sunter"], res["challenger_lightgbm"])
    if log_mlflow:
        try:
            log_seed(res, lgbm.m)
        except Exception as e:                          # tracking must never lose the results
            res["mlflow_error"] = repr(e)[:300]
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    for seed in SEEDS:
        path = f"{OUT}/seed_{seed}.json"
        if os.path.exists(path):
            continue
        res = run_seed(seed)
        with open(path, "w") as fh:
            json.dump(res, fh, indent=2, default=float)
        c, g = res["champion_fellegi_sunter"], res["challenger_lightgbm"]
        print(f"seed {seed}: champion F1 {c['pair_f1']:.3f} B3 {c['constrained_components']['bcubed_f1']:.3f} | "
              f"challenger F1 {g['pair_f1']:.3f} B3 {g['constrained_components']['bcubed_f1']:.3f} "
              f"({res['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
