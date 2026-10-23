"""
Blocking: only compare records that share a cheap key. The trade-off is recall (true matches
that never become candidate pairs are lost forever) against cost (candidate pairs to score).
"""
import time

import numpy as np
import pandas as pd

BLOCKING_RULES = {
    "zip+last3":            lambda d: d["zip5"] + "|" + d["last_name"].str[:3],
    "soundex+initial+zip":  lambda d: d["last_soundex"] + "|" + d["first_initial"] + "|" + d["zip5"],
    "email":                lambda d: d["email_clean"],
    "phone":                lambda d: d["phone_clean"],
    "card4+zip":            lambda d: d["card_last4"] + "|" + d["zip5"],
}
# The union used by the pipeline: everything except the skew-prone zip+last3 rule
PRODUCTION_RULES = ["soundex+initial+zip", "email", "phone", "card4+zip"]


def pairs_for_rule(df, rule, max_block=None):
    key = BLOCKING_RULES[rule](df)
    k = pd.DataFrame({"i": np.arange(len(df)), "key": key}).dropna()
    if max_block:
        size = k.groupby("key")["i"].transform("size")
        k = k[size <= max_block]
    m = k.merge(k, on="key", suffixes=("_l", "_r"))
    m = m[m.i_l < m.i_r]
    return m[["i_l", "i_r"]]


def candidate_pairs(df, rules, max_block=None):
    parts = [pairs_for_rule(df, r, max_block) for r in rules]
    return pd.concat(parts, ignore_index=True).drop_duplicates().reset_index(drop=True)


def true_pairs_count(person_ids):
    c = pd.Series(person_ids).value_counts()
    return int((c * (c - 1) // 2).sum())


def evaluate_blocking(df, person_col="person_id", max_block=None):
    """Pair completeness (recall of true pairs) vs. number of candidate pairs, per rule and union."""
    n = len(df)
    total_true = true_pairs_count(df[person_col])
    pid = df[person_col].to_numpy()
    rows = []
    configs = [[r] for r in BLOCKING_RULES] + [list(BLOCKING_RULES), PRODUCTION_RULES]
    for rules in configs:
        t0 = time.time()
        cp = candidate_pairs(df, rules, max_block)
        secs = time.time() - t0
        tp = int((pid[cp.i_l.to_numpy()] == pid[cp.i_r.to_numpy()]).sum())
        key_sizes = pd.concat([BLOCKING_RULES[r](df) for r in rules]).dropna().value_counts()
        rows.append({"blocking": " ∪ ".join(rules) if len(rules) > 1 else rules[0],
                     "candidate_pairs": len(cp), "true_pairs_found": tp,
                     "pair_completeness": tp / total_true,
                     "reduction_ratio": 1 - len(cp) / (n * (n - 1) / 2),
                     "largest_block": int(key_sizes.max()), "seconds": round(secs, 2)})
    return pd.DataFrame(rows), total_true
