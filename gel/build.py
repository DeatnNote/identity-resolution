"""Simulate, write warehouse-style Parquet tables to data/, and the known truth to results/truth.json.

  python -m gel.build
"""
import json
import os
import time


from .metrics import true_effects
from .sim import CUTOFF, make_users, simulate, to_tables


def ltv_truth(u, logs):
    """Realised 180-day revenue per signup, by channel, for users who signed up on days 0-24."""
    inv = logs["invoices"]
    paid = inv[inv.status == "paid"].merge(u[["user_id", "signup_day", "channel"]], on="user_id")
    paid = paid[paid.day - paid.signup_day < 150]
    early = u[u.signup_day < 25]
    rev = paid[paid.user_id.isin(early.user_id)].groupby("channel").amount_cents.sum() / 100
    return (rev / early.groupby("channel").size()).round(3).to_dict()


def main(seed=0, n=None):
    t0 = time.time()
    u = make_users(**({"n": n} if n else {}), seed=seed)
    logs, outcomes = simulate(u, seed=seed)
    tables = to_tables(u, logs)
    os.makedirs("data", exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(f"data/{name}.parquet", index=False)
    truth = true_effects(u, outcomes)
    truth["ltv_150d_per_signup_by_channel"] = ltv_truth(u, logs)
    truth["meta"] = {"users": len(u), "cutoff_day": CUTOFF, "seed": seed, "seconds": round(time.time() - t0, 1)}
    # the activation-threshold study needs w4 retention for EVERY user; save it as hidden truth for tests
    os.makedirs("results", exist_ok=True)
    with open("results/truth.json", "w") as fh:
        json.dump(truth, fh, indent=2)
    print({k: len(v) for k, v in tables.items()}, f"{time.time() - t0:.0f}s")
    return u, logs, outcomes


if __name__ == "__main__":
    main()
