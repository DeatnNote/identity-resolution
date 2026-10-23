"""
All analyses, run against the warehouse tables through SQL, then compared with the known truth.

  python -m gel.build && python -m gel.study && python -m gel.report
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from . import stats as S
from .warehouse import connect

OUT = "results"
SIGNUP_METRICS = ["activated", "d1_retained", "w4_retained", "generations_7d", "converted_28d", "revenue_28d"]
PRIMARY = {"share_button": "w4_retained", "onboarding_v2": "w4_retained", "template_gallery": "w4_retained"}
SIGNUP_EXPS = ["share_button", "onboarding_v2", "template_gallery"]


def load(con):
    um = con.execute("SELECT * FROM user_metrics").df()
    asg = con.execute("SELECT experiment, user_id, variant FROM assignments").df()
    exp = con.execute("SELECT experiment, user_id, variant FROM exposures").df()
    pm = con.execute("SELECT * FROM price_metrics").df()
    return um, asg, exp, pm


def covariates(df):
    return pd.get_dummies(df[["channel", "platform", "country_tier"]], drop_first=True).to_numpy(float)


def covers(r, truth):
    return bool(r["ci_low"] <= truth <= r["ci_high"])


# ------------------------------------------------------------------ 1. metric design
def activation_threshold(um):
    rows = []
    w4 = um.w4_retained.to_numpy()
    for k in range(1, 13):
        a = (um.gens_3d >= k).to_numpy()
        phi = np.corrcoef(a, w4)[0, 1]
        rows.append({"threshold_generations_3d": k, "share_activated": a.mean(),
                     "w4_if_activated": w4[a].mean(), "w4_if_not": w4[~a].mean(), "phi_with_w4": phi})
    df = pd.DataFrame(rows)
    best = int(df.loc[df.phi_with_w4.idxmax(), "threshold_generations_3d"])
    return {"table": df.to_dict(orient="records"), "chosen_threshold": best}


def retention_drivers(um):
    d = um.copy()
    d["gens_3d_bucket"] = pd.cut(d.gens_3d, [-1, 0, 2, 4, 9, 10_000], labels=["0", "1-2", "3-4", "5-9", "10+"])
    fit = smf.logit("w4_retained ~ C(gens_3d_bucket) + d1_retained + C(channel) + C(platform) + C(country_tier)",
                    data=d).fit(disp=0)
    orr = pd.DataFrame({"odds_ratio": np.exp(fit.params), "ci_low": np.exp(fit.conf_int()[0]),
                        "ci_high": np.exp(fit.conf_int()[1])}).drop("Intercept")
    return {"odds_ratios": orr.round(3).reset_index(names="term").to_dict(orient="records"),
            "n": int(len(d)), "pseudo_r2": float(fit.prsquared)}


# ------------------------------------------------------------------ 2. signup experiments
def analyse_signup_experiment(e, um, asg, exp, truth):
    a = asg[asg.experiment == e].merge(um, on="user_id")
    t = (a.variant == "treatment").to_numpy()
    res = {"n_assigned": int(len(a)),
           "srm_assignments": S.srm_test((~t).sum(), t.sum())}
    x = exp[exp.experiment == e].merge(um, on="user_id")
    tx = (x.variant == "treatment").to_numpy()
    res["srm_exposures"] = S.srm_test((~tx).sum(), tx.sum())
    X = covariates(a)
    metrics = {}
    for m in SIGNUP_METRICS:
        raw = S.diff(a[m], t)
        adj = S.regression_adjusted(a[m].to_numpy(), t, X)
        tr = truth[e][m]["true_effect"]
        metrics[m] = {"raw": raw, "adjusted": adj, "true_effect": tr,
                      "raw_covers_truth": covers(raw, tr), "adjusted_covers_truth": covers(adj, tr),
                      "se_reduction": 1 - adj["se"] / raw["se"]}
        # exposure-based analysis (what you get if you trust the client log)
        metrics[m]["exposure_based"] = S.diff(x[m], tx)
        metrics[m]["exposure_based_covers_truth"] = covers(metrics[m]["exposure_based"], tr)
    sig, padj = S.benjamini_hochberg([metrics[m]["adjusted"]["p_value"] for m in SIGNUP_METRICS])
    for m, s_, p in zip(SIGNUP_METRICS, sig, padj):
        metrics[m]["bh_significant"], metrics[m]["bh_p"] = bool(s_), float(p)
    res["metrics"] = metrics
    return res


def novelty_profile(con, truth, e="onboarding_v2"):
    """Daily active rate by day since signup, treatment minus control, with CIs (user-level)."""
    q = f"""
    WITH a AS (SELECT user_id, variant FROM assignments WHERE experiment = '{e}'),
    d AS (SELECT range AS n FROM range(0, 28))
    SELECT d.n, a.variant, AVG(CASE WHEN ud.user_id IS NULL THEN 0 ELSE 1 END) AS p, COUNT(*) AS n_users
    FROM a CROSS JOIN d
    LEFT JOIN user_day ud ON ud.user_id = a.user_id AND ud.days_since_signup = d.n
    GROUP BY 1, 2 ORDER BY 1, 2"""
    df = con.execute(q).df().pivot(index="n", columns="variant", values=["p", "n_users"])
    pt, pc = df[("p", "treatment")], df[("p", "control")]
    nt, nc = df[("n_users", "treatment")], df[("n_users", "control")]
    se = np.sqrt(pt * (1 - pt) / nt + pc * (1 - pc) / nc)
    tr = truth[e]["daily_active_effect_by_day"]
    rows = [{"day": int(d), "effect": float(pt[d] - pc[d]), "ci_low": float(pt[d] - pc[d] - S.Z * se[d]),
             "ci_high": float(pt[d] - pc[d] + S.Z * se[d]), "true_effect": tr[int(d)]} for d in df.index]
    week1 = np.mean([r["effect"] for r in rows if 1 <= r["day"] <= 7])
    week4 = np.mean([r["effect"] for r in rows if 21 <= r["day"] <= 27])
    return {"by_day": rows, "mean_effect_days_1_7": float(week1), "mean_effect_days_21_27": float(week4)}


def interaction(um, asg, truth):
    w = asg[asg.experiment.isin(["onboarding_v2", "template_gallery"])].pivot(index="user_id", columns="experiment",
                                                                               values="variant").dropna()
    d = w.reset_index().merge(um, on="user_id")
    d["onb"] = (d.onboarding_v2 == "treatment").astype(int)
    d["tg"] = (d.template_gallery == "treatment").astype(int)
    out = {"n_users_in_both": int(len(d))}
    for m in ("activated", "w4_retained", "generations_7d"):
        fit = smf.ols(f"{m} ~ onb * tg + C(channel) + C(platform) + C(country_tier)", data=d).fit(cov_type="HC1")
        ci = fit.conf_int().loc["onb:tg"]
        tg_alone = fit.params["tg"]
        out[m] = {"interaction": float(fit.params["onb:tg"]), "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                  "p_value": float(fit.pvalues["onb:tg"]), "tg_effect_without_onb": float(tg_alone),
                  "tg_effect_with_onb": float(tg_alone + fit.params["onb:tg"]),
                  "true_interaction": truth["interaction"][m]["interaction"],
                  "true_tg_effect_without_onb": truth["interaction"][m]["tg_effect_without_onb"],
                  "true_tg_effect_with_onb": truth["interaction"][m]["tg_effect_with_onb"]}
    return out


# ------------------------------------------------------------------ 3. pricing test
def price_test(pm, truth):
    t = (pm.variant == "treatment").to_numpy()
    tr = truth["paywall_price"]
    res = {"n": int(len(pm)), "srm": S.srm_test((~t).sum(), t.sum()), "metrics": {}}
    X = np.column_stack([pm.pre_generations_14d, pm.pre_active_days_14d])
    for m in ("converted_28d", "revenue_28d", "w4_retained"):
        raw = S.diff(pm[m], t)
        cu = S.cuped(pm[m], t, pm.pre_generations_14d)
        adj = S.regression_adjusted(pm[m].to_numpy(), t, X)
        res["metrics"][m] = {"raw": raw, "cuped": cu, "regression_adjusted": adj,
                             "true_effect": tr[m]["true_effect"],
                             "cuped_se_reduction": 1 - cu["se"] / raw["se"],
                             "adjusted_se_reduction": 1 - adj["se"] / raw["se"],
                             "adjusted_covers_truth": covers(adj, tr[m]["true_effect"])}
    seg_rows = []
    for col in ("country_tier", "platform", "channel"):
        for v, g in pm.groupby(col):
            tt = (g.variant == "treatment").to_numpy()
            r = S.regression_adjusted(g.revenue_28d.to_numpy(), tt, np.column_stack([g.pre_generations_14d, g.pre_active_days_14d]))
            seg_rows.append({"segment": f"{col}={v}", "n": int(len(g)), **r,
                             "true_effect": tr["revenue_by_country"].get(v) if col == "country_tier" else None})
    sig, padj = S.benjamini_hochberg([r["p_value"] for r in seg_rows])
    for r, s_, p in zip(seg_rows, sig, padj):
        r["raw_significant"], r["bh_significant"], r["bh_p"] = r["p_value"] < 0.05, bool(s_), float(p)
    res["segments"] = seg_rows
    # power: how big a test would detect the true overall revenue effect?
    res["n_per_arm_to_detect_true_revenue_effect"] = S.n_per_arm_for(pm.revenue_28d.std(), abs(tr["revenue_28d"]["true_effect"]))
    res["exposures_per_day"] = float(len(pm) / pm.exposed_date.nunique())
    return res


# ------------------------------------------------------------------ 4. validating the method (A/A)
def aa_study(con, n_sims=2000, seed=0):
    """Random 50/50 splits of real users: how often does each method call a false 'significant'?"""
    um = con.execute("SELECT user_id, activated, generations_7d, active_days_7d FROM user_metrics").df()
    days = con.execute("SELECT user_id, generations FROM user_day WHERE days_since_signup BETWEEN 0 AND 6").df()
    um = um[um.active_days_7d > 0].reset_index(drop=True)
    uid_index = pd.Series(np.arange(len(um)), index=um.user_id)
    day_user = uid_index.reindex(days.user_id).to_numpy()
    keep = ~np.isnan(day_user)
    day_user, day_gen = day_user[keep].astype(int), days.generations.to_numpy()[keep].astype(float)
    rng = np.random.default_rng(seed)
    hits = {"activation_diff": 0, "ratio_delta_method": 0, "ratio_naive_day_level": 0}
    for _ in range(n_sims):
        t = rng.random(len(um)) < 0.5
        hits["activation_diff"] += S.diff(um.activated, t)["p_value"] < 0.05
        hits["ratio_delta_method"] += S.ratio_delta(um.generations_7d, um.active_days_7d, t)["p_value"] < 0.05
        hits["ratio_naive_day_level"] += S.diff(day_gen, t[day_user])["p_value"] < 0.05
    return {"n_sims": n_sims, "users": int(len(um)), "user_days": int(len(day_gen)),
            "false_positive_rate": {k: v / n_sims for k, v in hits.items()}}


# ------------------------------------------------------------------ 5. sizing
def sizing(um):
    signups_per_day = len(um) / um.signup_date.nunique()
    rows = []
    for m in ("activated", "w4_retained", "converted_28d", "revenue_28d"):
        sd, mean = um[m].std(), um[m].mean()
        r = {"metric": m, "baseline": float(mean)}
        for days in (7, 14, 28):
            r[f"mde_rel_{days}d"] = S.mde(sd, signups_per_day * days / 2) / mean
        for rel in (0.02, 0.05):
            r[f"days_for_{int(rel * 100)}pct"] = float(S.n_per_arm_for(sd, rel * mean) * 2 / signups_per_day)
        rows.append(r)
    return {"signups_per_day": float(signups_per_day), "table": rows}


def opportunity_sizing(um, onb_result, truth):
    """Sizing from correlations vs. what the experiment actually delivered."""
    act = um.activated == 1
    d_act = onb_result["metrics"]["activated"]["adjusted"]["estimate"]
    out = {"measured_activation_lift": d_act}
    for m in ("w4_retained", "revenue_28d"):
        gap = um.loc[act, m].mean() - um.loc[~act, m].mean()
        out[m] = {"observational_gap_activated_vs_not": float(gap),
                  "implied_effect_from_activation_lift": float(d_act * gap),
                  "experiment_estimate": onb_result["metrics"][m]["adjusted"]["estimate"],
                  "experiment_ci": [onb_result["metrics"][m]["adjusted"]["ci_low"], onb_result["metrics"][m]["adjusted"]["ci_high"]],
                  "true_effect": truth["onboarding_v2"][m]["true_effect"]}
    return out


# ------------------------------------------------------------------ 6. LTV
def ltv_projection(con, truth, horizon=150):
    """
    Project revenue per signup to 150 days for the day 0-24 signup cohorts using only data visible
    at the cutoff: subscribers renew with the observed monthly renewal rate, and new subscriptions
    in each later 30-day cycle shrink by the observed cycle-1 / cycle-0 ratio.
    Compared with the realised 150-day revenue (known from the longer simulation) and with a
    naive linear extrapolation of 28-day revenue.
    """
    q = """
    WITH c AS (SELECT user_id, channel FROM users WHERE signup_date < DATE '2026-01-05' + 25)
    SELECT c.channel, COUNT(DISTINCT c.user_id) AS signups,
      COUNT(*) FILTER (WHERE i.billing_reason = 'subscription_create' AND i.days_since_signup BETWEEN 0 AND 29) AS new_c0,
      COUNT(*) FILTER (WHERE i.billing_reason = 'subscription_create' AND i.days_since_signup BETWEEN 30 AND 59) AS new_c1,
      COALESCE(SUM(i.amount_cents) FILTER (WHERE i.days_since_signup BETWEEN 0 AND 59), 0) / 100.0 AS rev_60,
      COALESCE(SUM(i.amount_cents) FILTER (WHERE i.days_since_signup BETWEEN 0 AND 27), 0) / 100.0 AS rev_28,
      COALESCE(AVG(i.amount_cents) / 100.0, 12) AS avg_invoice
    FROM c LEFT JOIN paid_invoices i USING (user_id) GROUP BY 1"""
    df = con.execute(q).df()
    rho = float(con.execute("""
      SELECT COUNT(*) FILTER (WHERE status = 'paid')::DOUBLE / COUNT(*)
      FROM (SELECT status FROM invoices WHERE billing_reason = 'subscription_cycle'
            UNION ALL SELECT 'canceled' FROM subscriptions WHERE end_reason = 'canceled')""").fetchone()[0])
    proj = []
    for _, r in df.iterrows():
        decay = r.new_c1 / r.new_c0 if r.new_c0 else 0.0
        active = r.new_c0 * rho + r.new_c1            # subscribers paying in cycle 1
        rev = r.rev_60
        new = r.new_c1
        for _k in range(2, horizon // 30):            # cycles 2..4 cover days 60-149
            new = new * decay
            active = active * rho + new
            rev += active * r.avg_invoice
        proj.append(rev / r.signups)
    df["projected_150d"] = proj
    df["naive_150d_from_28d"] = df.rev_28 / df.signups * horizon / 28
    df["true_150d"] = df.channel.map(truth["ltv_150d_per_signup_by_channel"])
    df["projection_error"] = df.projected_150d / df.true_150d - 1
    df["naive_error"] = df.naive_150d_from_28d / df.true_150d - 1
    df["observed_60d"] = df.rev_60 / df.signups
    return {"monthly_renewal_rate": rho, "by_channel": df.round(4).to_dict(orient="records")}


def main():
    os.makedirs(OUT, exist_ok=True)
    truth = json.load(open(f"{OUT}/truth.json"))
    con = connect()
    um, asg, exp, pm = load(con)
    res = {"activation_threshold": activation_threshold(um), "retention_drivers": retention_drivers(um)}
    res["experiments"] = {e: analyse_signup_experiment(e, um, asg, exp, truth) for e in SIGNUP_EXPS}
    res["novelty"] = novelty_profile(con, truth)
    res["interaction"] = interaction(um, asg, truth)
    res["price_test"] = price_test(pm, truth)
    res["aa"] = aa_study(con)
    res["sizing"] = sizing(um)
    res["opportunity_sizing"] = opportunity_sizing(um, res["experiments"]["onboarding_v2"], truth)
    res["ltv"] = ltv_projection(con, truth)
    with open(f"{OUT}/study.json", "w") as fh:
        json.dump(res, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("done")


if __name__ == "__main__":
    main()
