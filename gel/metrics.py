"""
One definition of each metric, shared by the truth calculation (simulation outcomes) and the
analysis (SQL in sql/metrics.sql implements the same definitions on the warehouse tables;
tests/test_metrics.py checks the two agree).
"""

ACTIVATION_GENERATIONS = 7        # chosen from the retention-driver analysis (results/activation_threshold.json)
ACTIVATION_DAYS = 3

USER_METRICS = {
    "activated":        "generated >= 7 assets in signup days 0-2",
    "d1_retained":      "active on the day after signup",
    "w4_retained":      "active on any of days 21-27 after signup",
    "generations_7d":   "assets generated in days 0-6",
    "converted_28d":    "started a paid subscription within days 0-27",
    "revenue_28d":      "paid invoice amount (USD) within days 0-27",
}
PRICE_METRICS = {
    "converted_28d":    "subscribed within 28 days of first paywall view in the test",
    "revenue_28d":      "paid invoices (USD) within 28 days of exposure",
    "w4_retained":      "active on any of days 21-27 after exposure",
}


def user_metrics(o):
    a, g = o["active_t"], o["gens_t"].astype(int)
    return {
        "activated": (g[:, :ACTIVATION_DAYS].sum(1) >= ACTIVATION_GENERATIONS).astype(float),
        "d1_retained": a[:, 1].astype(float),
        "w4_retained": (a[:, 21:28].max(1) > 0).astype(float),
        "generations_7d": g[:, :7].sum(1).astype(float),
        "converted_28d": (o["revenue_28"] > 0).astype(float),
        "revenue_28d": o["revenue_28"],
    }


def price_metrics(o):
    return {
        "converted_28d": (o["revenue_after_price"] > 0).astype(float),
        "revenue_28d": o["revenue_after_price"],
        "w4_retained": (o["active_after_price"][:, 21:28].max(1) > 0).astype(float),
    }


def true_effects(u, outcomes):
    """True average treatment effects on each experiment's analysed population."""
    out = {}
    for e in ("share_button", "onboarding_v2", "template_gallery"):
        pop = u[e].to_numpy() >= 0
        m1, m0 = user_metrics(outcomes[f"{e}=1"]), user_metrics(outcomes[f"{e}=0"])
        out[e] = {k: {"control_mean": float(m0[k][pop].mean()), "true_effect": float((m1[k] - m0[k])[pop].mean())}
                  for k in m1}
        # effect over days since signup on daily activity (novelty profile)
        a1, a0 = outcomes[f"{e}=1"]["active_t"][pop], outcomes[f"{e}=0"]["active_t"][pop]
        out[e]["daily_active_effect_by_day"] = (a1.mean(0) - a0.mean(0))[:28].round(5).tolist()
    both = (u.onboarding_v2.to_numpy() >= 0) & (u.template_gallery.to_numpy() >= 0)
    mm = {k: user_metrics(outcomes[k]) for k in ("onb=0,tg=0", "onb=0,tg=1", "onb=1,tg=0", "onb=1,tg=1")}
    out["interaction"] = {k: {
        "tg_effect_without_onb": float((mm["onb=0,tg=1"][k] - mm["onb=0,tg=0"][k])[both].mean()),
        "tg_effect_with_onb": float((mm["onb=1,tg=1"][k] - mm["onb=1,tg=0"][k])[both].mean()),
    } for k in ("activated", "w4_retained", "generations_7d")}
    for k in out["interaction"]:
        d = out["interaction"][k]
        d["interaction"] = d["tg_effect_with_onb"] - d["tg_effect_without_onb"]
    exposed = outcomes["obs"]["price_exposed_day"]
    pop = (exposed >= 0) & (exposed < 84 - 28)                        # full 28-day window visible by cutoff
    p1, p0 = price_metrics(outcomes["price=1"]), price_metrics(outcomes["price=0"])
    out["paywall_price"] = {k: {"control_mean": float(p0[k][pop].mean()), "true_effect": float((p1[k] - p0[k])[pop].mean())}
                            for k in p1}
    seg = u.country_tier.to_numpy()
    out["paywall_price"]["revenue_by_country"] = {
        c: float((p1["revenue_28d"] - p0["revenue_28d"])[pop & (seg == c)].mean()) for c in ("US", "EU", "ROW")}
    out["paywall_price"]["n_population"] = int(pop.sum())
    return out
