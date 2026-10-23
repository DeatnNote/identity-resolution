"""
Simulated freemium AI-creation app with KNOWN causal effects.

120,000 users sign up over 8 weeks. Each day a user is active or not; active users generate assets.
Free users run out of credits after FREE_CREDITS generations and then see a paywall on active
days; some subscribe. Subscriptions renew every 30 days (Stripe-style invoices), with voluntary
churn that depends on recent activity and price, and involuntary churn from failed payments.

Four experiments run on top, with effects chosen in advance:

  share_button      signups days 0-27   true effect ZERO. A client logging bug: on Android the
                                         treatment exposure is only logged in a LATER session, so
                                         treatment users who do not come back within 7 days are
                                         missing from the EXPOSURE log (server-side assignment is correct).
  onboarding_v2     signups days 14-41  big NOVELTY effect on early activity that decays within a
                                         week, small lasting effect.
  template_gallery  signups days 28-55  small lasting lift, but a NEGATIVE INTERACTION with
                                         onboarding_v2 (they help the same users in the same way).
  paywall_price     paywall views on day 30+: $12 (control) vs $15. Fewer conversions, higher price;
                                         price sensitivity differs by country tier.

Truth comes from common random numbers: the same users are simulated in several "worlds" that
differ only in experiment assignment, with identical random draws, so each experiment's true
average effect on the analysed population is known exactly.
"""
import numpy as np
import pandas as pd

N_USERS = 120_000
SIGNUP_DAYS = 56
CUTOFF = 84            # analysis "as of" day: data after this day is not visible to the analyst
HORIZON = 180          # the observed world is simulated further, to check LTV projections
FREE_CREDITS = 40
PRICE = {"control": 12.0, "treatment": 15.0}
START = pd.Timestamp("2026-01-05")

CHANNELS = {"organic": (0.35, 0.0), "paid_social": (0.30, -0.3), "tiktok_ugc": (0.20, -0.5), "referral": (0.15, 0.4)}
PLATFORMS = {"web": 0.5, "ios": 0.3, "android": 0.2}
COUNTRIES = {"US": 0.35, "EU": 0.25, "ROW": 0.40}
PRICE_EFFECT = {"US": -0.25, "EU": -0.10, "ROW": -0.55}          # log-odds of converting at $15 vs $12

EXPERIMENTS = {
    "share_button":     (0, 27),
    "onboarding_v2":    (14, 41),
    "template_gallery": (28, 55),
}
PRICE_TEST_START = 30


def poisson_icdf(u, lam, kmax=60):
    """Poisson draw by inverse CDF from a given uniform (keeps random numbers common across worlds)."""
    k = np.zeros(lam.shape, np.int32)
    p = np.exp(-lam)
    F = p.copy()
    for i in range(1, kmax):
        more = u > F
        if not more.any():
            break
        k += more
        p = p * lam / i
        F = F + p
    return k


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_users(n=N_USERS, seed=0):
    rng = np.random.default_rng(seed)
    dow_w = np.array([1.0, 1.0, 1.0, 1.0, 1.1, 1.25, 1.2])        # Mon..Sun
    w = dow_w[np.arange(SIGNUP_DAYS) % 7]
    signup = rng.choice(SIGNUP_DAYS, n, p=w / w.sum())
    ch = rng.choice(list(CHANNELS), n, p=[v[0] for v in CHANNELS.values()])
    shift = np.array([CHANNELS[c][1] for c in ch])
    u = pd.DataFrame({
        "user_id": np.arange(n),
        "signup_day": signup,
        "channel": ch,
        "platform": rng.choice(list(PLATFORMS), n, p=list(PLATFORMS.values())),
        "country_tier": rng.choice(list(COUNTRIES), n, p=list(COUNTRIES.values())),
    })
    u["intent"] = shift + rng.normal(0, 1, n)                       # latent, never shown to the analyst
    # server-side assignment: hash-like independent coin flips per experiment
    for e, (a, b) in EXPERIMENTS.items():
        elig = (u.signup_day >= a) & (u.signup_day <= b)
        u[e] = np.where(elig, rng.integers(0, 2, n), -1)            # -1 = not in the experiment
    u["paywall_price"] = rng.integers(0, 2, n)                     # applies to paywall views on day 30+
    return u


def _world_assignments(u):
    """Each world = a full assignment vector per experiment. 'obs' is what actually happened."""
    obs = {e: u[e].to_numpy() for e in list(EXPERIMENTS) + ["paywall_price"]}

    def force(**kw):
        w = {k: v.copy() for k, v in obs.items()}
        for e, val in kw.items():
            w[e] = np.where(obs[e] >= 0, val, -1) if e != "paywall_price" else np.full(len(u), val)
        return w
    worlds = {"obs": obs}
    for e in EXPERIMENTS:
        worlds[f"{e}=0"], worlds[f"{e}=1"] = force(**{e: 0}), force(**{e: 1})
    for a in (0, 1):
        for b in (0, 1):
            worlds[f"onb={a},tg={b}"] = force(onboarding_v2=a, template_gallery=b)
    worlds["price=0"], worlds["price=1"] = force(paywall_price=0), force(paywall_price=1)
    return worlds


def simulate(u, seed=0, horizon=HORIZON, worlds=None):
    """
    Returns (obs_logs, outcomes). obs_logs holds event tables for the observed world only;
    outcomes[world] holds per-user outcome arrays used to compute true effects.
    """
    worlds = worlds or _world_assignments(u)
    names = list(worlds)
    W, n = len(names), len(u)
    signup = u.signup_day.to_numpy()
    z = u.intent.to_numpy()
    country = u.country_tier.to_numpy()
    pe = np.array([PRICE_EFFECT[c] for c in country])
    A = {e: np.stack([worlds[w][e] for w in names]) for e in worlds["obs"]}          # W x n

    habit = np.zeros((W, n))
    cum_gen = np.zeros((W, n))
    paid = np.zeros((W, n), bool)
    sub_price = np.zeros((W, n))
    next_bill = np.full((W, n), -1)
    price_exposed_day = np.full((W, n), -1)
    # outcome accumulators (per world): active matrix for days since signup 0..34, revenue by day
    MAXT = 35
    active_t = np.zeros((W, n, MAXT), np.int8)
    gens_t = np.zeros((W, n, MAXT), np.int16)
    conv_day = np.full((W, n), -1)
    revenue_after_price = np.zeros((W, n))
    revenue_28 = np.zeros((W, n))                                   # within 28 days of signup
    active_after_price = np.zeros((W, n, 28), np.int8)

    logs_act, logs_sub, logs_inv, logs_exp = [], [], [], []
    sub_id = np.full(n, -1)
    sub_counter = 0
    inv_counter = 0

    for day in range(horizon):
        rng = np.random.default_rng(seed * 100_003 + day)
        u_act, u_gen, u_conv, u_churn, u_pay = (rng.random(n) for _ in range(5))
        live_worlds = W if day < CUTOFF else 1                       # only the observed world runs long
        sl = slice(0, live_worlds)
        t = day - signup                                             # days since signup
        alive = t >= 0
        tt = np.clip(t, 0, None)

        onb = A["onboarding_v2"][sl] == 1
        tg = A["template_gallery"][sl] == 1
        eff = (onb * (0.40 * np.exp(-tt / 3.0) + 0.05)
               + tg * 0.10
               - (onb & tg) * 0.12)
        logit = np.where(tt == 0, 3.0, -0.2 + 0.9 * z - 0.85 * np.log1p(tt) + 1.6 * habit[sl] + 0.9 * paid[sl]) + eff
        active = alive & (u_act < sigmoid(logit))

        lam = np.exp(0.9 + 0.3 * z + 0.3 * paid[sl] + onb * 0.10 * np.exp(-tt / 3.0) + tg * 0.06)
        gens = np.where(active, poisson_icdf(np.broadcast_to(u_gen, lam.shape), lam), 0).astype(int)
        out_of_credits = (~paid[sl]) & (cum_gen[sl] >= FREE_CREDITS)
        gens = np.where(out_of_credits, np.minimum(gens, 2), gens)   # small daily refill for free users
        paywall = active & out_of_credits

        # price test: first paywall view on/after day 30 is the exposure
        price_arm = A["paywall_price"][sl]
        in_price = day >= PRICE_TEST_START
        newly = paywall & in_price & (price_exposed_day[sl] < 0)
        price_exposed_day[sl] = np.where(newly, day, price_exposed_day[sl])
        price_now = np.where(in_price & (price_arm == 1), PRICE["treatment"], PRICE["control"])
        conv_logit = -4.4 + 0.7 * z + np.where(price_now > 12, pe, 0.0)
        convert = paywall & (u_conv < sigmoid(conv_logit))

        # billing: new subscriptions charge today; renewals every 30 days
        revenue_today = np.zeros((live_worlds, n))
        renew = paid[sl] & (next_bill[sl] == day)
        recent = habit[sl]
        churn_p = sigmoid(-2.4 - 2.5 * (recent - 0.4) + np.where(sub_price[sl] > 12, 0.15, 0.0))
        vol_churn = renew & (u_churn < churn_p)
        pay_fail = renew & ~vol_churn & (u_pay < 0.025)              # failed payment after retries
        renew_ok = renew & ~vol_churn & ~pay_fail
        revenue_today += renew_ok * sub_price[sl]
        next_bill[sl] = np.where(renew_ok, day + 30, next_bill[sl])
        ended = vol_churn | pay_fail
        paid[sl] = paid[sl] & ~ended
        new_sub = convert & ~paid[sl]
        sub_price[sl] = np.where(new_sub, price_now, sub_price[sl])
        paid[sl] = paid[sl] | new_sub
        next_bill[sl] = np.where(new_sub, day + 30, next_bill[sl])
        revenue_today += new_sub * price_now
        conv_day[sl] = np.where(new_sub & (conv_day[sl] < 0), day, conv_day[sl])

        habit[sl] = 0.75 * habit[sl] + 0.25 * active
        cum_gen[sl] += gens

        # outcome accumulators
        inwin = alive & (t < MAXT)
        if inwin.any() and day < CUTOFF:
            idx = np.where(inwin)[0]
            active_t[sl, idx, t[idx]] = active[:, idx]
            gens_t[sl, idx, t[idx]] = np.minimum(gens[:, idx], 32000)
        revenue_28[sl] += revenue_today * (alive & (t < 28))
        ped = price_exposed_day[sl]
        k = day - ped
        win = (ped >= 0) & (k >= 0) & (k < 28)
        revenue_after_price[sl] += revenue_today * win
        if day < CUTOFF:
            wi, ui = np.where(win)
            active_after_price[wi, ui, k[wi, ui]] = active[wi, ui]

        # ---- event logs for the observed world (index 0) only, and only what the analyst could see
        o_active = active[0]
        if o_active.any():
            ids = np.where(o_active)[0]
            logs_act.append(pd.DataFrame({"user_id": ids, "day": day, "generations": gens[0, ids],
                                          "paywall_views": paywall[0, ids].astype(int)}))
        if newly[0].any():
            ids = np.where(newly[0])[0]
            logs_exp.append(pd.DataFrame({"experiment": "paywall_price", "user_id": ids, "day": day,
                                          "variant": price_arm[0, ids]}))
        for ids, kind in ((np.where(new_sub[0])[0], "new"), (np.where(renew_ok[0])[0], "renewal")):
            if len(ids):
                if kind == "new":
                    sub_id[ids] = sub_counter + np.arange(len(ids))
                    sub_counter += len(ids)
                    logs_sub.append(pd.DataFrame({"subscription_id": sub_id[ids], "user_id": ids, "started_day": day,
                                                  "price": price_now[0, ids], "ended_day": -1, "end_reason": None}))
                logs_inv.append(pd.DataFrame({"invoice_id": inv_counter + np.arange(len(ids)), "subscription_id": sub_id[ids],
                                              "user_id": ids, "day": day, "amount_cents": (sub_price[0, ids] * 100).round().astype(int),
                                              "status": "paid", "billing_reason": "subscription_create" if kind == "new" else "subscription_cycle"}))
                inv_counter += len(ids)
        for ids, reason in ((np.where(vol_churn[0])[0], "canceled"), (np.where(pay_fail[0])[0], "payment_failed")):
            if len(ids):
                logs_sub.append(pd.DataFrame({"subscription_id": sub_id[ids], "user_id": ids, "started_day": -1,
                                              "price": np.nan, "ended_day": day, "end_reason": reason}))
                if reason == "payment_failed":
                    logs_inv.append(pd.DataFrame({"invoice_id": inv_counter + np.arange(len(ids)), "subscription_id": sub_id[ids],
                                                  "user_id": ids, "day": day, "amount_cents": (sub_price[0, ids] * 100).round().astype(int),
                                                  "status": "uncollectible", "billing_reason": "subscription_cycle"}))
                    inv_counter += len(ids)

    outcomes = {w: {"active_t": active_t[i], "gens_t": gens_t[i], "conv_day": conv_day[i],
                    "revenue_28": revenue_28[i], "price_exposed_day": price_exposed_day[i],
                    "revenue_after_price": revenue_after_price[i], "active_after_price": active_after_price[i]}
                for i, w in enumerate(names)}
    activity = pd.concat(logs_act, ignore_index=True)
    subs = pd.concat(logs_sub, ignore_index=True)
    starts = subs[subs.started_day >= 0].drop(columns=["ended_day", "end_reason"])
    ends = subs[subs.ended_day >= 0][["subscription_id", "ended_day", "end_reason"]]
    subscriptions = starts.merge(ends, on="subscription_id", how="left")
    subscriptions["ended_day"] = subscriptions["ended_day"].fillna(-1).astype(int)
    logs = {"activity": activity, "subscriptions": subscriptions,
            "invoices": pd.concat(logs_inv, ignore_index=True),
            "price_exposures": pd.concat(logs_exp, ignore_index=True)}
    return logs, outcomes


def exposures(u, activity):
    """Client-side exposure log for the three signup experiments, WITH the share_button logging bug."""
    rows = []
    r = activity.merge(u[["user_id", "signup_day"]], on="user_id")
    returned = set(r.loc[(r.day > r.signup_day) & (r.day <= r.signup_day + 7), "user_id"])
    for e in EXPERIMENTS:
        x = u[u[e] >= 0][["user_id", "signup_day", "platform", e]].rename(columns={e: "variant"})
        if e == "share_button":
            lost = (x.variant == 1) & (x.platform == "android") & ~x.user_id.isin(returned)
            x = x[~lost]
        rows.append(pd.DataFrame({"experiment": e, "user_id": x.user_id, "day": x.signup_day, "variant": x.variant}))
    return pd.concat(rows, ignore_index=True)


def to_tables(u, logs):
    """Analyst-facing tables with real dates (like a warehouse export). Latent intent is dropped."""
    d = lambda s: START + pd.to_timedelta(s, unit="D")              # noqa: E731
    cut = lambda df, col: df[df[col] < CUTOFF]                       # noqa: E731
    users = u[["user_id", "signup_day", "channel", "platform", "country_tier"]].copy()
    users["signup_date"] = d(users.pop("signup_day"))
    assignments = []
    for e in EXPERIMENTS:
        x = u[u[e] >= 0]
        assignments.append(pd.DataFrame({"experiment": e, "user_id": x.user_id, "variant": x[e],
                                         "assigned_date": d(x.signup_day)}))
    pe = cut(logs["price_exposures"], "day")
    assignments.append(pd.DataFrame({"experiment": "paywall_price", "user_id": pe.user_id, "variant": pe.variant,
                                     "assigned_date": d(pe.day)}))
    act = cut(logs["activity"], "day").copy()
    act["date"] = d(act.pop("day"))
    exp = exposures(u, logs["activity"][logs["activity"].day < CUTOFF])
    exp = pd.concat([exp, pe], ignore_index=True)
    exp["exposed_date"] = d(exp.pop("day"))
    subs = logs["subscriptions"][logs["subscriptions"].started_day < CUTOFF].copy()
    subs["started_date"] = d(subs.pop("started_day"))
    e = subs.pop("ended_day")
    visible_end = (e >= 0) & (e < CUTOFF)
    subs["ended_date"] = d(e).where(visible_end)
    subs["end_reason"] = subs["end_reason"].where(visible_end)
    inv = cut(logs["invoices"], "day").copy()
    inv["created_date"] = d(inv.pop("day"))
    variant = {0: "control", 1: "treatment"}
    a = pd.concat(assignments, ignore_index=True)
    a["variant"] = a["variant"].map(variant)
    exp["variant"] = exp["variant"].map(variant)
    return {"users": users, "assignments": a, "exposures": exp, "daily_activity": act,
            "subscriptions": subs, "invoices": inv}
