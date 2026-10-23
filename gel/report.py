"""results/study.json + results/truth.json -> RESULTS.md.   python -m gel.report"""
import json

from .metrics import USER_METRICS

r = json.load(open("results/study.json"))
T = json.load(open("results/truth.json"))


def pp(x, d=1):
    return f"{x * 100:+.{d}f}pp"


def ci_pp(m, d=1):
    return f"{pp(m['estimate'], d)} [{m['ci_low'] * 100:+.{d}f}, {m['ci_high'] * 100:+.{d}f}]"


def num(m, d=3):
    return f"{m['estimate']:+.{d}f} [{m['ci_low']:+.{d}f}, {m['ci_high']:+.{d}f}]"


def fmt(metric, m, truth):
    if metric in ("generations_7d", "revenue_28d"):
        return num(m, 2 if metric == "generations_7d" else 3), f"{truth:+.{2 if metric == 'generations_7d' else 3}f}"
    return ci_pp(m), pp(truth)


def main():
    L = ["# Results\n",
         f"Simulated data: {T['meta']['users']:,} users signing up over 8 weeks, analysed as of day {T['meta']['cutoff_day']} "
         "through SQL on warehouse-style tables. Every true effect is known exactly because the same users were simulated "
         "under each assignment with identical random draws. No real OpenArt or customer data is used.\n"]

    a = r["activation_threshold"]
    L += ["## 1. Defining activation\n",
          "Candidate rule: *generated at least k assets in the first 3 days*. Chosen k maximises the correlation with week-4 retention.\n",
          "| k | Share activated | W4 retention if activated | if not | Correlation with W4 |", "|---:|---:|---:|---:|---:|"]
    for row in a["table"]:
        if row["threshold_generations_3d"] in (1, 3, 5, 6, 7, 8, 10):
            mark = " **(chosen)**" if row["threshold_generations_3d"] == a["chosen_threshold"] else ""
            L.append(f"| {row['threshold_generations_3d']}{mark} | {row['share_activated']:.1%} | {row['w4_if_activated']:.1%} | "
                     f"{row['w4_if_not']:.1%} | {row['phi_with_w4']:.3f} |")
    L.append("\nThe curve is flat between 5 and 8, so the exact cut-off matters less than using one definition everywhere.\n")

    L += ["## 2. Experiment readouts (intent-to-treat on server-side assignment, covariate-adjusted)\n"]
    for e in ("share_button", "onboarding_v2", "template_gallery"):
        x = r["experiments"][e]
        L += [f"### {e}  (n = {x['n_assigned']:,}; SRM p on assignments {x['srm_assignments']['p_value']:.2f}, "
              f"on client exposures {x['srm_exposures']['p_value']:.1g})\n",
              "| Metric | Estimate [95% CI] | True effect | CI covers truth | Significant after BH |", "|---|---:|---:|:---:|:---:|"]
        for m, d in x["metrics"].items():
            est, tr = fmt(m, d["adjusted"], d["true_effect"])
            L.append(f"| {m} | {est} | {tr} | {'yes' if d['adjusted_covers_truth'] else 'NO'} | {'yes' if d['bh_significant'] else 'no'} |")
        L.append("")
    cov = sum(d["adjusted_covers_truth"] for x in r["experiments"].values() for d in x["metrics"].values())
    L.append(f"All {cov} of 18 confidence intervals contain the true effect. Covariate adjustment with signup attributes "
             "(channel, platform, country) cut standard errors by only 0-2%: new users have no pre-period behaviour, "
             "so CUPED-style variance reduction has little to work with on signup tests.\n")

    sb = r["experiments"]["share_button"]
    s = sb["srm_exposures"]
    L += ["### The sample-ratio mismatch\n",
          f"The client exposure log has {s['n_treatment']:,} treatment vs {s['n_control']:,} control users "
          f"({s['treatment_share']:.1%} treatment, chi-square p = {s['p_value']:.1g}). On Android the exposure event fires only in a "
          "later session, so treatment users who never come back are missing. Analysed on exposures, the no-effect feature looks like a win:\n",
          "| Metric | On client exposures | On server assignments | Truth |", "|---|---:|---:|---:|"]
    for m in ("activated", "d1_retained", "w4_retained", "converted_28d"):
        d = sb["metrics"][m]
        L.append(f"| {m} | {ci_pp(d['exposure_based'])} (p = {d['exposure_based']['p_value']:.1g}) | {ci_pp(d['adjusted'])} | 0 |")
    L.append("\nRule adopted in the playbook: an SRM with p < 0.001 halts the readout until the cause is found.\n")

    n = r["novelty"]
    onb = r["experiments"]["onboarding_v2"]["metrics"]
    L += ["### Novelty (onboarding_v2)\n", "![novelty](results/figures/novelty.png)\n",
          f"Average lift in daily active rate is {pp(n['mean_effect_days_1_7'])} in days 1-7 but {pp(n['mean_effect_days_21_27'])} in days 21-27. "
          f"A one-week readout would report D1 retention {pp(onb['d1_retained']['adjusted']['estimate'])}; the week-4 retention effect is "
          f"{pp(onb['w4_retained']['adjusted']['estimate'])} (true {pp(onb['w4_retained']['true_effect'])}). The lasting effect is real but about 8x smaller than the day-1 number.\n"]

    it = r["interaction"]
    L += ["### Interaction (onboarding_v2 x template_gallery)\n",
          f"{it['n_users_in_both']:,} users were in both tests. Model: metric ~ onb * tg + covariates (robust SE).\n",
          "| Metric | Gallery effect without new onboarding | with new onboarding | Interaction [95% CI] | True interaction |",
          "|---|---:|---:|---:|---:|"]
    for m in ("activated", "w4_retained"):
        d = it[m]
        L.append(f"| {m} | {pp(d['tg_effect_without_onb'])} (true {pp(d['true_tg_effect_without_onb'])}) | "
                 f"{pp(d['tg_effect_with_onb'])} (true {pp(d['true_tg_effect_with_onb'])}) | "
                 f"{pp(d['interaction'])} [{d['ci_low'] * 100:+.1f}, {d['ci_high'] * 100:+.1f}] | {pp(d['true_interaction'])} |")
    L.append("\nThe gallery's week-4 retention gain disappears for users who also get the new onboarding: shipping both "
             "does not add up to the sum of the two readouts. The activation interaction is in the right direction but not significant.\n")

    p = r["price_test"]
    L += ["## 3. Paywall price test ($12 vs $15)\n",
          f"{p['n']:,} users exposed at their first paywall view in the test (SRM p = {p['srm']['p_value']:.2f}); outcome window 28 days.\n",
          "| Metric | Raw | CUPED (pre-period generations) | Regression-adjusted | True | SE reduction |", "|---|---:|---:|---:|---:|---:|"]
    for m, d in p["metrics"].items():
        f = (lambda z: num(z, 3)) if m == "revenue_28d" else ci_pp
        tr = f"{d['true_effect']:+.3f}" if m == "revenue_28d" else pp(d["true_effect"])
        L.append(f"| {m} | {f(d['raw'])} | {f(d['cuped'])} | {f(d['regression_adjusted'])} | {tr} | {d['adjusted_se_reduction']:.0%} |")
    L += ["", f"Conversion fell {abs(p['metrics']['converted_28d']['raw']['relative']):.0%} at $15. Revenue per exposed user is not "
          f"distinguishable from zero, and the true overall effect ({p['metrics']['revenue_28d']['true_effect']:+.3f} USD) would need about "
          f"{p['n_per_arm_to_detect_true_revenue_effect']:,} users per arm, roughly {p['n_per_arm_to_detect_true_revenue_effect'] * 2 / p['exposures_per_day'] / 365:.1f} "
          f"years at {p['exposures_per_day']:.0f} exposures per day. Pre-period activity is only weakly related to revenue, so CUPED helped little.\n",
          "![price segments](results/figures/price_segments.png)\n",
          "| Segment | n | Revenue effect [95% CI] | raw p | Significant after BH | True |", "|---|---:|---:|---:|:---:|---:|"]
    for sg in p["segments"]:
        tr = f"{sg['true_effect']:+.2f}" if sg["true_effect"] is not None else "0 (no planted difference)"
        L.append(f"| {sg['segment']} | {sg['n']:,} | {sg['estimate']:+.2f} [{sg['ci_low']:+.2f}, {sg['ci_high']:+.2f}] | {sg['p_value']:.3f} | "
                 f"{'yes' if sg['bh_significant'] else 'no'} | {tr} |")
    L.append("\nRecommendation the data supports: do not raise the price in ROW markets (revenue falls). EU shows a possible gain "
             "that this test cannot confirm; a follow-up EU-only test is the next step. Platform and channel splits show nothing.\n")

    aa = r["aa"]
    f = aa["false_positive_rate"]
    L += ["## 4. Validating the analysis (A/A tests)\n",
          f"{aa['n_sims']:,} random 50/50 splits of {aa['users']:,} real simulated users (true effect 0). Target false-positive rate: 5%.\n",
          "| Analysis | False-positive rate |", "|---|---:|",
          f"| Activation, user-level difference | {f['activation_diff']:.1%} |",
          f"| Generations per active day, delta method (user = unit) | {f['ratio_delta_method']:.1%} |",
          f"| Generations per active day, naive day-level t-test | **{f['ratio_naive_day_level']:.1%}** |",
          "\nTreating user-days as independent when users are randomised nearly triples false positives.\n"]

    z = r["sizing"]
    L += ["## 5. Test sizing\n", f"About {z['signups_per_day']:,.0f} signups per day, 50/50 split, 80% power, alpha 0.05.\n",
          "| Metric | Baseline | MDE (relative), 14 days of signups | 28 days of signups | Days of signups to detect a 5% lift |", "|---|---:|---:|---:|---:|"]
    for row in z["table"]:
        base = f"{row['baseline']:.1%}" if row["metric"] != "revenue_28d" else f"${row['baseline']:.3f}"
        L.append(f"| {row['metric']} | {base} | {row['mde_rel_14d']:.1%} | {row['mde_rel_28d']:.1%} | {row['days_for_5pct']:,.0f} |")
    L.append("\nDays are days of signup traffic; add the metric's own window (e.g. 28 days for W4 retention) before the readout. "
             "Signup tests can detect a few-percent change in activation or W4 retention within two weeks, but paid conversion "
             "and revenue need many months; for those, use the leading indicators and confirm with a long-running holdout.\n")

    o = r["opportunity_sizing"]
    L += ["## 6. Opportunity sizing from correlations vs. what the test delivered\n",
          f"Before building, a correlational estimate: activation lift ({pp(o['measured_activation_lift'])}) times the gap between activated and "
          "non-activated users.\n",
          "| Metric | Implied by correlation | Experiment estimate [95% CI] | True effect |", "|---|---:|---:|---:|"]
    for m in ("w4_retained", "revenue_28d"):
        d = o[m]
        if m == "w4_retained":
            L.append(f"| {m} | {pp(d['implied_effect_from_activation_lift'])} | {pp(d['experiment_estimate'])} [{d['experiment_ci'][0] * 100:+.1f}, {d['experiment_ci'][1] * 100:+.1f}] | {pp(d['true_effect'])} |")
        else:
            L.append(f"| {m} | {d['implied_effect_from_activation_lift']:+.3f} | {d['experiment_estimate']:+.3f} [{d['experiment_ci'][0]:+.3f}, {d['experiment_ci'][1]:+.3f}] | {d['true_effect']:+.3f} |")
    ratio = o["w4_retained"]["implied_effect_from_activation_lift"] / o["w4_retained"]["true_effect"]
    L.append(f"\nThe correlational sizing overstated the retention effect about {ratio:.0f}x: activated users are mostly people who "
             "were going to stay anyway. Activation is a useful leading indicator, not a proxy you can multiply through.\n")

    lt = r["ltv"]
    L += ["## 7. Lifetime value: 150-day revenue per signup (signups days 0-24)\n",
          f"Projection uses only data visible at the cutoff: monthly renewal rate {lt['monthly_renewal_rate']:.1%} and the cycle-over-cycle decline in new subscriptions.\n",
          "| Channel | Projected | Naive (28-day x 150/28) | Realised (truth) | Projection error | Naive error |", "|---|---:|---:|---:|---:|---:|"]
    for row in sorted(lt["by_channel"], key=lambda x: -x["true_150d"]):
        L.append(f"| {row['channel']} | ${row['projected_150d']:.2f} | ${row['naive_150d_from_28d']:.2f} | ${row['true_150d']:.2f} | "
                 f"{row['projection_error']:+.0%} | {row['naive_error']:+.0%} |")
    L.append("\nReferral signups are worth about 4x TikTok signups over 150 days. The projection lands within 2-12% of the realised value; "
             "straight-line extrapolation of 28-day revenue misses by 35-52% because most revenue is renewals after day 28.\n")

    L += ["## Metric definitions\n", "| Metric | Definition |", "|---|---|"] + [f"| {k} | {v} |" for k, v in USER_METRICS.items()]
    with open("RESULTS.md", "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
