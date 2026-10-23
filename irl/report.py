"""Aggregate results/*.json into RESULTS.md (mean ± sd over seeds).  python -m irl.report"""
import glob
import json

import numpy as np

R = [json.load(open(p)) for p in sorted(glob.glob("results/seed_*.json"))]
MODELS = [("champion_fellegi_sunter", "Champion: Fellegi-Sunter (unsupervised)"),
          ("challenger_lightgbm", "Challenger: LightGBM (~2,000 labelled pairs)")]


def ms(vals, d=3):
    v = np.array(vals, float)
    return f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}" if len(v) > 1 else f"{v.mean():.{d}f}"


def get(path):
    out = []
    for r in R:
        x = r
        for k in path:
            x = x[k]
        out.append(x)
    return out


def main():
    L = []
    r0 = R[0]
    L.append("# Results\n")
    L.append(f"All numbers come from simulated data: about {r0['n_records']:,} records for "
             f"{r0['n_people']:,} synthetic people per seed, {len(R)} seeds (mean ± sd). "
             "No real customer data was used. Pair metrics and B-cubed are measured only on "
             "people the challenger never saw in training (person-level split). The challenger's labels "
             "come from the simulation's ground truth, i.e. they stand in for perfect clerical review.\n")
    L.append("## 1. Blocking (seed 0)\n")
    L.append("| Blocking rule | Candidate pairs | Pair completeness | Largest block |")
    L.append("|---|---:|---:|---:|")
    for b in r0["blocking"]:
        L.append(f"| {b['blocking']} | {b['candidate_pairs']:,} | {b['pair_completeness']:.3f} | {b['largest_block']:,} |")
    L.append(f"\nTrue matching pairs: {r0['true_pairs']:,}. The production union drops the skew-prone "
             "zip+last3 rule: it keeps nearly all of the recall with about 1/6 of the pairs.\n")
    L.append("## 2. Pairwise matching (held-out people)\n")
    L.append("| Model | Avg precision | Precision | Recall | F1 | Share of false positives that are same-household |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for k, name in MODELS:
        L.append(f"| {name} | {ms(get([k, 'pair_ap']))} | {ms(get([k, 'pair_precision']))} | "
                 f"{ms(get([k, 'pair_recall']))} | {ms(get([k, 'pair_f1']))} | {ms(get([k, 'fp_same_household_share']), 2)} |")
    L.append(f"\nThresholds were chosen to maximise F1 on reviewed pairs from the training people only. "
             f"The Fellegi-Sunter posterior is badly calibrated (Brier {ms(get(['champion_fellegi_sunter', 'brier_uncalibrated_posterior']))} "
             f"vs. {ms(get(['challenger_lightgbm', 'brier']))} for LightGBM) because household members agree on "
             "surname, address and phone together, which breaks its independence assumption; the champion "
             "therefore uses the match weight with a reviewed-pair threshold, not the posterior.\n")
    L.append("## 3. Clustering into identities (B-cubed, held-out people)\n")
    L.append("| Model | Clustering | B³ precision | B³ recall | B³ F1 | Over-merged clusters | Largest cluster |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for k, name in MODELS:
        for c, cn in (("connected_components", "Connected components"), ("constrained_components", "Constrained (cannot-link + size cap)")):
            L.append(f"| {name.split(':')[0]} | {cn} | {ms(get([k, c, 'bcubed_precision']))} | {ms(get([k, c, 'bcubed_recall']))} | "
                     f"{ms(get([k, c, 'bcubed_f1']))} | {ms(get([k, c, 'overmerged_clusters']), 0)} | {ms(get([k, c, 'largest_cluster']), 0)} |")
    L.append("\n## 4. Champion / challenger decision\n")
    L.append("Rule: promote only if B³ F1 rises by at least 0.005 **and** over-merged clusters do not increase "
             "(merging two different people is the costlier error for marketing activation).\n")
    L.append("| Seed | B³ F1 gain | Change in over-merged clusters | Promoted? |")
    L.append("|---|---:|---:|---|")
    for r in R:
        p = r["promotion"]
        L.append(f"| {r['seed']} | {p['bcubed_f1_gain']:+.3f} | {p['overmerge_change']:+d} | {'yes' if p['promote_challenger'] else 'no'} |")
    L.append("\nThe challenger finds clearly more of the true links, but it also merges a few more distinct people, "
             "so the gate keeps the champion. Whether that trade is worth it is a business decision; the gate "
             "makes it explicit instead of letting a higher headline F1 decide.\n")
    L.append("## 5. Drift and data quality on a later batch (seed 0)\n")
    L.append("A simulated later batch adds a new `kiosk` source (no email, mostly no phone, surname only).\n")
    L.append("| Feature | PSI | Alert (> 0.25) |")
    L.append("|---|---:|---|")
    for f, v in r0["drift_psi"].items():
        L.append(f"| {f} | {v['psi']:.3f} | {'yes' if v['alert'] else 'no'} |")
    fails = [e for e in r0["drift_expectations"] if not e["passed"]]
    L.append("\nFailed (warning-level) expectations: " + "; ".join(f"`{e['expectation']}` ({e['detail']})" for e in fails)
             + ". No critical expectation failed, so the batch is not blocked; it is flagged for review.\n")
    L.append("## 6. Spark blocking and key skew (zip + first 3 letters of surname)\n")
    L.append("Local Spark (`local[*]`, 2 cores, 16 shuffle partitions, AQE and broadcast joins off so skew is visible).\n")
    L.append("| Records | Strategy | Candidate pairs | Pair completeness | Largest task (rows) | Largest / median task |")
    L.append("|---:|---|---:|---:|---:|---:|")
    for s in sorted((json.load(open(p)) for p in glob.glob("results/spark_skew_*.json")), key=lambda s: s["records"]):
        for name, v in s["strategies"].items():
            L.append(f"| {s['records']:,} | {name} | {v['candidate_pairs']:,} | {v['pair_completeness']:.3f} | "
                     f"{v['max_task_rows']:,} | {v['max_over_median']} |")
    L.append("\nSalting splits each block larger than 100 records across 8 tasks: the same pairs and recall, with a "
             "much smaller largest task. Capping removes the skew by dropping big blocks, at a large recall cost for "
             "this rule (other rules must catch those pairs). On 2 local cores wall-clock time did not improve "
             "(salting adds replication work); the benefit is a balanced largest task, which matters when a cluster "
             "has more executors than the skewed key can use.\n")
    with open("RESULTS.md", "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
