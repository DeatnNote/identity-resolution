# Experimentation playbook

The standards each test must meet before and after launch. The checks are implemented in
`gel/stats.py` and demonstrated in [RESULTS.md](RESULTS.md).

## Before launch
1. **One primary metric** and pre-registered secondary and guardrail metrics, taken from the metric
   definitions in `sql/10_metrics.sql`. No new metric is invented after the readout.
2. **Size the test.** Compute the minimum detectable effect from baseline, variance and traffic
   (`stats.mde`, `stats.n_per_arm_for`). If the MDE is larger than the effect the team expects, the test
   is underpowered: change the metric (a leading indicator), the traffic, or the duration, or don't run it.
3. **Duration** = days of traffic needed + the metric's maturity window, rounded up to whole weeks so
   every day of the week is represented equally.
4. **Randomisation unit = analysis unit.** Users are randomised, so user-level metrics are analysed per
   user; ratio metrics (generations per active day) use the delta method, never a day-level t-test.
5. **Check overlap** with other running tests that touch the same surface; plan an interaction
   analysis if users can be in both.

## During the test
6. **Sample-ratio mismatch check daily** on both the assignment table and the exposure log.
   SRM p < 0.001 halts the readout until the cause is found and fixed.
7. **No peeking at p-values** for a ship decision. Look only at SRM and guardrails, or use a
   sequential (always-valid) test if early stopping is needed.

## Readout
8. **Intent-to-treat** on server-side assignment. Exposure-based results only as a secondary view,
   and only when the exposure log has passed SRM.
9. **Variance reduction**: CUPED with a pre-period version of the metric where one exists; regression
   adjustment on pre-assignment attributes otherwise. Report the SE reduction.
10. **Multiple comparisons**: Benjamini-Hochberg across secondary metrics and across segments.
    A segment result that is significant only before correction is a hypothesis for the next test.
11. **Novelty and primacy**: plot the effect by days since exposure. Ship decisions use the effect
    after it stabilises, not the first week.
12. **Interactions**: for overlapping tests, estimate `metric ~ A * B`; if the interaction is material,
    the combined launch is evaluated on the both-on cell, not by adding the two readouts.
13. **Recommendation in one sentence**: ship / don't ship / iterate, the expected impact with its
    interval, and what would change the decision.

## After launch
14. **Holdout**: keep 5% of users on the old experience for long-run metrics (paid conversion, LTV)
    that the test itself cannot power.
