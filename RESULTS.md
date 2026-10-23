# Results

All numbers come from simulated data: about 49,521 records for 20,000 synthetic people per seed, 3 seeds (mean ± sd). No real customer data was used. Pair metrics and B-cubed are measured only on people the challenger never saw in training (person-level split). The challenger's labels come from the simulation's ground truth, i.e. they stand in for perfect clerical review.

## 1. Blocking (seed 0)

| Blocking rule | Candidate pairs | Pair completeness | Largest block |
|---|---:|---:|---:|
| zip+last3 | 644,453 | 0.920 | 473 |
| soundex+initial+zip | 90,084 | 0.915 | 71 |
| email | 27,348 | 0.316 | 31 |
| phone | 24,349 | 0.388 | 12 |
| card4+zip | 20,870 | 0.397 | 9 |
| zip+last3 ∪ soundex+initial+zip ∪ email ∪ phone ∪ card4+zip | 663,852 | 0.990 | 473 |
| soundex+initial+zip ∪ email ∪ phone ∪ card4+zip | 109,560 | 0.981 | 71 |

True matching pairs: 50,876. The production union drops the skew-prone zip+last3 rule: it keeps nearly all of the recall with about 1/6 of the pairs.

## 2. Pairwise matching (held-out people)

| Model | Avg precision | Precision | Recall | F1 | Share of false positives that are same-household |
|---|---:|---:|---:|---:|---:|
| Champion: Fellegi-Sunter (unsupervised) | 0.961 ± 0.002 | 0.943 ± 0.005 | 0.879 ± 0.002 | 0.910 ± 0.003 | 0.52 ± 0.05 |
| Challenger: LightGBM (~2,000 labelled pairs) | 0.994 ± 0.001 | 0.938 ± 0.009 | 0.988 ± 0.003 | 0.962 ± 0.003 | 0.06 ± 0.01 |

Thresholds were chosen to maximise F1 on reviewed pairs from the training people only. The Fellegi-Sunter posterior is badly calibrated (Brier 0.124 ± 0.008 vs. 0.035 ± 0.001 for LightGBM) because household members agree on surname, address and phone together, which breaks its independence assumption; the champion therefore uses the match weight with a reviewed-pair threshold, not the posterior.

## 3. Clustering into identities (B-cubed, held-out people)

| Model | Clustering | B³ precision | B³ recall | B³ F1 | Over-merged clusters | Largest cluster |
|---|---|---:|---:|---:|---:|---:|
| Champion | Connected components | 0.943 ± 0.002 | 0.956 ± 0.002 | 0.949 ± 0.001 | 463 ± 18 | 33 ± 7 |
| Champion | Constrained (cannot-link + size cap) | 0.990 ± 0.001 | 0.922 ± 0.002 | 0.955 ± 0.001 | 110 ± 6 | 10 ± 1 |
| Challenger | Connected components | 0.926 ± 0.006 | 0.992 ± 0.002 | 0.958 ± 0.002 | 415 ± 5 | 57 ± 10 |
| Challenger | Constrained (cannot-link + size cap) | 0.990 ± 0.001 | 0.973 ± 0.001 | 0.981 ± 0.001 | 122 ± 11 | 9 ± 1 |

## 4. Champion / challenger decision

Rule: promote only if B³ F1 rises by at least 0.005 **and** over-merged clusters do not increase (merging two different people is the costlier error for marketing activation).

| Seed | B³ F1 gain | Change in over-merged clusters | Promoted? |
|---|---:|---:|---|
| 0 | +0.027 | +13 | no |
| 1 | +0.025 | +3 | no |
| 2 | +0.027 | +21 | no |

The challenger finds clearly more of the true links, but it also merges a few more distinct people, so the gate keeps the champion. Whether that trade is worth it is a business decision; the gate makes it explicit instead of letting a higher headline F1 decide.

## 5. Drift and data quality on a later batch (seed 0)

A simulated later batch adds a new `kiosk` source (no email, mostly no phone, surname only).

| Feature | PSI | Alert (> 0.25) |
|---|---:|---|
| source | 3.539 | yes |
| email_domain | 0.257 | yes |
| has_email | 0.256 | yes |
| has_phone | 0.105 | no |
| name_tokens | 3.509 | yes |

Failed (warning-level) expectations: `source_in_known_set` (unknown=['kiosk']); `null_rates_within_tolerance` (increases={'email': 0.249, 'phone': 0.16, 'card_last4': 0.212}). No critical expectation failed, so the batch is not blocked; it is flagged for review.

## 6. Spark blocking and key skew (zip + first 3 letters of surname)

Local Spark (`local[*]`, 2 cores, 16 shuffle partitions, AQE and broadcast joins off so skew is visible).

| Records | Strategy | Candidate pairs | Pair completeness | Largest task (rows) | Largest / median task |
|---:|---|---:|---:|---:|---:|
| 49,521 | naive | 644,453 | 0.920 | 149,284 | 5.4 |
| 49,521 | capped | 290,174 | 0.858 | 25,202 | 1.4 |
| 49,521 | salted | 644,453 | 0.920 | 63,442 | 1.7 |
| 249,670 | naive | 15,657,424 | 0.923 | 4,016,915 | 6.5 |
| 249,670 | capped | 2,104,461 | 0.718 | 159,223 | 1.2 |
| 249,670 | salted | 15,657,424 | 0.923 | 1,506,129 | 1.7 |

Salting splits each block larger than 100 records across 8 tasks: the same pairs and recall, with a much smaller largest task. Capping removes the skew by dropping big blocks, at a large recall cost for this rule (other rules must catch those pairs). On 2 local cores wall-clock time did not improve (salting adds replication work); the benefit is a balanced largest task, which matters when a cluster has more executors than the skewed key can use.
