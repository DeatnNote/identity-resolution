import numpy as np
import pandas as pd

from irl.blocking import PRODUCTION_RULES, candidate_pairs, true_pairs_count
from irl.cluster import bcubed, connected_components
from irl.generate import generate
from irl.blocking import pairs_for_rule
from irl.match import FellegiSunter, compare, levels, random_pairs
from irl.standardize import standardize


def small(seed=0):
    return standardize(generate(1500, seed)[0])


def test_generation_is_deterministic():
    a, b = generate(300, 5)[0], generate(300, 5)[0]
    pd.testing.assert_frame_equal(a, b)


def test_blocking_keeps_most_true_pairs():
    d = small()
    cp = candidate_pairs(d, PRODUCTION_RULES)
    pid = d.person_id.to_numpy()
    found = (pid[cp.i_l] == pid[cp.i_r]).sum()
    assert found / true_pairs_count(d.person_id) > 0.95
    assert (cp.i_l < cp.i_r).all() and not cp.duplicated().any()


def test_fellegi_sunter_weights_have_the_right_sign_and_rank_well():
    from sklearn.metrics import average_precision_score
    d = small()
    cp = candidate_pairs(d, PRODUCTION_RULES)
    lv = levels(compare(d, cp))
    rand = levels(compare(d, random_pairs(len(d), 30000, 0)))
    sessions = {r: levels(compare(d, pairs_for_rule(d, r))) for r in ("email", "phone", "soundex+initial+zip")}
    fs = FellegiSunter().fit_sessions(sessions, rand, lv)
    t = fs.table().set_index(["field", "level"])["weight"]
    assert t[("email", 3)] > 5 and t[("phone", 2)] > 3       # exact email / phone agreement: strong evidence
    assert t[("initial", 1)] < 0                              # different first initial: evidence against
    pid = d.person_id.to_numpy()
    y = pid[cp.i_l] == pid[cp.i_r]
    assert average_precision_score(y, fs.match_weight(lv)) > 0.85


def test_bcubed_perfect_and_overmerged():
    truth = np.array([0, 0, 1, 1])
    assert bcubed(truth, truth)["bcubed_f1"] == 1.0
    merged = bcubed(np.zeros(4, int), truth)
    assert merged["bcubed_precision"] == 0.5 and merged["overmerged_clusters"] == 1


def test_connected_components_threshold():
    pairs = pd.DataFrame({"i_l": [0, 1, 3], "i_r": [1, 2, 4]})
    lab = connected_components(5, pairs, np.array([0.9, 0.2, 0.8]), 0.5)
    assert lab[0] == lab[1] != lab[2] and lab[3] == lab[4]
