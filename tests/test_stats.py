import numpy as np

from gel import stats as S


def test_srm_detects_imbalance():
    assert S.srm_test(5000, 5000)["p_value"] > 0.9
    assert S.srm_test(5000, 4700)["p_value"] < 0.01


def test_diff_ci_and_cuped_variance_reduction():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 20000)
    t = rng.random(20000) < 0.5
    y = 2 * x + 0.3 * t + rng.normal(0, 1, 20000)
    raw, cu = S.diff(y, t), S.cuped(y, t, x)
    assert raw["ci_low"] < 0.3 < raw["ci_high"]
    assert cu["se"] < 0.5 * raw["se"] and abs(cu["estimate"] - 0.3) < 0.05


def test_delta_method_matches_bootstrap():
    rng = np.random.default_rng(1)
    n = 4000
    den = rng.integers(1, 8, n).astype(float)
    num = rng.poisson(den * rng.gamma(2, 1.5, n))
    t = rng.random(n) < 0.5
    se = S.ratio_delta(num, den, t)["se"]
    boots = []
    for _ in range(400):
        i = rng.integers(0, n, n)
        tt, nn, dd = t[i], num[i], den[i]
        boots.append(nn[tt].sum() / dd[tt].sum() - nn[~tt].sum() / dd[~tt].sum())
    assert abs(se / np.std(boots) - 1) < 0.15


def test_benjamini_hochberg():
    sig, padj = S.benjamini_hochberg([0.001, 0.01, 0.035, 0.2, 0.9])
    assert sig.tolist() == [True, True, False, False, False]
    assert np.all(np.diff(padj[np.argsort([0.001, 0.01, 0.035, 0.2, 0.9])]) >= 0)


def test_mde_and_sample_size_are_inverse():
    n = S.n_per_arm_for(1.0, 0.05)
    assert abs(S.mde(1.0, n) - 0.05) < 0.001
