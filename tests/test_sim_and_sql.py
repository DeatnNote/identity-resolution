import numpy as np
import pandas as pd

from gel.metrics import price_metrics, true_effects, user_metrics
from gel.sim import make_users
from gel.warehouse import connect, named_queries, query


def test_users_deterministic():
    pd.testing.assert_frame_equal(make_users(500, 7), make_users(500, 7))


def test_common_random_numbers_isolate_each_experiment(small_world):
    u, _, out, _ = small_world
    # users outside share_button have identical outcomes in its two worlds
    outside = u.share_button.to_numpy() < 0
    a1, a0 = out["share_button=1"]["active_t"], out["share_button=0"]["active_t"]
    assert np.array_equal(a1[outside], a0[outside])
    # share_button has no effect at all, so its worlds are identical for everyone
    assert np.array_equal(a1, a0)
    t = true_effects(u, out)
    assert all(abs(v["true_effect"]) < 1e-12 for k, v in t["share_button"].items() if isinstance(v, dict) and "true_effect" in v)
    assert t["onboarding_v2"]["daily_active_effect_by_day"][1] > t["onboarding_v2"]["daily_active_effect_by_day"][20]


def test_sql_metrics_match_python_definitions(small_world):
    u, _, out, root = small_world
    con = connect(root)
    sq = con.execute("SELECT * FROM user_metrics ORDER BY user_id").df()
    py = pd.DataFrame(user_metrics(out["obs"]))
    py["user_id"] = np.arange(len(py))
    py = py[py.user_id.isin(sq.user_id)].reset_index(drop=True)
    assert len(sq) == len(py) > 1000
    for m in ("activated", "d1_retained", "w4_retained", "generations_7d", "converted_28d"):
        assert np.allclose(sq[m].to_numpy(float), py[m].to_numpy(float)), m
    assert np.allclose(sq.revenue_28d, py.revenue_28d)


def test_sql_price_metrics_match_python(small_world):
    u, _, out, root = small_world
    con = connect(root)
    sq = con.execute("SELECT * FROM price_metrics ORDER BY user_id").df()
    pm = price_metrics(out["obs"])
    ids = sq.user_id.to_numpy()
    assert len(sq) > 50
    assert np.allclose(sq.revenue_28d, pm["revenue_28d"][ids])
    assert np.allclose(sq.converted_28d, pm["converted_28d"][ids])


def test_dashboard_queries_run_and_mrr_reconciles(small_world):
    _, _, _, root = small_world
    con = connect(root)
    for name in named_queries():
        assert len(query(con, name)) > 0, name
    gap = query(con, "mrr_reconciliation")["gap"].iloc[0]
    assert abs(gap) < 1e-6


def test_exposure_log_has_the_planted_srm(small_world):
    _, _, _, root = small_world
    con = connect(root)
    df = con.execute("""SELECT variant, COUNT(*) n FROM exposures WHERE experiment='share_button' GROUP BY 1""").df()
    n = dict(zip(df.variant, df.n))
    assert n["treatment"] < n["control"]
