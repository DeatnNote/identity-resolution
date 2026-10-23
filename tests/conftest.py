import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def small_world(tmp_path_factory):
    from gel.sim import make_users, simulate, to_tables
    u = make_users(n=4000, seed=3)
    logs, outcomes = simulate(u, seed=3, horizon=90)
    root = tmp_path_factory.mktemp("wh")
    os.makedirs(root / "data")
    for name, df in to_tables(u, logs).items():
        df.to_parquet(root / "data" / f"{name}.parquet", index=False)
    return u, logs, outcomes, root
