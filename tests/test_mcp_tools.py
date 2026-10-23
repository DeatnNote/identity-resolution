import pandas as pd

from irl import pipeline
from mcp_server.identity_tools import IdentityStore


def test_mcp_tools_return_masked_data_only(tmp_path):
    ds, root = "2026-01-02", str(tmp_path)
    for name, fn in pipeline.TASKS:
        kw = {"n_people": 800} if name == "ingest" else {"threshold": 13.5} if name == "cluster" else {}
        fn(ds, root=root, **kw)
    raw = pd.read_parquet(f"{root}/run={ds}/raw.parquet").dropna(subset=["email", "phone"])
    s = IdentityStore(root=root)
    email = raw["email"].iloc[0]
    hit = s.lookup_by_email(email.upper())
    assert hit["matches"], "normalised email lookup should find the record"
    ident = s.get_identity(hit["matches"][0]["identity_id"])
    text = str(ident)
    assert email.lower() not in text.lower()
    assert raw["phone"].iloc[0] not in text
    assert s.lookup_by_phone("123")["error"]
    assert s.match_stats()["records"] == len(pd.read_parquet(f"{root}/run={ds}/raw.parquet"))
