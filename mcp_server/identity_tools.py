"""
Plain-Python tool functions behind the MCP server (kept separate so they are unit-testable
without an MCP client). They read only the PUBLISHED identity table, so the server can never
return raw email, phone, address or card data - that is the data contract.
"""
import json
import os

import pandas as pd

from irl.pipeline import _h
from irl.standardize import NICK, clean_email, clean_phone

ROOT = os.environ.get("IRL_DATA_ROOT", "data")
MAX_ROWS = 25


class IdentityStore:
    def __init__(self, root=ROOT, run=None):
        run = run or open(os.path.join(root, "LATEST")).read().strip()
        self.run = run
        self.t = pd.read_parquet(os.path.join(root, f"run={run}", "identities.parquet"))
        with open(os.path.join(root, f"run={run}", "match_stats.json")) as fh:
            self.stats = json.load(fh)

    def _summ(self, rows):
        out = []
        for iid, g in rows.groupby("identity_id"):
            full = self.t[self.t["identity_id"] == iid]
            out.append({"identity_id": iid, "matching_records": int(len(g)), "identity_records": int(len(full)),
                        "sources": sorted(full["source"].unique()),
                        "names": sorted(full["name_display"].dropna().unique())[:5]})
        return out[:MAX_ROWS]

    def lookup_by_email(self, email):
        e = clean_email(email)
        if e is None:
            return {"error": "not a valid email address"}
        return {"run": self.run, "matches": self._summ(self.t[self.t["email_hash"] == _h(e)])}

    def lookup_by_phone(self, phone):
        p = clean_phone(phone)
        if p is None:
            return {"error": "phone must have 10 digits"}
        return {"run": self.run, "matches": self._summ(self.t[self.t["phone_hash"] == _h(p)])}

    def lookup_by_name_zip(self, first_name, last_name, zip5):
        f = first_name.strip().lower()
        key = f"{NICK.get(f, f)}|{last_name.strip().lower()}"
        rows = self.t[(self.t["name_key"] == key) & (self.t["zip5"] == str(zip5)[:5])]
        return {"run": self.run, "matches": self._summ(rows)}

    def get_identity(self, identity_id):
        g = self.t[self.t["identity_id"] == identity_id]
        if g.empty:
            return {"error": f"unknown identity_id {identity_id}"}
        cols = ["record_id", "source", "name_display", "zip5", "email_masked", "phone_masked"]
        return {"run": self.run, "identity_id": identity_id,
                "records": g[cols].head(MAX_ROWS).astype(object).where(g[cols].notna().head(MAX_ROWS), None)
                .to_dict(orient="records")}

    def match_stats(self):
        return self.stats
