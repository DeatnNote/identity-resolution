"""
Data-quality expectations and drift checks (Great-Expectations style, dependency-free).

Each expectation returns a result dict; `run_expectations` raises if any CRITICAL expectation
fails, so an orchestrator task fails loudly instead of passing bad data downstream.
"""
import numpy as np

EXPECTED_COLUMNS = ["record_id", "source", "full_name", "email", "phone", "address", "zip",
                    "card_last4", "birth_year"]
KNOWN_SOURCES = {"pos", "online", "loyalty"}


class DataQualityError(RuntimeError):
    pass


def _res(name, ok, critical, detail):
    return {"expectation": name, "passed": bool(ok), "critical": critical, "detail": detail}


def expectations(df, baseline_null_rates=None, null_tolerance=0.10):
    r = []
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    r.append(_res("schema_has_expected_columns", not missing, True, f"missing={missing}"))
    if missing:
        return r
    r.append(_res("record_id_unique", df["record_id"].is_unique, True,
                  f"duplicates={int(df['record_id'].duplicated().sum())}"))
    r.append(_res("record_id_not_null", df["record_id"].notna().all(), True, ""))
    z = df["zip"].astype(str).str.fullmatch(r"\d{5}")
    r.append(_res("zip_is_5_digits", z.mean() >= 0.99, True, f"share_valid={z.mean():.3f}"))
    unknown = set(df["source"].dropna().unique()) - KNOWN_SOURCES
    r.append(_res("source_in_known_set", not unknown, False, f"unknown={sorted(map(str, unknown))}"))
    em = df["email"].dropna().astype(str)
    ok = em.str.fullmatch(r"[^@\s]+@[^@\s]+\.[a-z]{2,}").mean() if len(em) else 1.0
    r.append(_res("email_well_formed", ok >= 0.99, False, f"share_valid={ok:.3f}"))
    ph = df["phone"].dropna().astype(str).str.replace(r"\D", "", regex=True)
    ok = ph.str.len().isin([10, 11]).mean() if len(ph) else 1.0
    r.append(_res("phone_has_10_digits", ok >= 0.99, False, f"share_valid={ok:.3f}"))
    if baseline_null_rates is not None:
        cur = df[EXPECTED_COLUMNS].isna().mean()
        jumps = {c: round(float(cur[c] - baseline_null_rates[c]), 3) for c in EXPECTED_COLUMNS
                 if cur[c] - baseline_null_rates[c] > null_tolerance}
        r.append(_res("null_rates_within_tolerance", not jumps, False, f"increases={jumps}"))
    return r


def run_expectations(df, **kw):
    res = expectations(df, **kw)
    failed = [x for x in res if not x["passed"] and x["critical"]]
    if failed:
        raise DataQualityError("; ".join(f"{x['expectation']} ({x['detail']})" for x in failed))
    return res


def psi(expected, actual, eps=1e-4):
    """Population stability index between two categorical distributions (dict or Series of shares)."""
    keys = set(expected) | set(actual)
    e = np.array([max(expected.get(k, 0), eps) for k in keys])
    a = np.array([max(actual.get(k, 0), eps) for k in keys])
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(base, new):
    """PSI on the distributions matching depends on. >0.25 is a common 'investigate' threshold."""
    def shares(s):
        return s.value_counts(normalize=True, dropna=False).rename(lambda k: str(k)).to_dict()
    feats = {
        "source": lambda d: d["source"],
        "email_domain": lambda d: d["email"].str.split("@").str[-1],
        "has_email": lambda d: d["email"].notna(),
        "has_phone": lambda d: d["phone"].notna(),
        "name_tokens": lambda d: d["full_name"].str.split().str.len().clip(upper=3),
    }
    out = {k: psi(shares(f(base)), shares(f(new))) for k, f in feats.items()}
    return {k: {"psi": round(v, 3), "alert": v > 0.25} for k, v in out.items()}
