"""
Pairwise comparison and two matchers.

  champion   Fellegi-Sunter model fitted by EM: unsupervised, no labels needed. Each field is
             compared into discrete agreement levels; EM estimates, per field and level, how
             often it occurs among matches (m) and non-matches (u). The match weight is the sum
             of log(m/u). This is the classic record-linkage model used by tools like Splink.
  challenger LightGBM classifier on continuous comparison features, trained on a small set of
             clerically reviewed pairs (here: labels taken from the simulation's truth).
"""
import jellyfish
import numpy as np
import pandas as pd


def _jw(a, b):
    return np.array([jellyfish.jaro_winkler_similarity(x, y) if isinstance(x, str) and isinstance(y, str)
                     else np.nan for x, y in zip(a, b)])


def _eq(a, b):
    a, b = pd.Series(a).to_numpy(object), pd.Series(b).to_numpy(object)
    both = pd.notna(a) & pd.notna(b)
    out = np.full(len(a), np.nan)
    out[both] = (a[both] == b[both]).astype(float)
    return out


def _jaccard(a, b):
    out = np.full(len(a), np.nan)
    for i, (x, y) in enumerate(zip(a, b)):
        if isinstance(x, str) and isinstance(y, str):
            sx, sy = set(x.split()), set(y.split())
            out[i] = len(sx & sy) / len(sx | sy)
    return out


def compare(df, pairs):
    """Continuous comparison features for candidate pairs (NaN = at least one side missing)."""
    L = df.iloc[pairs.i_l.to_numpy()].reset_index(drop=True)
    R = df.iloc[pairs.i_r.to_numpy()].reset_index(drop=True)
    f = pd.DataFrame({"i_l": pairs.i_l.to_numpy(), "i_r": pairs.i_r.to_numpy()})
    f["first_jw"] = _jw(L.first_name, R.first_name)
    f["initial_eq"] = _eq(L.first_initial, R.first_initial)
    f["last_jw"] = _jw(L.last_name, R.last_name)
    f["email_eq"] = _eq(L.email_clean, R.email_clean)
    f["email_local_jw"] = _jw(L.email_clean.str.split("@").str[0], R.email_clean.str.split("@").str[0])
    f["phone_eq"] = _eq(L.phone_clean, R.phone_clean)
    f["zip_eq"] = _eq(L.zip5, R.zip5)
    f["addr_jacc"] = _jaccard(L.address_clean, R.address_clean)
    f["card_eq"] = _eq(L.card_last4, R.card_last4)
    f["birth_eq"] = _eq(L.birth_year, R.birth_year)
    return f


FEATURES = ["first_jw", "initial_eq", "last_jw", "email_eq", "email_local_jw", "phone_eq", "zip_eq",
            "addr_jacc", "card_eq", "birth_eq"]


def levels(f):
    """Discrete agreement levels for Fellegi-Sunter (0 = missing, then disagree ... exact)."""
    def cut(x, edges):
        out = np.zeros(len(x), dtype=int)
        ok = ~np.isnan(x)
        out[ok] = 1 + np.digitize(x[ok], edges)
        return out
    return pd.DataFrame({
        "first": cut(f.first_jw.to_numpy(), [0.80, 0.92, 0.999]),
        "initial": cut(f.initial_eq.to_numpy(), [0.5]),
        "last": cut(f.last_jw.to_numpy(), [0.80, 0.92, 0.999]),
        "email": np.where(f.email_eq.to_numpy() == 1, 3, cut(f.email_local_jw.to_numpy(), [0.9])),
        "phone": cut(f.phone_eq.to_numpy(), [0.5]),
        "zip": cut(f.zip_eq.to_numpy(), [0.5]),
        "addr": cut(f.addr_jacc.to_numpy(), [0.5, 0.999]),
        "card": cut(f.card_eq.to_numpy(), [0.5]),
        "birth": cut(f.birth_eq.to_numpy(), [0.5]),
    })


def random_pairs(n, k, seed=0):
    """Uniformly random record pairs - almost all non-matches - used to estimate u."""
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, n, k), rng.integers(0, n, k)
    keep = a != b
    return pd.DataFrame({"i_l": np.minimum(a, b)[keep], "i_r": np.maximum(a, b)[keep]})


class FellegiSunter:
    """
    EM for the Fellegi-Sunter model. Level 0 (missing) is neutral: m = u, weight 0.

    If `lv_random` (comparison levels of random record pairs) is given, u is estimated from it
    and held fixed, and EM only learns m and the match rate. Without it, EM on blocked pairs
    alone tends to learn "same household" instead of "same person", because blocking has
    already made almost every candidate pair agree on surname and zip.
    """

    def __init__(self, n_iter=60, init_lambda=0.02):
        self.n_iter, self.lam = n_iter, init_lambda

    def fit(self, lv, lv_random=None):
        X = lv.to_numpy()
        self.cols = list(lv.columns)
        self.k = [int(X[:, j].max()) + 1 for j in range(X.shape[1])]
        # initial guesses: matches concentrate on the highest level, non-matches on the lowest
        m = [np.linspace(0.1, 1.0, k) ** 3 for k in self.k]
        u = [np.linspace(1.0, 0.1, k) ** 3 for k in self.k]
        m = [v / v.sum() for v in m]
        u = [v / v.sum() for v in u]
        fixed_u = lv_random is not None
        if fixed_u:
            Xr = lv_random.to_numpy()
            for j in range(X.shape[1]):
                obs = Xr[:, j] > 0
                cu = np.bincount(Xr[obs, j], minlength=self.k[j]).astype(float)[:self.k[j]] + 1.0
                cu[0] = 1e-3
                u[j] = cu / cu.sum()
        lam = self.lam
        for _ in range(self.n_iter):
            lm = np.log(lam) + sum(np.log(m[j][X[:, j]]) * (X[:, j] > 0) for j in range(X.shape[1]))
            lu = np.log(1 - lam) + sum(np.log(u[j][X[:, j]]) * (X[:, j] > 0) for j in range(X.shape[1]))
            g = 1 / (1 + np.exp(lu - lm))
            lam = float(np.clip(g.mean(), 1e-6, 0.5))
            for j in range(X.shape[1]):
                obs = X[:, j] > 0
                cm = np.bincount(X[obs, j], weights=g[obs], minlength=self.k[j]) + 1e-3
                cu = np.bincount(X[obs, j], weights=1 - g[obs], minlength=self.k[j]) + 1e-3
                cm[0] = cu[0] = 1e-3
                m[j] = cm / cm.sum()
                if not fixed_u:
                    u[j] = cu / cu.sum()
        self.m, self.u, self.lam = m, u, lam
        return self

    # fields that a blocking rule forces to agree, so they carry no information inside that session
    SESSION_EXCLUDES = {"email": {"email"}, "phone": {"phone"},
                        "soundex+initial+zip": {"initial", "last", "zip"}}

    def fit_sessions(self, session_levels, lv_random, lv_candidates):
        """
        Splink-style training. u comes from random pairs. For each blocking rule, EM runs on the
        pairs that rule produces, using and updating ONLY the fields the rule does not force to
        agree. m for each field is the average over the sessions that estimated it. Finally the
        match rate among candidate pairs is re-estimated with m and u held fixed.
        """
        self.cols = list(lv_random.columns)
        allx = [lv_random.to_numpy()] + [v.to_numpy() for v in session_levels.values()]
        self.k = [int(max(x[:, j].max() for x in allx)) + 1 for j in range(len(self.cols))]
        Xr = lv_random.to_numpy()
        u = []
        for j in range(len(self.cols)):
            obs = Xr[:, j] > 0
            cu = np.bincount(Xr[obs, j], minlength=self.k[j]).astype(float)[:self.k[j]] + 1.0
            cu[0] = 1e-3
            u.append(cu / cu.sum())
        m_est = {c: [] for c in self.cols}
        for rule, lv in session_levels.items():
            X = lv.to_numpy()
            use = [j for j, c in enumerate(self.cols) if c not in self.SESSION_EXCLUDES[rule]]
            m = [np.linspace(0.1, 1.0, k) ** 3 / (np.linspace(0.1, 1.0, k) ** 3).sum() for k in self.k]
            lam = 0.3
            for _ in range(self.n_iter):
                lm = np.log(lam) + sum(np.log(m[j][X[:, j]]) * (X[:, j] > 0) for j in use)
                lu = np.log(1 - lam) + sum(np.log(u[j][X[:, j]]) * (X[:, j] > 0) for j in use)
                g = 1 / (1 + np.exp(np.clip(lu - lm, -50, 50)))
                lam = float(np.clip(g.mean(), 1e-6, 0.999))
                for j in use:
                    obs = X[:, j] > 0
                    cm = np.bincount(X[obs, j], weights=g[obs], minlength=self.k[j])[:self.k[j]] + 1e-3
                    cm[0] = 1e-3
                    m[j] = cm / cm.sum()
            for j in use:
                m_est[self.cols[j]].append(m[j])
        self.m = [np.mean(m_est[c], axis=0) if m_est[c] else u[j] for j, c in enumerate(self.cols)]
        self.u = u
        # Prior: probability that two RANDOM records match, estimated without labels from a
        # deterministic rule (same cleaned email) and an assumed rule recall of 50%.
        self.lam = 0.5   # placeholder; set by set_prior()
        return self

    def set_prior(self, n_records, n_rule_pairs, rule_recall=0.5):
        total = n_records * (n_records - 1) / 2
        self.lam = float(np.clip(n_rule_pairs / rule_recall / total, 1e-9, 0.5))
        return self

    def match_weight(self, lv):
        X = lv.to_numpy()
        return sum(np.log(self.m[j][X[:, j]] / self.u[j][X[:, j]]) * (X[:, j] > 0) for j in range(X.shape[1]))

    def predict_proba(self, lv):
        """Posterior under conditional independence. NOT well calibrated when fields are
        correlated (household members agree on surname, address and phone together)."""
        w = self.match_weight(lv) + np.log(self.lam / (1 - self.lam))
        return 1 / (1 + np.exp(-np.clip(w, -50, 50)))

    def table(self):
        rows = []
        for j, c in enumerate(self.cols):
            for lvl in range(1, self.k[j]):
                rows.append({"field": c, "level": lvl, "m": self.m[j][lvl], "u": self.u[j][lvl],
                             "weight": np.log(self.m[j][lvl] / self.u[j][lvl])})
        return pd.DataFrame(rows)


class LGBMMatcher:
    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, f, y):
        import lightgbm as lgb
        self.m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                    min_child_samples=10, random_state=self.seed, verbose=-1)
        self.m.fit(f[FEATURES], y)
        return self

    def predict_proba(self, f):
        return self.m.predict_proba(f[FEATURES])[:, 1]
