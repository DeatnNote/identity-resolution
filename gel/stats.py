"""Experiment statistics: SRM, difference estimates, CUPED, delta method, BH, power."""
import numpy as np
from scipy import stats

Z = stats.norm.ppf(0.975)


def srm_test(n_control, n_treatment, expected_share=0.5):
    n = n_control + n_treatment
    exp = np.array([n * (1 - expected_share), n * expected_share])
    chi2 = float(((np.array([n_control, n_treatment]) - exp) ** 2 / exp).sum())
    return {"n_control": int(n_control), "n_treatment": int(n_treatment),
            "treatment_share": n_treatment / n, "chi2": chi2, "p_value": float(stats.chi2.sf(chi2, 1))}


def diff(y, t):
    """Difference in means (treatment - control) with Welch standard error."""
    y, t = np.asarray(y, float), np.asarray(t).astype(bool)
    a, b = y[t], y[~t]
    est = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return _res(est, se, b.mean())


def cuped(y, t, x):
    """CUPED: y - theta (x - mean x), theta from pooled data (x must be pre-treatment)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    theta = np.cov(y, x)[0, 1] / x.var(ddof=1)
    r = diff(y - theta * (x - x.mean()), t)
    r["theta"] = float(theta)
    r["control_mean"] = float(np.asarray(y)[~np.asarray(t).astype(bool)].mean())
    return r


def regression_adjusted(y, t, X):
    """OLS of y on treatment + centred covariates (Lin 2013), HC1 errors. X: n x k dummies/numbers."""
    import statsmodels.api as sm
    t = np.asarray(t, float)
    Xc = X - X.mean(0)
    D = np.column_stack([np.ones(len(t)), t, Xc, Xc * t[:, None]])
    fit = sm.OLS(np.asarray(y, float), D).fit(cov_type="HC1")
    r = _res(fit.params[1], fit.bse[1], float(np.asarray(y)[t == 0].mean()))
    return r


def ratio_delta(num, den, t):
    """Ratio metric sum(num)/sum(den) per arm with delta-method SE (user is the randomisation unit)."""
    num, den, t = np.asarray(num, float), np.asarray(den, float), np.asarray(t).astype(bool)
    out = []
    for g in (t, ~t):
        n, d = num[g], den[g]
        mn, md = n.mean(), d.mean()
        r = mn / md
        var = (n.var(ddof=1) - 2 * r * np.cov(n, d)[0, 1] + r ** 2 * d.var(ddof=1)) / (md ** 2 * g.sum())
        out.append((r, var))
    est = out[0][0] - out[1][0]
    return _res(est, np.sqrt(out[0][1] + out[1][1]), out[1][0])


def ratio_naive(num, den, t):
    """WRONG on purpose: treats each user-day as independent (unit of analysis != unit of randomisation)."""
    num, den, t = np.asarray(num, float), np.asarray(den, float), np.asarray(t).astype(bool)
    days_t, days_c = den[t].sum(), den[~t].sum()
    rt, rc = num[t].sum() / days_t, num[~t].sum() / days_c
    # per-day variance approximated by Poisson-like dispersion of daily counts
    vt = (num[t] ** 2 / np.maximum(den[t], 1)).sum() / days_t - rt ** 2
    vc = (num[~t] ** 2 / np.maximum(den[~t], 1)).sum() / days_c - rc ** 2
    return _res(rt - rc, np.sqrt(vt / days_t + vc / days_c), rc)


def _res(est, se, base):
    p = 2 * stats.norm.sf(abs(est / se)) if se > 0 else 1.0
    return {"estimate": float(est), "se": float(se), "ci_low": float(est - Z * se), "ci_high": float(est + Z * se),
            "p_value": float(p), "control_mean": float(base),
            "relative": float(est / base) if base else float("nan")}


def benjamini_hochberg(p, q=0.05):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    thresh = q * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(m, bool)
    out[order[:k]] = True
    adj = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    p_adj = np.empty(m)
    p_adj[order] = np.minimum(adj, 1)
    return out, p_adj


def mde(sd, n_per_arm, alpha=0.05, power=0.8):
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * sd * np.sqrt(2 / n_per_arm))


def n_per_arm_for(sd, mde_abs, alpha=0.05, power=0.8):
    return int(np.ceil(2 * ((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * sd / mde_abs) ** 2))
