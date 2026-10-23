"""
Turn scored pairs into identities.

connected_components: every pair above the threshold is an edge; each connected component is one
identity. Simple and transitive - which is exactly its weakness: one wrong edge between two
households merges them, and chains of borderline edges can snowball into a giant cluster.

constrained_components: the same, but components are split back apart when they contain a
"cannot-link" conflict (two different non-null birth years, or two clearly different canonical
first names) or are implausibly large. Conflicting components are re-clustered with a stricter
threshold until they are clean.
"""
import numpy as np
import pandas as pd
import jellyfish


class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        p = self.p
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def connected_components(n, pairs, scores, threshold):
    uf = UnionFind(n)
    sel = scores >= threshold
    for a, b in zip(pairs.i_l.to_numpy()[sel], pairs.i_r.to_numpy()[sel]):
        uf.union(a, b)
    return np.array([uf.find(i) for i in range(n)])


def _conflict(members, df):
    by = df["birth_year"].to_numpy()[members]
    by = by[~pd.isna(by)]
    if len(set(by)) > 1:
        return True
    fn = [f for f in df["first_name"].to_numpy()[members] if isinstance(f, str) and len(f) > 1]
    fn = sorted(set(fn))
    for i in range(len(fn)):
        for j in range(i + 1, len(fn)):
            if jellyfish.jaro_winkler_similarity(fn[i], fn[j]) < 0.75:
                return True
    return False


def constrained_components(df, pairs, scores, threshold, max_size=12, step=0.03, max_threshold=0.999):
    n = len(df)
    labels = connected_components(n, pairs, scores, threshold)
    i_l, i_r = pairs.i_l.to_numpy(), pairs.i_r.to_numpy()
    for _ in range(12):
        comp = pd.Series(np.arange(n)).groupby(labels).apply(lambda s: s.to_numpy())
        bad = [m for m in comp if len(m) > 1 and (len(m) > max_size or _conflict(m, df))]
        if not bad or threshold >= max_threshold:
            break
        threshold = min(threshold + step, max_threshold)
        bad_set = np.zeros(n, bool)
        for m in bad:
            bad_set[m] = True
        # re-cluster only the members of conflicting components, with a stricter threshold
        keep = bad_set[i_l] & bad_set[i_r] & (scores >= threshold)
        uf = UnionFind(n)
        for a, b in zip(i_l[keep], i_r[keep]):
            uf.union(a, b)
        new = np.array([uf.find(i) for i in range(n)])
        labels = np.where(bad_set, n + new, labels)          # offset keeps label spaces separate
    return labels


def bcubed(labels, truth):
    """B-cubed precision/recall/F1 over records (standard entity-resolution cluster metric)."""
    df = pd.DataFrame({"c": labels, "t": truth})
    both = df.groupby(["c", "t"]).size().rename("n").reset_index()
    csize = df.groupby("c").size()
    tsize = df.groupby("t").size()
    both["p"] = both["n"] / both["c"].map(csize)
    both["r"] = both["n"] / both["t"].map(tsize)
    P = float((both["n"] * both["p"]).sum() / len(df))
    R = float((both["n"] * both["r"]).sum() / len(df))
    return {"bcubed_precision": P, "bcubed_recall": R, "bcubed_f1": 2 * P * R / (P + R),
            "n_clusters": int(df["c"].nunique()), "n_true": int(df["t"].nunique()),
            "largest_cluster": int(csize.max()),
            "overmerged_clusters": int((df.groupby("c")["t"].nunique() > 1).sum())}
