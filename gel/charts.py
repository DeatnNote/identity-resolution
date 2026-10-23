"""Static figures for the README (matplotlib). Palette: reference categorical slots 1-3 + neutral ink."""
import json
import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "axes.titlelocation": "left", "axes.titlecolor": INK})


def novelty(r, path):
    d = r["novelty"]["by_day"]
    x = [p["day"] for p in d]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.fill_between(x, [p["ci_low"] * 100 for p in d], [p["ci_high"] * 100 for p in d], color=BLUE, alpha=0.15, lw=0)
    ax.plot(x, [p["effect"] * 100 for p in d], color=BLUE, lw=2, label="Measured (95% CI)")
    ax.plot(x, [p["true_effect"] * 100 for p in d], color=INK, lw=1.2, ls="--", label="True effect")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Days since signup")
    ax.set_ylabel("Daily active rate lift (pp)")
    ax.set_title("onboarding_v2: most of the lift is novelty and fades within a week")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def srm(r, path):
    m = r["experiments"]["share_button"]["metrics"]
    names = ["activated", "d1_retained", "w4_retained", "converted_28d"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for k, (key, col, lab, off) in enumerate((("exposure_based", ORANGE, "Client exposure log (SRM p < 1e-8)", 0.15),
                                               ("adjusted", BLUE, "Server-side assignment", -0.15))):
        for i, n in enumerate(names):
            e = m[n][key]
            ax.errorbar(e["estimate"] * 100, i + off, xerr=[[(e["estimate"] - e["ci_low"]) * 100], [(e["ci_high"] - e["estimate"]) * 100]],
                        fmt="o", color=col, ms=6, capsize=0, lw=2, label=lab if i == 0 else None)
    ax.axvline(0, color=INK, lw=1, ls="--")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Treatment - control (pp); true effect is 0 for every metric")
    ax.set_title("share_button (true effect 0): the buggy exposure log shows fake wins")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def price_segments(r, path):
    seg = r["price_test"]["segments"]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for i, s in enumerate(seg):
        col = ORANGE if s["bh_significant"] else BLUE
        ax.errorbar(s["estimate"], i, xerr=[[s["estimate"] - s["ci_low"]], [s["ci_high"] - s["estimate"]]], fmt="o",
                    color=col, ms=6, lw=2, capsize=0)
        if s["true_effect"] is not None:
            ax.plot(s["true_effect"], i, marker="|", color=INK, ms=14, mew=2)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(seg)), [f"{s['segment']} (n={s['n']:,})" for s in seg])
    ax.invert_yaxis()
    ax.set_xlabel(r"28-day revenue per exposed user, \$15 minus \$12 (USD, 95% CI)")
    ax.set_title("Price test by segment\norange = significant after BH correction; black tick = true effect", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    r = json.load(open("results/study.json"))
    os.makedirs("results/figures", exist_ok=True)
    novelty(r, "results/figures/novelty.png")
    srm(r, "results/figures/srm_bias.png")
    price_segments(r, "results/figures/price_segments.png")
    print("figures written")


if __name__ == "__main__":
    main()
