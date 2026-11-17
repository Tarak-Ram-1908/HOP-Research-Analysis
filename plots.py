"""Make every figure of the report/PPT from results/*.json -> figures/*.png (+ .pdf)."""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e7e6e2"
# one colour per entity, in every figure (reference categorical order, light mode)
COL = {"hop": "#2a78d6", "ours": "#eb6834", "ideal": "#1baf7a", "static20": "#eda100",
       "split": "#6250d6", "printed": "#e34948"}
LABEL = {"split": "Split, no FEC (ECF)", "static20": "Static FEC +20%", "hop": "HOP (JUMP+SKIP)",
         "ideal": "Ideal JUMP", "hop_relax": "HOP + graceful relax (ours)",
         "hop_printed": "HOP with eq. (5) as printed"}
SCOL = {"split": COL["split"], "static20": COL["static20"], "hop": COL["hop"],
        "ideal": COL["ideal"], "hop_relax": COL["ours"], "hop_printed": COL["printed"]}
MARK = {"split": "v", "static20": "s", "hop": "o", "ideal": "D", "hop_relax": "^",
        "hop_printed": "x"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "legend.fontsize": 8.5, "lines.linewidth": 2, "lines.markersize": 5,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def load(name):
    with open(f"results/{name}.json") as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png")
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)


def ci95(v):
    v = np.asarray(v, float)
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def target_line(ax, y=0.02, text="target 1 − P_thr = 2%"):
    ax.axhline(y, color=INK2, lw=1.2, ls="--", zorder=1)
    ax.text(ax.get_xlim()[0] + 1, y + 0.002, text, color=INK2, fontsize=8, va="bottom")


# =========================================================================== E1
def fig_e1():
    d = load("e1")
    agg = defaultdict(list)
    for r in d["rows"]:
        agg[(r["scen"], r["scheme"], r["B"])].append(r)
    sizes = [B for B in d["sizes"]]
    titles = {"balanced": "Balanced (40/40 Mb/s)", "unbalanced": "Unbalanced (32/48 Mb/s)",
              "highly_unbalanced": "Highly unbalanced (64/16 Mb/s)",
              "rtt_unbalanced": "RTT unbalanced (20/30 ms)"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    for ax, scen in zip(axes.flat, titles):
        x = np.array(sizes) / 1000
        bd = d["bounds"][scen]
        ax.plot(x, [bd[str(B)] if str(B) in bd else bd[B] for B in sizes], color=MUTED, lw=1.2,
                ls=":", label="Theoretical bound")
        for s in ("split", "static20", "hop", "ideal", "hop_relax"):
            ys = [np.mean([r["fail"] for r in agg[(scen, s, B)]]) for B in sizes]
            es = [ci95([r["fail"] for r in agg[(scen, s, B)]]) for B in sizes]
            ax.errorbar(x, ys, yerr=es, color=SCOL[s], marker=MARK[s], label=LABEL[s],
                        capsize=2, elinewidth=1, ls="--" if s == "ideal" else "-")
        ax.set_ylim(0, 0.12)
        ax.set_xlim(78, 202)
        ax.set_title(titles[scen], loc="left")
        target_line(ax)
    for ax in axes[1]:
        ax.set_xlabel("Block size (kB)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Late-block probability")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.03), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "e1_reproduction")

    # goodput, balanced + highly unbalanced
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, scen in zip(axes, ("balanced", "highly_unbalanced", "rtt_unbalanced")):
        x = np.array(sizes) / 1000
        ax.plot(x, x * 8 / 25, color=MUTED, lw=1.2, ls=":", label="Offered load")
        for s in ("split", "static20", "hop", "ideal", "hop_relax"):
            ys = [np.mean([r["goodput"] for r in agg[(scen, s, B)]]) for B in sizes]
            ax.plot(x, ys, color=SCOL[s], marker=MARK[s], label=LABEL[s],
                    ls="--" if s == "ideal" else "-")
        ax.set_title(titles[scen], loc="left")
        ax.set_xlabel("Block size (kB)")
    axes[0].set_ylabel("On-time goodput (Mb/s)")
    axes[0].legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    save(fig, "e1_goodput")

    # HOP vs eq.(5)-as-printed, balanced
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    x = np.array(sizes) / 1000
    for s in ("hop_printed", "hop", "ideal"):
        ys = [np.mean([r["fail"] for r in agg[("balanced", s, B)]]) for B in sizes]
        ax.plot(x, ys, color=SCOL[s], marker=MARK[s], label=LABEL[s],
                ls="--" if s == "ideal" else "-")
    ax.set_ylim(0, 0.3)
    ax.set_xlabel("Block size (kB)")
    ax.set_ylabel("Late-block probability")
    ax.set_title("Balanced scenario", loc="left")
    target_line(ax)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "e2_printed_sim")


# =========================================================================== E2
def fig_e2():
    d = load("e2")
    rows = [r for r in d["rows"] if r["paths"] == "40/40" and r["k"] == 97]
    red = [round(sum(r["sched"]) / r["k"] - 1, 2) * 100 for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    ax = axes[0]
    ax.plot(red, [r["mc"] for r in rows], color=INK2, marker="o", lw=3, alpha=0.35,
            label="Monte Carlo ground truth")
    ax.plot(red, [r["exact"] for r in rows], color=COL["ours"], marker="^",
            label="Exact ψ: λ(d+1) − λ(d)  (ours)")
    ax.plot(red, [r["printed"] for r in rows], color=COL["printed"], marker="x",
            label="Eq. (5) as printed")
    ax.set_xlabel("Redundancy added (% of k)")
    ax.set_ylabel("Block failure probability P_e")
    ax.set_title("P_e for a 145 kB block (k = 97), 40/40 Mb/s", loc="left")
    ax.legend()
    ax = axes[1]
    allr = d["rows"]
    sums = [max(r["psi_sum"]) for r in allr]
    ext = [sum(r["sched"]) / r["k"] - 1 for r in allr]
    ax.scatter(np.array(ext) * 100, sums, color=COL["printed"], s=22, zorder=3)
    ax.axhline(1.0, color=INK2, ls="--", lw=1.2)
    ax.set_ylim(0.75, None)
    ax.text(12, 0.84, "a probability distribution sums to 1", color=INK2, fontsize=8)
    ax.set_xlabel("Redundancy added (% of k)")
    ax.set_ylabel("Sum of eq. (5) terms over d")
    ax.set_title("Eq. (5) is not normalised", loc="left")
    fig.tight_layout()
    save(fig, "e2_eq5")


# =========================================================================== E3
def fig_e3():
    d = load("e3")
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for k, ls in ((40, ":"), (80, "--"), (120, "-")):
        rows = [r for r in d["rows"] if r["k"] == k]
        ms = [r["m"] for r in rows]
        ax.plot(ms, [r["t_conv"] * 1e3 for r in rows], color=COL["ours"], ls=ls, marker="^",
                label=f"Convolution (ours), k={k}")
        en = [(r["m"], r["t_enum"] * 1e3) for r in rows if r["t_enum"] is not None]
        ax.plot([a for a, _ in en], [b for _, b in en], color=COL["hop"], ls=ls, marker="o",
                label=f"Enumeration (paper), k={k}")
    ax.set_yscale("log")
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.set_xlabel("Number of paths m")
    ax.set_ylabel("Time per P_e evaluation (ms, log)")
    ax.set_title("Cost of one P_e evaluation", loc="left")
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    ax.set_ylim(1e-3, 1e5)
    fig.tight_layout()
    save(fig, "e3_complexity")


# =========================================================================== E4
def fig_e4():
    d = load("e4")
    agg = defaultdict(list)
    for r in d["rows"]:
        agg[(r["variant"], r["B"])].append(r)
    sizes = d["sizes"]
    show = [("HOP", COL["hop"], "o", "-"), ("+relax", COL["ours"], "^", "-"),
            ("+cap & relax (full)", COL["ours"], "s", "--"),
            ("Ideal JUMP", COL["ideal"], "D", "-"), ("Ideal + cap & relax", COL["ideal"], "^", "--")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
    x = np.array(sizes) / 1000
    for name, c, mk, ls in show:
        lab = {"+relax": "HOP + relax (ours)", "+cap & relax (full)": "HOP + exp. backlog cap + relax",
               "Ideal + cap & relax": "Ideal + cap + relax"}.get(name, name)
        axes[0].plot(x, [np.mean([r["fail"] for r in agg[(name, B)]]) for B in sizes],
                     color=c, marker=mk, ls=ls, label=lab)
        axes[1].plot(x, [np.mean([r["goodput"] for r in agg[(name, B)]]) for B in sizes],
                     color=c, marker=mk, ls=ls, label=lab)
    axes[0].set_ylabel("Late-block probability")
    axes[1].set_ylabel("On-time goodput (Mb/s)")
    for ax in axes[:2]:
        ax.set_xlabel("Block size (kB)")
    axes[0].set_title("Beyond the knee: failures", loc="left")
    axes[1].set_title("…and goodput", loc="left")
    axes[0].legend(fontsize=7.5)
    ax = axes[2]
    tr = d["traces"]
    t = np.arange(len(tr["HOP"]["backlog"])) * 0.025 + 1.0
    for name, c in (("HOP", COL["hop"]), ("HOP + relax", COL["ours"])):
        ax.plot(t, tr[name]["backlog"], color=c, lw=1.4,
                label=f"{name} ({100 * np.mean(tr[name]['fail']):.0f}% late)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Queue carried into next block (kB)")
    ax.set_title("Self-inflicted queue, 170 kB blocks", loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "e4_collapse")


# =========================================================================== E5
def fig_e5():
    """E5a: promised vs true failure probability of the schedule each joint model picks."""
    d = load("e5a")
    rhos = sorted({r["rho"] for r in d["rows"]})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    for ax, k, title in ((axes[0], 94, "140 kB blocks: redundancy spread over both paths"),
                         (axes[1], 80, "120 kB blocks: no redundancy needed")):
        for model, c, lab in (("indep", COL["hop"], "Independent paths (HOP)"),
                              ("corr", COL["ours"], "Correlation-aware (ours)")):
            rows = sorted([r for r in d["rows"] if r["k"] == k and r["model"] == model],
                          key=lambda r: r["rho"])
            ax.plot(rhos, [r["true"] for r in rows], color=c, marker="o",
                    label=f"{lab}: true")
            ax.plot(rhos, [r["promised"] for r in rows], color=c, marker="o", ls=":",
                    mfc="white", label=f"{lab}: promised")
            for r in rows:
                if not r["feasible"]:
                    ax.annotate("flags infeasible", (r["rho"], r["true"]), xytext=(-62, 8),
                                textcoords="offset points", fontsize=7.5, color=c)
        ax.axhline(0.02, color=INK2, ls="--", lw=1.2)
        ax.set_xlabel("Correlation between path capacities ρ")
        ax.set_title(title, loc="left", fontsize=10)
    axes[0].set_ylabel("Block failure probability")
    axes[0].legend(fontsize=7.5, loc="upper left")
    axes[0].text(0.0, 0.0204, "target 2%", color=INK2, fontsize=8)
    fig.tight_layout()
    save(fig, "e5_correlation")


# =========================================================================== E6
def fig_e6():
    d = load("e6")
    agg = defaultdict(list)
    for r in d["rows"]:
        agg[(r["scheme"], r["B"])].append(r)
    sizes = d["sizes"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), gridspec_kw=dict(width_ratios=[1, 1.35]))
    ax = axes[0]
    x = np.array(sizes) / 1000
    show = [("split", SCOL["split"], "v", "-", LABEL["split"]),
            ("static20", SCOL["static20"], "s", "-", LABEL["static20"]),
            ("hop", SCOL["hop"], "o", "-", "HOP (SKIP t_thr = 3)"),
            ("hop_fast", SCOL["hop"], "o", "--", "HOP, fast outage detection (t_thr = 1)"),
            ("hop_fast_relax", COL["ours"], "^", "-", "HOP, fast detection + relax (ours)")]
    for s, c, mk, ls, lab in show:
        ax.errorbar(x, [np.mean([r["fail"] for r in agg[(s, B)]]) for B in sizes],
                    yerr=[ci95([r["fail"] for r in agg[(s, B)]]) for B in sizes],
                    color=c, marker=mk, ls=ls, label=lab, capsize=2, elinewidth=1)
    ax.set_xlabel("Block size (kB)")
    ax.set_ylabel("Late-block probability")
    ax.set_title("mmWave (blockages) + LTE", loc="left")
    ax.set_ylim(0, 0.25)
    target_line(ax)
    ax.legend(fontsize=7.3, loc="upper left")
    ax = axes[1]
    tr = np.array(d["est_trace"], dtype=float)
    tf = np.array(d["est_trace_fast"], dtype=float)
    sel = (tr[:, 0] >= 2.0) & (tr[:, 0] <= 7.0)
    ax.plot(tr[sel, 0], tr[sel, 1], color=MUTED, lw=1, label="mmWave: true capacity")
    ax.plot(tr[sel, 0], tr[sel, 3], color=COL["hop"], lw=1.8, label="SKIP estimate, t_thr = 3 (paper)")
    sel2 = (tf[:, 0] >= 2.0) & (tf[:, 0] <= 7.0)
    ax.plot(tf[sel2, 0], tf[sel2, 3], color=COL["ours"], lw=1.6, ls="--",
            label="SKIP estimate, t_thr = 1, faster zero samples")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Capacity (Mb/s)")
    ax.set_title("Can SKIP see a blockage?", loc="left")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_ylim(0, 115)
    fig.tight_layout()
    save(fig, "e6_blockage")


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["e1", "e2", "e3", "e4", "e5", "e6"]
    for name in todo:
        if os.path.exists(f"results/{name}.json"):
            globals()[f"fig_{name}"]()
            print("figure", name)
