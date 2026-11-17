"""LinkedIn carousel slides (1080x1350, 4:5) built from results/*.json.

Run from the project root:  python linkedin/make_slides.py
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

BG, INK, INK2, MUTED, GRID = "#fbfaf7", "#0b0b0b", "#52514e", "#8a8984", "#e7e6e2"
BLUE, ORANGE, RED, GREEN, GOLD = "#2a78d6", "#eb6834", "#e34948", "#1baf7a", "#eda100"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "legend.frameon": False, "figure.facecolor": BG, "axes.facecolor": BG,
})
W, H, DPI = 10.8, 13.5, 100          # 1080 x 1350 px


def load(name):
    with open(os.path.join(ROOT, "results", f"{name}.json")) as f:
        return json.load(f)


def frame(n, kicker, title, sub=None):
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.text(0.07, 0.94, kicker.upper(), fontsize=17, color=ORANGE, weight="bold")
    fig.text(0.07, 0.905, title, fontsize=34, color=INK, weight="bold", va="top",
             linespacing=1.15)
    if sub:
        fig.text(0.07, 0.77, sub, fontsize=19, color=INK2, va="top", linespacing=1.4)
    fig.text(0.07, 0.03, "HOP protocol: reproduction & extensions  |  CS3001, IIITDM Kancheepuram",
             fontsize=13, color=MUTED)
    fig.text(0.93, 0.03, f"{n}/4", fontsize=13, color=MUTED, ha="right")
    return fig


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), dpi=DPI, facecolor=BG)
    plt.close(fig)


# ------------------------------------------------------------------ 1. cover
def cover():
    s = load("summary")
    fig = frame(1, "Research paper analysis",
                "Can a network promise\na deadline?",
                "We re-implemented HOP (IEEE/ACM ToN 2021), a multipath\n"
                "transport that uses adaptive FEC to deliver each data block\n"
                "within 35 ms with 98% reliability, then stress-tested it.")
    stats = [
        ("~5,000×", "faster exact failure-probability\ncomputation at 4 paths"),
        (f"{s['e1x']['rtt180']['relax']:.1f} vs {s['e1x']['rtt180']['hop']:.1f}",
         "Mb/s kept by our graceful-degradation\nrule where plain HOP collapses"),
        ("2.86%", "true failure rate when the model\npromises 1.94% on correlated paths"),
    ]
    y = 0.55
    for big, small in stats:
        fig.text(0.07, y, big, fontsize=50, color=BLUE, weight="bold", va="center")
        fig.text(0.07, y - 0.065, small, fontsize=18, color=INK2, va="center", linespacing=1.3)
        y -= 0.15
    save(fig, "slide1_cover.png")


# ------------------------------------------------------------------ 2. eq. (5)
def eq5():
    rows = [r for r in load("e2")["rows"] if r["k"] == 97 and r["paths"] == "40/40"]
    k = 97
    red = [100 * (sum(r["sched"]) - k) / k for r in rows]
    fig = frame(2, "Finding 1: a formula error",
                "One printed equation\nhides the risk",
                "Implemented literally, the paper's eq. (5) says a block can\n"
                "never fail once any redundancy is added. Under the paper's\n"
                "own model the exact term is λ(d+1) − λ(d).")
    ax = fig.add_axes([0.13, 0.12, 0.8, 0.47])
    ax.plot(red, [100 * r["mc"] for r in rows], color="#bbbab5", lw=9, alpha=0.7,
            label="Monte Carlo ground truth", solid_capstyle="round")
    ax.plot(red, [100 * r["exact"] for r in rows], color=ORANGE, lw=3.5, marker="o", ms=10,
            label="Exact formula (ours)")
    ax.plot(red, [100 * r["printed"] for r in rows], color=RED, lw=3.5, marker="X", ms=12,
            label="Eq. (5) as printed")
    ax.set_xlabel("Redundancy added (% of block)", fontsize=18)
    ax.set_ylabel("Block failure probability (%)", fontsize=18)
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=16, loc="upper right")
    ax.set_title("145 kB block over two 40 Mb/s paths", fontsize=17, color=INK2, loc="left")
    save(fig, "slide2_equation.png")


# ------------------------------------------------------------------ 3. speed
def speed():
    rows = [r for r in load("e3")["rows"] if r["k"] == 120]
    s = load("summary")["e3"]
    fig = frame(3, "Improvement 1: exact and fast",
                "From exponential to\npolynomial time",
                "The paper enumerates every delivery vector: O(kᵐ).\n"
                "The same sum is a convolution: O(m k²). Identical results\n"
                f"(max difference {s['max_abs_diff']:.0e}), and real-time for 3–6 paths.")
    ax = fig.add_axes([0.13, 0.12, 0.8, 0.47])
    m = [r["m"] for r in rows]
    ax.semilogy(m, [r["t_conv"] * 1e3 for r in rows], color=ORANGE, lw=3.5, marker="o", ms=11,
                label="Convolution (ours)")
    en = [(r["m"], r["t_enum"] * 1e3) for r in rows if r["t_enum"] is not None]
    ax.semilogy(*zip(*en), color=BLUE, lw=3.5, marker="s", ms=11, label="Enumeration (paper)")
    for mm, n in [(5, rows[3]["vectors"]), (6, rows[4]["vectors"])]:
        ax.annotate(f"{n/1e6:.0f}M vectors:\nnot run", (mm, 300), ha="center", fontsize=13,
                    color=BLUE)
    ax.axhline(25, color=MUTED, ls="--", lw=1.5)
    ax.text(6.15, 18, "25 ms: one block period", ha="right", fontsize=13, color=MUTED)
    ax.set_ylim(0.01, 5000)
    ax.set_xticks(m)
    ax.set_xlabel("Number of network paths", fontsize=18)
    ax.set_ylabel("Time per evaluation (ms, log scale)", fontsize=18)
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=16, loc="upper left")
    ax.set_title(f"Block of k = 120 packets  |  {s['speedup_m4k120']:,.0f}× faster at 4 paths",
                 fontsize=17, color=INK2, loc="left")
    save(fig, "slide3_speed.png")


# ------------------------------------------------------------------ 4. collapse
def collapse():
    e1 = load("e1")
    acc = defaultdict(list)
    for r in e1["rows"]:
        if r["scen"] == "rtt_unbalanced":
            acc[(r["scheme"], r["B"])].append(r["goodput"])
    sizes = e1["sizes"]
    kb = [b / 1000 for b in sizes]
    fig = frame(4, "Improvement 2: fail gracefully",
                "Stop a reliability\ncollapse from spiralling",
                "When 98% is out of reach, HOP adds maximum redundancy,\n"
                "which builds queues and makes the next block late too.\n"
                "Falling back to the best reachable target breaks the loop.")
    ax = fig.add_axes([0.13, 0.12, 0.8, 0.47])
    ax.plot(kb, [b * 8 / 0.025 / 1e6 for b in sizes], color=MUTED, ls=":", lw=2,
            label="Offered load")
    for sch, col, lab, mk in [("hop", BLUE, "HOP as specified", "o"),
                              ("hop_relax", ORANGE, "HOP + graceful relax (ours)", "^")]:
        ax.plot(kb, [np.mean(acc[(sch, b)]) for b in sizes], color=col, lw=3.5, marker=mk,
                ms=10, label=lab)
    s = load("summary")["e1x"]["rtt180"]
    ax.annotate(f"{s['relax']:.1f} Mb/s", (180, s["relax"]), (186, 44), fontsize=16,
                color=ORANGE, weight="bold", arrowprops=dict(arrowstyle="->", color=ORANGE))
    ax.annotate(f"{s['hop']:.1f} Mb/s", (180, s["hop"]), (184, 10), fontsize=16,
                color=BLUE, weight="bold", arrowprops=dict(arrowstyle="->", color=BLUE))
    ax.set_ylim(0, 66)
    ax.set_xlabel("Block size (kB), one block every 25 ms", fontsize=18)
    ax.set_ylabel("On-time goodput (Mb/s)", fontsize=18)
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=16, loc="upper left")
    ax.set_title("Two paths with 20 / 30 ms RTT, 35 ms deadline", fontsize=17, color=INK2,
                 loc="left")
    save(fig, "slide4_collapse.png")


if __name__ == "__main__":
    cover(); eq5(); speed(); collapse()
    print("slides written to", HERE)
