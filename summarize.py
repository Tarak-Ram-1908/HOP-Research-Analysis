"""Extract the headline numbers quoted in the report and slides -> results/summary.json."""
import json
from collections import defaultdict

import numpy as np


def load(n):
    with open(f"results/{n}.json") as f:
        return json.load(f)


def mean_by(rows, keys, val):
    agg = defaultdict(list)
    for r in rows:
        agg[tuple(r[k] for k in keys)].append(r[val])
    return {k: float(np.mean(v)) for k, v in agg.items()}


out = {}

# ---- E1: largest block meeting 98% (mean failure <= 2%), goodput there
e1 = load("e1")
fail = mean_by(e1["rows"], ["scen", "scheme", "B"], "fail")
gp = mean_by(e1["rows"], ["scen", "scheme", "B"], "goodput")
ov = mean_by(e1["rows"], ["scen", "scheme", "B"], "overhead")
sizes = e1["sizes"]
out["e1"] = {}
for scen in ("balanced", "unbalanced", "highly_unbalanced", "rtt_unbalanced"):
    d = {}
    for s in ("split", "static20", "hop", "ideal", "hop_relax", "hop_printed"):
        if (scen, s, sizes[0]) not in fail:
            continue
        ok = [B for B in sizes if fail[(scen, s, B)] <= 0.02]
        # largest size such that every smaller size also meets the target
        best = None
        for B in sizes:
            if fail[(scen, s, B)] <= 0.02:
                best = B
            else:
                break
        d[s] = dict(max_B_kB=(best or 0) / 1000,
                    goodput_at_max=gp[(scen, s, best)] if best else None,
                    overhead_at_max=ov[(scen, s, best)] if best else None,
                    fail_80=fail[(scen, s, 80000)], fail_120=fail[(scen, s, 120000)],
                    fail_150=fail[(scen, s, 150000)], fail_180=fail[(scen, s, 180000)],
                    peak_goodput=max(gp[(scen, s, B)] for B in sizes))
    bd = e1["bounds"][scen]
    okb = [B for B in sizes if bd[str(B)] <= 0.02]
    d["bound_max_B_kB"] = max(okb) / 1000 if okb else None
    out["e1"][scen] = d

out["e1x"] = {
    "rtt180": dict(hop=gp[("rtt_unbalanced", "hop", 180000)], relax=gp[("rtt_unbalanced", "hop_relax", 180000)],
                   hop_fail=fail[("rtt_unbalanced", "hop", 180000)], relax_fail=fail[("rtt_unbalanced", "hop_relax", 180000)]),
    "hu180": dict(hop=gp[("highly_unbalanced", "hop", 180000)], relax=gp[("highly_unbalanced", "hop_relax", 180000)],
                  hop_fail=fail[("highly_unbalanced", "hop", 180000)], relax_fail=fail[("highly_unbalanced", "hop_relax", 180000)]),
    "printed150": dict(printed=fail[("balanced", "hop_printed", 150000)], hop=fail[("balanced", "hop", 150000)],
                       printed_ov=ov[("balanced", "hop_printed", 150000)], hop_ov=ov[("balanced", "hop", 150000)]),
}

# ---- E2
e2 = load("e2")
red = [r for r in e2["rows"] if sum(r["sched"]) > r["k"]]
out["e2"] = dict(max_psi_sum=max(max(r["psi_sum"]) for r in e2["rows"]),
                 printed_zero_frac=float(np.mean([r["printed"] < 1e-9 for r in red])),
                 max_abs_err_exact=float(max(abs(r["exact"] - r["mc"]) for r in e2["rows"])),
                 n=len(e2["rows"]))

# ---- E3
e3 = load("e3")
r4 = [r for r in e3["rows"] if r["m"] == 4 and r["k"] == 120][0]
out["e3"] = dict(m4k120_enum_ms=r4["t_enum"] * 1e3, m4k120_conv_us=r4["t_conv"] * 1e6,
                 speedup_m4k120=r4["t_enum"] / r4["t_conv"],
                 max_conv_us=max(r["t_conv"] for r in e3["rows"]) * 1e6,
                 max_abs_diff=max(abs(r["v_conv"] - r["v_enum"]) for r in e3["rows"]
                                  if r["v_enum"] is not None),
                 max_rel_diff=max(abs(r["v_conv"] - r["v_enum"]) / r["v_enum"] for r in e3["rows"]
                                  if r["v_enum"] is not None),
                 pe_range=[min(r["v_conv"] for r in e3["rows"]),
                           max(r["v_conv"] for r in e3["rows"])])

# ---- E4
e4 = load("e4")
f4 = mean_by(e4["rows"], ["variant", "B"], "fail")
g4 = mean_by(e4["rows"], ["variant", "B"], "goodput")
o4 = mean_by(e4["rows"], ["variant", "B"], "overhead")
out["e4"] = {v: {str(B): dict(fail=f4[(v, B)], goodput=g4[(v, B)], overhead=o4[(v, B)])
                 for B in e4["sizes"]} for v in {k[0] for k in f4}}
out["e4_traces"] = {k: dict(late=float(np.mean(v["fail"])), mean_backlog_kB=float(np.mean(v["backlog"])),
                            max_backlog_kB=float(np.max(v["backlog"])))
                    for k, v in e4["traces"].items()}

# ---- E5
e5 = load("e5")
f5 = mean_by(e5["rows"], ["scheme", "rho", "p_thr"], "fail")
p5 = mean_by(e5["rows"], ["scheme", "rho", "p_thr"], "pred")
i5 = mean_by(e5["rows"], ["scheme", "rho", "p_thr"], "infeasible")
r5 = mean_by(e5["rows"], ["scheme", "rho", "p_thr"], "rho_est")
out["e5"] = {f"{s}|{rho}|{p}": dict(fail=f5[(s, rho, p)], pred=p5[(s, rho, p)],
                                     infeasible=i5[(s, rho, p)], rho_est=r5[(s, rho, p)])
             for (s, rho, p) in f5}

# ---- E5b (ideal marginals; pooled late/blocks)
import os
e5b = load("e5b") if os.path.exists("results/e5b.json") else {"rows": []}
agg = defaultdict(list)
for r in e5b["rows"]:
    agg[(r["scheme"], r["rho"])].append(r)
out["e5b"] = {f"{s}|{rho}": dict(fail=sum(r["late"] for r in v) / sum(r["blocks"] for r in v),
                                 pred=float(np.mean([r["pred"] for r in v])),
                                 infeasible=float(np.mean([r["infeasible"] for r in v])),
                                 overhead=float(np.mean([r["overhead"] for r in v])),
                                 blocks=sum(r["blocks"] for r in v))
              for (s, rho), v in agg.items()}

# ---- E5a
e5a = load("e5a")
out["e5a"] = {f"{r['k']}|{r['rho']}|{r['model']}": r for r in e5a["rows"]}

# ---- E6
e6 = load("e6")
f6 = mean_by(e6["rows"], ["scheme", "B"], "fail")
g6 = mean_by(e6["rows"], ["scheme", "B"], "goodput")
out["e6"] = {f"{s}|{B}": dict(fail=f6[(s, B)], goodput=g6[(s, B)]) for (s, B) in f6}

with open("results/summary.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print(json.dumps(out["e1"], indent=1, default=float)[:3000])
