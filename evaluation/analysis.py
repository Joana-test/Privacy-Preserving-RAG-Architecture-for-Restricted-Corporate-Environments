"""
Generates every number reported in the thesis evaluation chapter from the
canonical data sources, and writes a Markdown report with thesis-formatted
tables plus PNG figures.

Author: J. Fermin, Master's thesis, LMU München, 2026.

Usage (from the project root):
    python -m evaluation.analysis
    python -m evaluation.analysis --output-dir evaluation/results/thesis_report
"""
import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

PHASE3_ROOT = "evaluation/results/misclassification"
RERUN_FILES = {
    
    "Llama 3.1 8B Q4_K_M": "evaluation/results/model_compare/llamaQ4/Q4.json",
    "Llama 3.1 8B Q8_0":   "evaluation/results/model_compare/lammaQ8/Q8.json",
    "Qwen2.5-14B Q4_K_M":  "evaluation/results/model_compare/Qwen/Qwen14b.json",
}
CROSS_DOMAIN_ALLOW = "cross_domain_allow"
N_SEEDS = 3


# ---------------------------------------------------------------- Format
def de(x, dec=1):
    """Zahl im deutschen Format (Dezimalkomma)."""
    return f"{x:.{dec}f}".replace(".", ",")


def de_pct(x, n, dec=1):
    """Anteil als Prozentwert im deutschen Format; n/a bei leerem Nenner."""
    return f"{de(100.0 * x / n, dec)}\u2009%" if n else "n/a"


# ---------------------------------------------------------------- Phase 3
def load_phase3():
    data = {}
    for path in sorted(glob.glob(f"{PHASE3_ROOT}/eps*/raw_results.json")):
        d = json.load(open(path))
        data[d["epsilon"]] = d["by_seed"]
    if not data:
        raise SystemExit(f"Keine Phase-3-Dateien unter {PHASE3_ROOT} gefunden.")
    return data


def compute_phase3(phase3):
    
    out = {
        "afr_stress": [],      # tab:afr-stresstest
        "seed_skew": {},       # [A1b]
        "afr_sl": [],          # tab:afr-sl-vergleich
        "sl_perf": [],         # tab:sl-performance (je eps)
        "sl_perf_pool": None,  # gepoolte Zeile
        "fpr_cd": [],          # tab:fpr-crossdomain (je eps)
        "fpr_cd_pool": None,   # gepoolte Werte (56/90, 33/198)
        "fp_rules": None,      # tab:fp-analyse
        "latency": None,       # [A6]
    }

    # --- [A1] tab:afr-stresstest --------------------------------------
    for eps, seeds in sorted(phase3.items()):
        n = struct = ans = 0
        corrupted = []
        for s in seeds.values():
            recs = s["d6_results"]
            n += len(recs)
            struct += sum(r["structural_leak"] for r in recs)
            ans += sum(r["answer_leak"] for r in recs)
            if "n_corrupted_chunks" in s:
                corrupted.append(s["n_corrupted_chunks"])
        out["afr_stress"].append({
            "eps": eps, "n": n,
            "corrupted_mean": statistics.mean(corrupted) if corrupted else None,
            "struct": struct, "answer": ans,
        })

    # --- [A1b] Seed-Schiefe --------------------------------------------
    for eps, seeds in sorted(phase3.items()):
        out["seed_skew"][eps] = {
            seed: sum(r["answer_leak"] for r in s["d6_results"])
            for seed, s in seeds.items()
        }

    # --- [A2] tab:afr-sl-vergleich -------------------------------------
    for eps, seeds in sorted(phase3.items()):
        n = a = b = blocks = 0
        for s in seeds.values():
            n += len(s["d6_results"])
            a += sum(r["answer_leak"] for r in s["d6_results"])
            b += sum(r["answer_leak"] for r in s["d6_sl_results"])
            blocks += sum(r["sl_verdict"] == "UNSAFE" for r in s["d6_sl_results"])
        out["afr_sl"].append({
            "eps": eps, "n": n, "afr": a, "sl": b,
            "reduction": (a - b) / a if a else None,
            "blocks_mean": blocks / N_SEEDS,
        })

    # --- [A3]-[A6] Confusion (Kreuz-Paarung), FPR, Rules, Latenz --------
    pool = defaultdict(int)
    fp_rule = defaultdict(int)
    cd_fp = cd_n = rest_fp = rest_n = 0
    d6_lat, d6sl_lat = [], []
    for eps, seeds in sorted(phase3.items()):
        c = defaultdict(int)
        eps_cd_fp = eps_cd_n = eps_rest_fp = eps_rest_n = 0
        for s in seeds.values():
            leaks = {r["query_id"]: r["answer_leak"] for r in s["d6_results"]}
            d6_lat += [r["total_time_ms"] for r in s["d6_results"]
                       if r.get("total_time_ms")]
            for r in s["d6_sl_results"]:
                if r.get("total_time_ms"):
                    d6sl_lat.append(r["total_time_ms"])
                deny = r["expected_decision"] == "deny"
                blocked = r["sl_verdict"] == "UNSAFE"
                leak = leaks.get(r["query_id"], False)
                is_cd = r.get("category") == CROSS_DOMAIN_ALLOW
                if not deny:
                    if is_cd:
                        cd_n += 1
                        eps_cd_n += 1
                    else:
                        rest_n += 1
                        eps_rest_n += 1
                    if blocked:
                        c["FP"] += 1
                        fp_rule[r.get("sl_violated_rule") or "null"] += 1
                        if is_cd:
                            cd_fp += 1
                            eps_cd_fp += 1
                        else:
                            rest_fp += 1
                            eps_rest_fp += 1
                    else:
                        c["TN"] += 1
                elif leak:
                    c["TP" if blocked else "FN"] += 1
        tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        out["sl_perf"].append({
            "eps": eps, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "prec": p, "rec": rc if tp + fn else None,
            "f1": f1 if tp + fn else None, "fpr": fpr,
        })
        out["fpr_cd"].append({
            "eps": eps,
            "cd_fp": eps_cd_fp, "cd_n": eps_cd_n,
            "rest_fp": eps_rest_fp, "rest_n": eps_rest_n,
        })
        for k, v in c.items():
            pool[k] += v
    tp, fp, fn, tn = pool["TP"], pool["FP"], pool["FN"], pool["TN"]
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    out["sl_perf_pool"] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                           "prec": p, "rec": rc, "f1": f1}
    out["fpr_cd_pool"] = {"cd_fp": cd_fp, "cd_n": cd_n,
                          "rest_fp": rest_fp, "rest_n": rest_n}
    out["fp_rules"] = dict(fp_rule)
    if d6_lat and d6sl_lat:
        m1, m2 = statistics.mean(d6_lat), statistics.mean(d6sl_lat)
        out["latency"] = {"d6_ms": m1, "d6sl_ms": m2,
                          "overhead_ms": m2 - m1,
                          "overhead_rel": (m2 - m1) / m1}
    return out


# ---------------------------------------------------------------- Rerun
def compute_rerun():
    models = []
    for name, path in RERUN_FILES.items():
        if not os.path.exists(path):
            models.append({"name": name, "missing": path})
            continue
        results = json.load(open(path))["results"]
        tp = sum(r["confusion"]["TP"] for r in results)
        fp = sum(r["confusion"]["FP"] for r in results)
        fn = sum(r["confusion"]["FN"] for r in results)
        tn = sum(r["confusion"]["TN"] for r in results)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        rules = defaultdict(int)
        for r in results:
            for k, v in r["fp_violated_rule"].items():
                rules[k] += v
        cd_fp = sum(r["fp_cross_domain"] for r in results)
        cd_n = sum(r["allow_cross_domain_n"] for r in results)
        rest_fp = sum(r["fp_rest"] for r in results)
        rest_n = sum(r["allow_rest_n"] for r in results)
        lats = [r["sl_latency_mean_ms"] for r in results
                if r.get("sl_latency_mean_ms")]
        models.append({
            "name": name,
            "eps_list": [r["epsilon"] for r in results],
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "prec": p, "rec": rc, "f1": f1,
            "recall_per_eps": [(r["epsilon"], r["recall"]) for r in results],
            "rules": dict(rules),
            "cd_fp": cd_fp, "cd_n": cd_n,
            "rest_fp": rest_fp, "rest_n": rest_n,
            "sl_latency_mean_ms": statistics.mean(lats) if lats else None,
        })
    return models


# ---------------------------------------------------------------- Figuren
def make_figures(a, models, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNUNG: matplotlib nicht installiert -- Grafiken werden "
              "uebersprungen (pip install matplotlib).")
        return []

    figs = []
    eps = [row["eps"] for row in a["afr_stress"]]

    # Fig 1: Structural vs. Answer Leak Rate (AFR)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(eps, [100 * r["struct"] / r["n"] for r in a["afr_stress"]],
            marker="o", label="Structural Leak Rate")
    ax.plot(eps, [100 * r["answer"] / r["n"] for r in a["afr_stress"]],
            marker="s", label="Answer Leak Rate")
    ax.set_xlabel("Fehlerrate $\\varepsilon$")
    ax.set_ylabel("Rate (%)")
    ax.set_title("AFR-Pipeline unter Fehlklassifikation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = os.path.join(outdir, "fig_afr_stresstest.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    figs.append(("AFR-Pipeline: Leak-Raten über $\\varepsilon$", p))

    # Fig 2: Answer Leak Rate AFR vs. AFR+SL
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(eps, [100 * r["afr"] / r["n"] for r in a["afr_sl"]],
            marker="o", label="AFR")
    ax.plot(eps, [100 * r["sl"] / r["n"] for r in a["afr_sl"]],
            marker="s", label="AFR+SL")
    ax.set_xlabel("Fehlerrate $\\varepsilon$")
    ax.set_ylabel("Answer Leak Rate (%)")
    ax.set_title("Answer Leak Rate: AFR vs. AFR+Security Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = os.path.join(outdir, "fig_afr_vs_sl.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    figs.append(("Answer Leak Rate: AFR vs. AFR+SL", p))

    # Fig 3: SL-Metriken ueber eps
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(eps, [r["prec"] for r in a["sl_perf"]], marker="o",
            label="Precision")
    ax.plot(eps, [r["rec"] if r["rec"] is not None else float("nan")
                  for r in a["sl_perf"]], marker="s", label="Recall")
    ax.plot(eps, [r["f1"] if r["f1"] is not None else float("nan")
                  for r in a["sl_perf"]], marker="^", label="$F_1$")
    ax.plot(eps, [r["fpr"] for r in a["sl_perf"]], marker="d",
            label="FPR", linestyle="--")
    ax.set_xlabel("Fehlerrate $\\varepsilon$")
    ax.set_ylabel("Wert")
    ax.set_ylim(0, 1.05)
    ax.set_title("Security-Layer-Performance (Pipeline-Regime)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = os.path.join(outdir, "fig_sl_performance.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    figs.append(("Security-Layer-Metriken über $\\varepsilon$", p))

    # Fig 4: FPR Cross-Domain vs. Rest
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(eps, [100 * r["cd_fp"] / r["cd_n"] if r["cd_n"] else 0
                  for r in a["fpr_cd"]], marker="o",
            label="Cross-Domain-Allow")
    ax.plot(eps, [100 * r["rest_fp"] / r["rest_n"] if r["rest_n"] else 0
                  for r in a["fpr_cd"]], marker="s",
            label="Übrige Allow-Kategorien")
    ax.set_xlabel("Fehlerrate $\\varepsilon$")
    ax.set_ylabel("FPR (%)")
    ax.set_title("False Positive Rate nach Query-Kategorie")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = os.path.join(outdir, "fig_fpr_categories.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    figs.append(("FPR nach Query-Kategorie", p))

    # Fig 5: Modellvergleich P/R/F1
    ok = [m for m in models if "missing" not in m]
    if ok:
        fig, ax = plt.subplots(figsize=(7, 4))
        idx = range(len(ok))
        w = 0.25
        ax.bar([i - w for i in idx], [m["prec"] for m in ok], w,
               label="Precision")
        ax.bar(list(idx), [m["rec"] for m in ok], w, label="Recall")
        ax.bar([i + w for i in idx], [m["f1"] for m in ok], w,
               label="$F_1$")
        ax.set_xticks(list(idx))
        ax.set_xticklabels([m["name"] for m in ok], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title("Modellvergleich (Filter-Regime, gepoolt)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        p = os.path.join(outdir, "fig_model_comparison.png")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        figs.append(("Modellvergleich: Precision, Recall, $F_1$", p))
    return figs


# ---------------------------------------------------------------- Report
def fmt_or(v, dec=3, dash="—"):
    return de(v, dec) if v is not None else dash


def write_report(a, models, figs, outdir):
    L = []
    add = L.append
    add("# Thesis-Zahlen: Evaluationsreport\n")
    add("Automatisch erzeugt aus den kanonischen Datenquellen "
        f"(`{PHASE3_ROOT}`, Rerun-JSONs). Messlogik identisch zur "
        "Original-`thesis_numbers.py`.\n")

    # A1
    add("\n## A1 — tab:afr-stresstest (D6, gepoolt über 3 Seeds)\n")
    add("| $\\varepsilon$ | Fehlklassifizierte Chunks (Ø) | "
        "Structural Leak Rate | Answer Leak Rate |")
    add("|---:|---:|---:|---:|")
    for r in a["afr_stress"]:
        ck = de(r["corrupted_mean"], 1) if r["corrupted_mean"] is not None \
            else "(fehlt)"
        add(f"| {de(r['eps'],1)} | {ck} "
            f"| {de_pct(r['struct'], r['n'])} ({r['struct']}/{r['n']}) "
            f"| {de_pct(r['answer'], r['n'])} ({r['answer']}/{r['n']}) |")

    # A1b
    add("\n### A1b — Answer Leaks je Seed (Seed-Schiefe)\n")
    seed_cols = sorted(next(iter(a["seed_skew"].values())), key=int)
    add("| $\\varepsilon$ | " + " | ".join(
        f"Seed {s}" for s in seed_cols) + " |")
    add("|---:|" + "---:|" * len(seed_cols))
    for eps, per in sorted(a["seed_skew"].items()):
        add(f"| {de(eps,1)} | " +
            " | ".join(str(per[s]) for s in sorted(per, key=int)) + " |")

    # A2
    add("\n## A2 — tab:afr-sl-vergleich (Answer Leaks, gepoolt)\n")
    add("| $\\varepsilon$ | AFR | AFR+SL | Leak-Reduktion | SL Blocks (Ø) |")
    add("|---:|---:|---:|---:|---:|")
    for r in a["afr_sl"]:
        red = de_pct(r["afr"] - r["sl"], r["afr"]) if r["afr"] else "—"
        add(f"| {de(r['eps'],1)} "
            f"| {de_pct(r['afr'], r['n'])} ({r['afr']}/{r['n']}) "
            f"| {de_pct(r['sl'], r['n'])} ({r['sl']}/{r['n']}) "
            f"| {red} | {de(r['blocks_mean'],2)} |")

    # A3
    add("\n## A3 — tab:sl-performance (Pipeline-Regime; "
        "Verdict D6_SL × Leak-GT D6)\n")
    add("Gepoolte Counts über drei Seeds; Metriken aus gepoolten "
        "Häufigkeiten. In Klammern: Ø je Seed (Count/3), wie in der "
        "Thesis-Tabelle dargestellt.\n")
    add("| $\\varepsilon$ | TP | FP | FN | TN | Prec. | Recall | $F_1$ | FPR |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in a["sl_perf"]:
        add(f"| {de(r['eps'],1)} "
            f"| {r['TP']} ({de(r['TP']/N_SEEDS,2)}) "
            f"| {r['FP']} ({de(r['FP']/N_SEEDS,2)}) "
            f"| {r['FN']} ({de(r['FN']/N_SEEDS,2)}) "
            f"| {r['TN']} ({de(r['TN']/N_SEEDS,2)}) "
            f"| {de(r['prec'],3)} | {fmt_or(r['rec'])} "
            f"| {fmt_or(r['f1'])} | {de(r['fpr'],3)} |")
    pp = a["sl_perf_pool"]
    add(f"| **Pool** | {pp['TP']} | {pp['FP']} | {pp['FN']} | {pp['TN']} "
        f"| {de(pp['prec'],3)} | {de(pp['rec'],3)} | {de(pp['f1'],3)} | |")

    # A4
    add("\n## A4 — tab:fpr-crossdomain (je Fehlerrate, gepoolt über Seeds)\n")
    add("| $\\varepsilon$ | Cross-Domain FP | FPR Cross-Domain "
        "| Rest FP | FPR Rest |")
    add("|---:|---:|---:|---:|---:|")
    for r in a["fpr_cd"]:
        add(f"| {de(r['eps'],1)} "
            f"| {r['cd_fp']}/{r['cd_n']} | {de_pct(r['cd_fp'], r['cd_n'])} "
            f"| {r['rest_fp']}/{r['rest_n']} "
            f"| {de_pct(r['rest_fp'], r['rest_n'])} |")
    cp = a["fpr_cd_pool"]
    add(f"\nGepoolt über alle Fehlerraten: Cross-Domain "
        f"{cp['cd_fp']}/{cp['cd_n']} = {de_pct(cp['cd_fp'], cp['cd_n'])}; "
        f"übrige Allow {cp['rest_fp']}/{cp['rest_n']} = "
        f"{de_pct(cp['rest_fp'], cp['rest_n'])}.\n")

    # A5
    add("\n## A5 — tab:fp-analyse (FP nach violated_rule, gepoolt)\n")
    total = sum(a["fp_rules"].values())
    add("| Verstoßtyp | Anzahl | Anteil |")
    add("|---|---:|---:|")
    for k in sorted(a["fp_rules"], key=a["fp_rules"].get, reverse=True):
        v = a["fp_rules"][k]
        add(f"| `{k}` | {v} | {de_pct(v, total)} |")
    add(f"| **Gesamt** | {total} | 100,0\u2009% |")

    # A6
    add("\n## A6 — Latenz-Overhead (Pipeline)\n")
    if a["latency"]:
        lat = a["latency"]
        add(f"- D6 (AFR): {lat['d6_ms']:.0f} ms")
        add(f"- D6_SL (AFR+SL): {lat['d6sl_ms']:.0f} ms")
        add(f"- Overhead: {lat['overhead_ms']:.0f} ms = "
            f"{de(100*lat['overhead_rel'],1)}\u2009%")
    else:
        add("(total_time_ms nicht in allen Records vorhanden)")

    # B
    add("\n## B — Modellvergleich (Filter-Regime, Rerun)\n")
    ok = [m for m in models if "missing" not in m]
    for m in models:
        if "missing" in m:
            add(f"\n**{m['name']}**: DATEI FEHLT (`{m['missing']}`)")
    if ok:
        add("\n| Metrik | " + " | ".join(m["name"] for m in ok) + " |")
        add("|---|" + "---:|" * len(ok))
        rows = [
            ("TP (gesamt)", lambda m: str(m["TP"])),
            ("FP (gesamt)", lambda m: str(m["FP"])),
            ("FN (gesamt)", lambda m: str(m["FN"])),
            ("TN (gesamt)", lambda m: str(m["TN"])),
            ("Precision", lambda m: de(m["prec"], 3)),
            ("Recall", lambda m: de(m["rec"], 3)),
            ("$F_1$", lambda m: de(m["f1"], 3)),
            ("Cross-Domain-FPR",
             lambda m: de_pct(m["cd_fp"], m["cd_n"])),
            ("Rest-FPR",
             lambda m: de_pct(m["rest_fp"], m["rest_n"])),
            ("Ø SL-Latenz",
             lambda m: f"{m['sl_latency_mean_ms']/1000:.1f} s".replace(".", ",")
             if m["sl_latency_mean_ms"] else "—"),
        ]
        for label, f in rows:
            add(f"| {label} | " + " | ".join(f(m) for m in ok) + " |")

        add("\n### FP nach violated_rule je Modell\n")
        keys = sorted({k for m in ok for k in m["rules"]})
        add("| Verstoßtyp | " + " | ".join(m["name"] for m in ok) + " |")
        add("|---|" + "---:|" * len(ok))
        for k in keys:
            add(f"| `{k}` | " +
                " | ".join(str(m["rules"].get(k, 0)) for m in ok) + " |")

        add("\n### Recall je Fehlerrate\n")
        eps_cols = [e for e, _ in ok[0]["recall_per_eps"]]
        add("| Modell | " + " | ".join(
            f"$\\varepsilon$={de(e,1)}" for e in eps_cols) + " |")
        add("|---|" + "---:|" * len(eps_cols))
        for m in ok:
            rec_by_eps = dict(m["recall_per_eps"])
            add(f"| {m['name']} | " + " | ".join(
                de(rec_by_eps[e], 3) if e in rec_by_eps else "—"
                for e in eps_cols) + " |")

    # Figuren
    if figs:
        add("\n## Grafiken\n")
        for title, path in figs:
            add(f"### {title}\n")
            add(f"![{title}]({os.path.basename(path)})\n")

    report_path = os.path.join(outdir, "thesis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return report_path


# ---------------------------------------------------------------- Main
def main():
    parser = argparse.ArgumentParser(
        description="Thesis-Tabellen als Markdown-Report mit Grafiken")
    parser.add_argument("--output-dir", type=str,
                        default="evaluation/results/thesis_report")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    phase3 = load_phase3()
    a = compute_phase3(phase3)
    models = compute_rerun()
    figs = make_figures(a, models, args.output_dir)
    report = write_report(a, models, figs, args.output_dir)

    print(f"Report geschrieben -> {report}")
    for _, p in figs:
        print(f"Grafik            -> {p}")


if __name__ == "__main__":
    main()
