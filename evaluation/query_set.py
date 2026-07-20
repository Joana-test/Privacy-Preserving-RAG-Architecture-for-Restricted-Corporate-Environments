# Adaptiert von Namboothiri et al. (2026), "Authorization-First Retrieval - Enforcing Least Privilege in Multi-Agent RAG Systems"
# Original: https://github.com/rohithzmoi/afr-eval-artifact/tree/main
# Changes:
#    - Queryset um zwei queries reduziert
# Anonymous Authors
# Licensed under the Apache License, Version 2.0
# See LICENSE file for details

"""
Results aggregator for AFR evaluation.

Reads raw_results.json and generates paper-ready markdown tables
and a CSV summary.

Usage:
    python -m evaluation.aggregate_results
    python -m evaluation.aggregate_results --input evaluation/results/raw_results.json
"""

import argparse
import csv
import json
import os
import sys
from typing import List, Dict, Any, Optional

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def load_results(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def _safe_rate(num: int, den: int) -> str:
    if den == 0:
        return "N/A"
    return f"{num/den*100:.1f}%"


def generate_report(results: List[Dict], output_dir: str, dry_run_mode: bool = False) -> str:
    """Generate evaluation report as markdown."""

    total = len(results)

    # ── Parse results ────────────────────────────────────────────────
    d3_structural = 0
    d6_structural = 0
    d3_answer = 0
    d6_answer = 0
    d3_grounded = 0
    d6_grounded = 0
    d3_ungrounded = 0
    d6_ungrounded = 0
    d3_false_positive = 0
    d6_false_positive = 0
    d3_times: List[float] = []
    d6_times: List[float] = []
    d3_retrieval_times: List[float] = []
    d6_retrieval_times: List[float] = []
    d3_gen_times: List[float] = []
    d6_gen_times: List[float] = []
    # Allow-only latency
    d3_allow_times: List[float] = []
    d6_allow_times: List[float] = []

    categories: Dict[str, Dict[str, int]] = {}

    for r in results:
        d3 = r.get("d3")
        d6 = r.get("d6")
        cat = r["query"]["category"]
        expected_leak = r["query"]["expected_leak"]

        if cat not in categories:
            categories[cat] = {
                "total": 0, "expected_leaks": 0,
                "d3_structural": 0, "d6_structural": 0,
                "d3_answer": 0, "d6_answer": 0,
            }
        categories[cat]["total"] += 1
        if expected_leak:
            categories[cat]["expected_leaks"] += 1

        if d3:
            if d3.get("structural_leak"):
                d3_structural += 1
                categories[cat]["d3_structural"] += 1
            if d3.get("answer_leak"):
                d3_answer += 1
                categories[cat]["d3_answer"] += 1
            cls3 = d3.get("leak_classification", "none")
            if cls3 == "grounded_leak":
                d3_grounded += 1
            elif cls3 == "ungrounded_guess":
                d3_ungrounded += 1
            elif cls3 == "false_positive":
                d3_false_positive += 1
            d3_times.append(d3["total_time_ms"])
            d3_retrieval_times.append(d3["retrieval_time_ms"])
            d3_gen_times.append(d3["generation_time_ms"])
            if not expected_leak:
                d3_allow_times.append(d3["total_time_ms"])

        if d6:
            if d6.get("structural_leak"):
                d6_structural += 1
                categories[cat]["d6_structural"] += 1
            if d6.get("answer_leak"):
                d6_answer += 1
                categories[cat]["d6_answer"] += 1
            cls6 = d6.get("leak_classification", "none")
            if cls6 == "grounded_leak":
                d6_grounded += 1
            elif cls6 == "ungrounded_guess":
                d6_ungrounded += 1
            elif cls6 == "false_positive":
                d6_false_positive += 1
            d6_times.append(d6["total_time_ms"])
            d6_retrieval_times.append(d6["retrieval_time_ms"])
            d6_gen_times.append(d6["generation_time_ms"])
            if not expected_leak:
                d6_allow_times.append(d6["total_time_ms"])

    # Combined "any leak" counts (structural OR answer)
    d3_any_leak = sum(
        1 for r in results
        if r.get("d3", {}).get("structural_leak") or r.get("d3", {}).get("answer_leak")
    )
    d6_any_leak = sum(
        1 for r in results
        if r.get("d6", {}).get("structural_leak") or r.get("d6", {}).get("answer_leak")
    )

    # Count expected-leak queries
    expected_leak_queries = [r for r in results if r["query"]["expected_leak"]]
    n_expected = len(expected_leak_queries)
    d3_leak_on_expected = sum(
        1 for r in expected_leak_queries if r.get("d3", {}).get("structural_leak")
    )
    d6_leak_on_expected = sum(
        1 for r in expected_leak_queries if r.get("d6", {}).get("structural_leak")
    )

    # ── Build Markdown ───────────────────────────────────────────────
    lines = []
    lines.append("# AFR Evaluation Report")
    lines.append("")
    lines.append(f"**Evaluation Date**: {_get_timestamp()}")
    lines.append(f"**Mode**: {'Structural Only (Dry Run)' if dry_run_mode else 'Full Evaluation (with LLM)'}")
    lines.append(f"**Total Queries**: {total}")
    lines.append(f"**Pipelines**: D3 (Retrieve-then-filter) vs D6 (Authorization-First Retrieval)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Table 1: Leakage Rate
    lines.append("## Table 1: Leakage Rate")
    lines.append("")
    if dry_run_mode:
        lines.append("| Pipeline | Structural Leaks | Total Queries | Structural Leak Rate |")
        lines.append("|----------|:----------------:|:-------------:|:--------------------:|")
        lines.append(f"| D3 (Retrieve-then-filter) | {d3_structural}/{total} | {total} | {_safe_rate(d3_structural, total)} |")
        lines.append(f"| **D6 (AFR)** | **{d6_structural}/{total}** | **{total}** | **{_safe_rate(d6_structural, total)}** |")
    else:
        lines.append("| Pipeline | Structural Leaks | True Answer Leaks | Any True Leak | Structural Rate | Answer Rate | Combined Rate |")
        lines.append("|----------|:----------------:|:-----------------:|:-------------:|:---------------:|:-----------:|:-------------:|")
        lines.append(
            f"| D3 (Retrieve-then-filter) | {d3_structural}/{total} | {d3_answer}/{total} | "
            f"{d3_any_leak}/{total} | {_safe_rate(d3_structural, total)} | "
            f"{_safe_rate(d3_answer, total)} | {_safe_rate(d3_any_leak, total)} |"
        )
        lines.append(
            f"| **D6 (AFR)** | **{d6_structural}/{total}** | **{d6_answer}/{total}** | "
            f"**{d6_any_leak}/{total}** | **{_safe_rate(d6_structural, total)}** | "
            f"**{_safe_rate(d6_answer, total)}** | **{_safe_rate(d6_any_leak, total)}** |"
        )
    lines.append("")
    if not dry_run_mode:
        lines.append("> **Structural Leak**: unauthorized chunks entered the LLM context window.")
        lines.append("> **True Answer Leak**: grounded or ungrounded keyword matches (excludes false positives from refusals).")
        lines.append("> **Any True Leak**: union of structural OR true answer leak.")
        lines.append("> False positives (keyword hits in refusals with no data-bearing tokens) are excluded from all leak counts.")
        lines.append("")

    # Table 2: Authorization Correctness
    lines.append("## Table 2: Authorization Correctness (Eq.1 Violations)")
    lines.append("")
    lines.append(f"Out of {n_expected} queries that **should be blocked** (unauthorized access):")
    lines.append("")
    lines.append("| Pipeline | Unauthorized Chunks in Context | Violation Rate |")
    lines.append("|----------|:------------------------------:|:--------------:|")
    lines.append(f"| D3 (Retrieve-then-filter) | {d3_leak_on_expected}/{n_expected} | {_safe_rate(d3_leak_on_expected, n_expected)} |")
    lines.append(f"| **D6 (AFR)** | **{d6_leak_on_expected}/{n_expected}** | **{_safe_rate(d6_leak_on_expected, n_expected)}** |")
    lines.append("")

    # Table 3: Latency (only for full runs)
    if not dry_run_mode and d3_times and d6_times:
        lines.append("## Table 3: Latency Comparison")
        lines.append("")
        lines.append("| Pipeline | Avg (ms) | Median (ms) | P95 (ms) | Avg Retrieval (ms) | Avg Generation (ms) |")
        lines.append("|----------|:--------:|:-----------:|:--------:|:------------------:|:-------------------:|")
        lines.append(
            f"| D3 (all) | {np.mean(d3_times):.0f} | {np.median(d3_times):.0f} | "
            f"{np.percentile(d3_times, 95):.0f} | {np.mean(d3_retrieval_times):.0f} | "
            f"{np.mean(d3_gen_times):.0f} |"
        )
        lines.append(
            f"| **D6 (all)** | **{np.mean(d6_times):.0f}** | **{np.median(d6_times):.0f}** | "
            f"**{np.percentile(d6_times, 95):.0f}** | **{np.mean(d6_retrieval_times):.0f}** | "
            f"**{np.mean(d6_gen_times):.0f}** |"
        )
        # Allow-only latency rows
        if d3_allow_times and d6_allow_times:
            lines.append(
                f"| D3 (allow only) | {np.mean(d3_allow_times):.0f} | {np.median(d3_allow_times):.0f} | "
                f"{np.percentile(d3_allow_times, 95):.0f} | — | — |"
            )
            lines.append(
                f"| **D6 (allow only)** | **{np.mean(d6_allow_times):.0f}** | **{np.median(d6_allow_times):.0f}** | "
                f"**{np.percentile(d6_allow_times, 95):.0f}** | **—** | **—** |"
            )
        lines.append("")
        savings = np.mean(d3_times) - np.mean(d6_times)
        if savings > 0:
            lines.append(f"> AFR reduced end-to-end latency by **{savings:.0f}ms** on average.")
        else:
            lines.append(f"> AFR added **{-savings:.0f}ms** average overhead over D3.")
        lines.append("> Allow-only rows compare latency on queries where D6 returns an answer (excludes deny queries with near-zero generation time).")
        lines.append("> D3 passes all retrieved chunks through **two** LLM calls (notes extraction + answer generation), while D6 uses a **single** LLM call over only authorized chunks.")
        lines.append("")

        # Context Exposure table
        d3_chunk_counts = []
        d6_chunk_counts = []
        d3_token_counts = []
        d6_token_counts = []
        for r in results:
            d3 = r.get("d3")
            d6 = r.get("d6")
            if d3:
                ctx = d3.get("context_chunk_ids", [])
                d3_chunk_counts.append(len(ctx))
            if d6:
                ctx = d6.get("context_chunk_ids", [])
                d6_chunk_counts.append(len(ctx))

        # Estimate tokens from answer length as proxy (actual context not stored)
        # Use generation_time as a relative proxy for context size
        if d3_chunk_counts and d6_chunk_counts:
            # Separate allow vs deny chunk counts for D6
            d6_allow_chunks = []
            d6_deny_chunks = []
            d3_retrieved_counts = []
            for r in results:
                d3 = r.get("d3")
                d6 = r.get("d6")
                exp_leak = r["query"]["expected_leak"]
                if d3:
                    d3_retrieved_counts.append(len(d3.get("retrieved_chunk_ids", [])))
                if d6:
                    n_ctx = len(d6.get("context_chunk_ids", []))
                    if exp_leak:
                        d6_deny_chunks.append(n_ctx)
                    else:
                        d6_allow_chunks.append(n_ctx)

            lines.append("### Context Exposure")
            lines.append("")
            lines.append("| Pipeline | Scope | Avg Retrieved | Avg LLM Context | LLM Calls |")
            lines.append("|----------|-------|:-------------:|:---------------:|:---------:|")
            d3_avg_c = np.mean(d3_chunk_counts)
            d3_avg_r = np.mean(d3_retrieved_counts) if d3_retrieved_counts else d3_avg_c
            d6_avg_c = np.mean(d6_chunk_counts)
            d6_allow_avg = np.mean(d6_allow_chunks) if d6_allow_chunks else 0
            d6_deny_avg = np.mean(d6_deny_chunks) if d6_deny_chunks else 0
            lines.append(f"| D3 | All queries | {d3_avg_r:.1f} | {d3_avg_c:.1f} | 2 |")
            lines.append(f"| **D6** | **All queries** | **{d3_avg_r:.1f}** | **{d6_avg_c:.1f}** | **1** |")
            lines.append(f"| **D6** | **Allow only** | **—** | **{d6_allow_avg:.1f}** | **1** |")
            lines.append(f"| **D6** | **Deny only** | **—** | **{d6_deny_avg:.1f}** | **1** |")
            lines.append("")
            lines.append(f"> D3 exposes all **{d3_avg_c:.0f}** retrieved chunks (including unauthorized) across 2 LLM calls.")
            lines.append(f"> D6 passes only **{d6_allow_avg:.0f}** authorized chunks to the LLM on allow queries; deny queries see **{d6_deny_avg:.0f}** chunks (refusal context only).")
            lines.append("")
    elif dry_run_mode and d3_times and d6_times:
        lines.append("## Table 3: Structural Filtering Latency")
        lines.append("")
        lines.append("| Pipeline | Avg (ms) | Median (ms) | P95 (ms) |")
        lines.append("|----------|:--------:|:-----------:|:--------:|")
        lines.append(
            f"| D3 | {np.mean(d3_times):.0f} | {np.median(d3_times):.0f} | "
            f"{np.percentile(d3_times, 95):.0f} |"
        )
        lines.append(
            f"| **D6 (AFR)** | **{np.mean(d6_times):.0f}** | **{np.median(d6_times):.0f}** | "
            f"**{np.percentile(d6_times, 95):.0f}** |"
        )
        lines.append("")

    # Table 4: By Category
    lines.append("## Table 4: Breakdown by Query Category")
    lines.append("")
    lines.append("| Category | Queries | Expected Deny | D3 Structural Leaks | D6 Structural Leaks |")
    lines.append("|----------|:-------:|:-------------:|:-------------------:|:-------------------:|")
    cat_order = [
        "direct_leak", "indirect_leak",
        "benign_clean", "benign_ambiguous",
        "cross_domain_deny", "cross_domain_allow",
    ]
    cat_labels = {
        "direct_leak": "Direct Leak",
        "indirect_leak": "Indirect Leak",
        "benign_clean": "Benign (Clean)",
        "benign_ambiguous": "Benign (Ambiguous)",
        "cross_domain_deny": "Cross-Domain Deny",
        "cross_domain_allow": "Cross-Domain Allow",
    }
    for cat in cat_order:
        if cat in categories:
            c = categories[cat]
            lines.append(
                f"| {cat_labels.get(cat, cat)} | {c['total']} | {c['expected_leaks']} | "
                f"{c['d3_structural']} | {c['d6_structural']} |"
            )
    lines.append("")

    # Table 5: Answer Leak Classification (full run only)
    if not dry_run_mode:
        lines.append("## Table 5: Answer Leak Classification")
        lines.append("")
        lines.append("3-way classification of keyword-matched answer leaks:")
        lines.append("")
        lines.append("| Pipeline | Grounded Leaks | Ungrounded Guesses | False Positives | Total Keyword Hits |")
        lines.append("|----------|:--------------:|:------------------:|:---------------:|:------------------:|")
        d3_kw_total = d3_grounded + d3_ungrounded + d3_false_positive
        d6_kw_total = d6_grounded + d6_ungrounded + d6_false_positive
        lines.append(
            f"| D3 (Retrieve-then-filter) | {d3_grounded} | {d3_ungrounded} | "
            f"{d3_false_positive} | {d3_kw_total} |"
        )
        lines.append(
            f"| **D6 (AFR)** | **{d6_grounded}** | **{d6_ungrounded}** | "
            f"**{d6_false_positive}** | **{d6_kw_total}** |"
        )
        lines.append("")
        lines.append("> **Grounded Leak**: answer contains restricted keyword AND unauthorized chunks were in context (real data leak).")
        lines.append("> **Ungrounded Guess**: answer contains restricted keyword but NO unauthorized context (LLM hallucination).")
        lines.append("> **False Positive**: keyword detector triggered on a refusal or query-echo (not an actual leak).")
        lines.append("")

    # Detailed leak log
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Leak Log (D3 Pipeline)")
    lines.append("")
    lines.append("Queries where D3 allowed unauthorized chunks into the LLM context:")
    lines.append("")

    leak_entries = [
        r for r in results
        if r.get("d3", {}).get("structural_leak")
    ]
    if leak_entries:
        lines.append("| Query ID | Query | Role | Unauthorized Chunks |")
        lines.append("|----------|-------|------|---------------------|")
        for r in leak_entries:
            q = r["query"]
            d3 = r["d3"]
            unauth = ", ".join(d3.get("unauthorized_in_context", []))
            query_text = q["query"][:50] + "..." if len(q["query"]) > 50 else q["query"]
            lines.append(f"| {q['query_id']} | {query_text} | {q['test_role']} | {unauth} |")
    else:
        lines.append("*No structural leaks detected.*")
    lines.append("")

    report = "\n".join(lines)

    # Save report
    report_path = os.path.join(output_dir, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"✓ Report saved to {report_path}")

    # Save CSV summary
    csv_path = os.path.join(output_dir, "summary.csv")
    _save_csv_summary(results, csv_path, dry_run_mode)
    print(f"✓ CSV summary saved to {csv_path}")

    return report


def _save_csv_summary(results: List[Dict], path: str, dry_run: bool) -> None:
    """Save per-query results as CSV."""

    fieldnames = [
        "query_id", "category", "query", "role", "expected_leak",
        "d3_structural_leak", "d6_structural_leak",
        "d3_time_ms", "d6_time_ms",
    ]
    if not dry_run:
        fieldnames.extend([
            "d3_answer_leak", "d6_answer_leak",
            "d3_true_leak_keywords", "d6_true_leak_keywords",
            "d3_false_positive_keywords", "d6_false_positive_keywords",
        ])

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            q = r["query"]
            d3 = r.get("d3") or {}
            d6 = r.get("d6") or {}

            row = {
                "query_id": q["query_id"],
                "category": q["category"],
                "query": q["query"],
                "role": q["test_role"],
                "expected_leak": q["expected_leak"],
                "d3_structural_leak": d3.get("structural_leak", ""),
                "d6_structural_leak": d6.get("structural_leak", ""),
                "d3_time_ms": f"{d3.get('total_time_ms', 0):.0f}",
                "d6_time_ms": f"{d6.get('total_time_ms', 0):.0f}",
            }
            if not dry_run:
                d3_grounded = d3.get("grounded_keywords", []) + d3.get("ungrounded_keywords", [])
                d6_grounded = d6.get("grounded_keywords", []) + d6.get("ungrounded_keywords", [])
                row["d3_answer_leak"] = d3.get("answer_leak", "")
                row["d6_answer_leak"] = d6.get("answer_leak", "")
                row["d3_true_leak_keywords"] = "; ".join(d3_grounded)
                row["d6_true_leak_keywords"] = "; ".join(d6_grounded)
                row["d3_false_positive_keywords"] = "; ".join(d3.get("false_positive_keywords", []))
                row["d6_false_positive_keywords"] = "; ".join(d6.get("false_positive_keywords", []))

            writer.writerow(row)


def _get_timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _generate_stress_table(
    baseline_results: List[Dict],
    stress_results: List[Dict],
    stress_k: int,
    dry_run: bool,
) -> str:
    """Generate a Table 6: Stress Test comparison (k=10 vs k=N)."""
    lines = []
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"## Table 6: Stress Test (k=10 vs k={stress_k})")
    lines.append("")
    lines.append(f"> With {stress_k} > corpus size (24 chunks), retrieval returns the **entire corpus**.")
    lines.append("> D3 exposes every document to the LLM; D6 must still filter correctly.")
    lines.append("")

    total = len(stress_results)
    # Baseline (k=10)
    b_d3_struct = sum(1 for r in baseline_results if r.get("d3", {}).get("structural_leak"))
    b_d6_struct = sum(1 for r in baseline_results if r.get("d6", {}).get("structural_leak"))
    # Stress (k=N)
    s_d3_struct = sum(1 for r in stress_results if r.get("d3", {}).get("structural_leak"))
    s_d6_struct = sum(1 for r in stress_results if r.get("d6", {}).get("structural_leak"))

    if dry_run:
        lines.append("| Config | D3 Structural Leaks | D6 Structural Leaks | D3 Rate | D6 Rate |")
        lines.append("|--------|:-------------------:|:-------------------:|:-------:|:-------:|")
        lines.append(
            f"| k=10 (baseline) | {b_d3_struct}/{total} | {b_d6_struct}/{total} | "
            f"{_safe_rate(b_d3_struct, total)} | {_safe_rate(b_d6_struct, total)} |"
        )
        lines.append(
            f"| **k={stress_k} (stress)** | **{s_d3_struct}/{total}** | **{s_d6_struct}/{total}** | "
            f"**{_safe_rate(s_d3_struct, total)}** | **{_safe_rate(s_d6_struct, total)}** |"
        )
    else:
        b_d3_answer = sum(1 for r in baseline_results if r.get("d3", {}).get("answer_leak"))
        b_d6_answer = sum(1 for r in baseline_results if r.get("d6", {}).get("answer_leak"))
        s_d3_answer = sum(1 for r in stress_results if r.get("d3", {}).get("answer_leak"))
        s_d6_answer = sum(1 for r in stress_results if r.get("d6", {}).get("answer_leak"))

        lines.append("| Config | D3 Structural | D6 Structural | D3 Answer | D6 Answer | D3 Rate | D6 Rate |")
        lines.append("|--------|:-------------:|:-------------:|:---------:|:---------:|:-------:|:-------:|")
        b_d3_any = sum(1 for r in baseline_results if r.get("d3", {}).get("structural_leak") or r.get("d3", {}).get("answer_leak"))
        b_d6_any = sum(1 for r in baseline_results if r.get("d6", {}).get("structural_leak") or r.get("d6", {}).get("answer_leak"))
        s_d3_any = sum(1 for r in stress_results if r.get("d3", {}).get("structural_leak") or r.get("d3", {}).get("answer_leak"))
        s_d6_any = sum(1 for r in stress_results if r.get("d6", {}).get("structural_leak") or r.get("d6", {}).get("answer_leak"))
        lines.append(
            f"| k=10 (baseline) | {b_d3_struct}/{total} | {b_d6_struct}/{total} | "
            f"{b_d3_answer}/{total} | {b_d6_answer}/{total} | "
            f"{_safe_rate(b_d3_any, total)} | {_safe_rate(b_d6_any, total)} |"
        )
        lines.append(
            f"| **k={stress_k} (stress)** | **{s_d3_struct}/{total}** | **{s_d6_struct}/{total}** | "
            f"**{s_d3_answer}/{total}** | **{s_d6_answer}/{total}** | "
            f"**{_safe_rate(s_d3_any, total)}** | **{_safe_rate(s_d6_any, total)}** |"
        )

    # Latency comparison
    b_d6_times = [r["d6"]["total_time_ms"] for r in baseline_results if r.get("d6")]
    s_d6_times = [r["d6"]["total_time_ms"] for r in stress_results if r.get("d6")]
    if b_d6_times and s_d6_times:
        lines.append("")
        lines.append(f"> D6 avg latency: k=10 → **{np.mean(b_d6_times):.0f}ms**, k={stress_k} → **{np.mean(s_d6_times):.0f}ms**.")

    lines.append("")
    if s_d6_struct == 0:
        lines.append(f"> ✅ **D6 maintains 0 structural leaks at k={stress_k}**, confirming that AFR's authorization boundary holds even when retrieval returns the full corpus.")
    else:
        lines.append(f"> ⚠ D6 had {s_d6_struct} structural leak(s) at k={stress_k}. Investigate.")
    lines.append("")

    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AFR Results Aggregator")
    parser.add_argument(
        "--input", type=str, default="evaluation/results/raw_results.json",
        help="Path to raw_results.json",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: same as input)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found. Run `python -m evaluation.run_eval` first.")
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(args.input)
    results = load_results(args.input)

    # Detect if this was a dry-run (check if any answer contains "DRY RUN")
    dry_run = any(
        r.get("d3", {}).get("answer", "").startswith("[DRY RUN")
        for r in results
    )

    print(f"Loaded {len(results)} query results from {args.input}")
    print(f"Detected mode: {'Dry Run' if dry_run else 'Full'}")
    print()

    report = generate_report(results, output_dir, dry_run)

    # Auto-detect stress test results and append Table 6
    input_dir = os.path.dirname(args.input)
    import glob
    stress_files = sorted(glob.glob(os.path.join(input_dir, "raw_results_k*.json")))
    for sf in stress_files:
        # Extract k value from filename (e.g. raw_results_k30.json → 30)
        basename = os.path.basename(sf)
        k_str = basename.replace("raw_results_k", "").replace(".json", "")
        try:
            stress_k = int(k_str)
        except ValueError:
            continue
        stress_results = load_results(sf)
        stress_dry_run = any(
            r.get("d3", {}).get("answer", "").startswith("[DRY RUN")
            for r in stress_results
        )
        stress_table = _generate_stress_table(results, stress_results, stress_k, stress_dry_run)
        # Append to report file
        report_path = os.path.join(output_dir, "evaluation_report.md")
        with open(report_path, "a") as f:
            f.write(stress_table)
        print(f"✓ Appended stress test table (k={stress_k}) to report")

    print()
    print("Done! Open evaluation_report.md to see paper-ready tables.")


if __name__ == "__main__":
    main()
