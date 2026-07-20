"""
Epsilon sweep: runs both pipeline variants on the same corrupted corpus per
(error rate, seed) combination. Produces the canonical result files from
which all reported evaluation numbers are derived.

Author: J. Fermin, Master's thesis, LMU München, 2026.
Uses evaluation infrastructure adapted from Namboothiri et al. (2026),
licensed under the Apache License, Version 2.0.

Usage (from the project root):
    python -m evaluation.eval_misclassification
    python -m evaluation.eval_misclassification --epsilons 0.2 --seeds 42

"""

import argparse
import json
import os

from afr.rag import RAGPipeline
from evaluation.test_corpus import get_test_corpus, get_corpus_access_map, get_leak_keywords
from evaluation.query_set import get_query_set
from evaluation.eval_combined_baseline import inject_corpus
from evaluation.eval_common import (
    get_ingester, reset_ingester, inject_misclassification,
    run_single_query_d6, run_single_query_d6_sl,
    summarize_confusion, save_results, to_flat_records,
)


def run_one_seed(epsilon, seed, corpus_template, access_map, leak_keywords, queries, pipeline):
    """Run one (epsilon, seed) cell: corrupt the corpus deterministically,
    rebuild the index, then run every query through both pipeline variants."""
    reset_ingester()
    ingester = get_ingester()
    corrupted, n_corrupted, corruption_detail = inject_misclassification(corpus_template, epsilon, seed=seed)
    print(f"    Corrupted chunks: {n_corrupted}/{len(corpus_template)}")
    if corruption_detail:
        for d in corruption_detail:
            print(f"      {d['chunk_id']}: {d['original_sensitivity']} -> {d['corrupted_to']}")
    inject_corpus(ingester, corrupted)

    d6_results, d6_sl_results = [], []
    for i, q in enumerate(queries, 1):
        print(f"    [{i:2d}/{len(queries)}] {q.query_id} ({q.category:18s}) role={q.test_role}")
        d6_results.append(run_single_query_d6(pipeline, q, access_map, leak_keywords))
        d6_sl_results.append(run_single_query_d6_sl(pipeline, q, access_map, leak_keywords))

    return d6_results, d6_sl_results, n_corrupted, corruption_detail


def main():
    parser = argparse.ArgumentParser(description="Phase 3: AFR + Security Layer, epsilon sweep")
    parser.add_argument("--epsilons", type=float, nargs="+",
                         default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777])
    parser.add_argument("--output-root", type=str,
                         default="evaluation/results/misclassification")
    args = parser.parse_args()

    corpus_template = get_test_corpus()
    access_map = get_corpus_access_map()
    leak_keywords = get_leak_keywords()
    queries = get_query_set()
    pipeline = RAGPipeline()

    print("=" * 70)
    print(f"PHASE 3: Combined model (AFR + Security Layer)")
    print(f"Epsilons: {args.epsilons} | Seeds: {args.seeds}")
    print("=" * 70)

    all_flat_records = []  # accumulates across all epsilons/seeds for the aggregator

    for epsilon in args.epsilons:
        print(f"\n=== epsilon = {epsilon:.0%} ===")
        by_seed = {}
        pooled_sl_results = []
        n_parse_failures = 0

        for seed in args.seeds:
            print(f"  --- seed {seed} ---")
            d6_results, d6_sl_results, n_corrupted, corruption_detail = run_one_seed(
                epsilon, seed, corpus_template, access_map, leak_keywords, queries, pipeline
            )
            d6_struct = sum(r["structural_leak"] for r in d6_results)
            d6_answer = sum(r["answer_leak"] for r in d6_results)
            sl_struct = sum(r["structural_leak"] for r in d6_sl_results)
            sl_answer = sum(r["answer_leak"] for r in d6_sl_results)
            seed_parse_fail = sum(r["sl_parse_failed"] for r in d6_sl_results)
            n_parse_failures += seed_parse_fail

            print(f"    -> D6:    structural {d6_struct}/{len(queries)}, answer {d6_answer}/{len(queries)}")
            print(f"    -> D6_SL: structural {sl_struct}/{len(queries)}, answer {sl_answer}/{len(queries)}"
                  f"  (SL parse failures: {seed_parse_fail})")

            by_seed[str(seed)] = {
                "n_corrupted_chunks": n_corrupted,
                "corruption_detail": corruption_detail,
                "d6_results": d6_results,
                "d6_sl_results": d6_sl_results,
            }
            pooled_sl_results.extend(d6_sl_results)
            all_flat_records.extend(
                to_flat_records(epsilon, seed, d6_results, d6_sl_results, corruption_detail)
            )

        confusion = summarize_confusion(pooled_sl_results)
        print(f"    Pooled SL confusion matrix: {confusion}")
        if n_parse_failures:
            print(f"    NOTE: {n_parse_failures} SL parse failures fell back to SAFE "
                  f"-- see sl_parse_failed field per query.")

        save_results(
            {
                "phase": "3_combined_stress",
                "epsilon": epsilon,
                "seeds": args.seeds,
                "n_queries_per_seed": len(queries),
                "pooled_sl_confusion": confusion,
                "pooled_sl_parse_failures": n_parse_failures,
                "by_seed": by_seed,
            },
            output_dir=f"{args.output_root}/eps{int(round(epsilon*100)):02d}",
        )

    flat_path = os.path.join(args.output_root, "stress_test_raw.json")
    os.makedirs(args.output_root, exist_ok=True)
    with open(flat_path, "w") as f:
        json.dump(all_flat_records, f, indent=2)
    print(f"\nFlat aggregator-compatible file written -> {flat_path}")
    print(f"Run: python -m evaluation.analysis")


if __name__ == "__main__":
    main()
