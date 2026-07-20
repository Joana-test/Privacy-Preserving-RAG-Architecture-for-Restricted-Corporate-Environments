"""
Evaluation of AFR baseline, no misclassification (epsilon = 0), no Security Layer.
 
Establishes the clean reference point of the thesis evaluation: how does
plain Authorization-First Retrieval (pipeline variant D6 of Namboothiri) perform when the
underlying sensitivity/domain tags are correct?
 
No corruption is injected, so a single deterministic run is sufficient --
no seed sweep is needed here (unlike the epsilon sweep in
eval_misclassification.py).
 
Author: J. Fermin, Master's thesis, LMU München, 2026.
Uses evaluation infrastructure adapted from Namboothiri et al. (2026),
licensed under the Apache License, Version 2.0.
 
Usage:
    python -m evaluation.eval_afr_baseline \\
        --output-dir evaluation/results/fr_baseline
"""

import argparse

from afr.rag import RAGPipeline
from evaluation.test_corpus import get_test_corpus, get_corpus_access_map, get_leak_keywords
from evaluation.query_set import get_query_set
from evaluation.eval_combined_baseline import inject_corpus
from evaluation.eval_common import (
    get_ingester, reset_ingester, run_single_query_d6, save_results,
)


def main():
    parser = argparse.ArgumentParser(description="Phase 1: AFR baseline (epsilon=0)")
    parser.add_argument("--output-dir", type=str,
                        default="evaluation/results/afr_baseline")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 1: AFR Baseline (epsilon = 0, no Security Layer)")
    print("=" * 70)

    # Load the evaluation corpus (clean labels) and the query set.
    corpus = get_test_corpus()
    access_map = get_corpus_access_map()
    leak_keywords = get_leak_keywords()
    queries = get_query_set()
    print(f"Corpus: {len(corpus)} chunks | Queries: {len(queries)}")

    # Build the FAISS index over the uncorrupted corpus.
    reset_ingester()
    ingester = get_ingester()
    inject_corpus(ingester, corpus)

    pipeline = RAGPipeline()

    # Run every query once through the AFR-only pipeline (D6).
    results = []
    for i, q in enumerate(queries, 1):
        print(f"  [{i:2d}/{len(queries)}] {q.query_id} ({q.category:18s}) role={q.test_role}")
        results.append(run_single_query_d6(pipeline, q, access_map, leak_keywords))

    n_struct = sum(r["structural_leak"] for r in results)
    n_answer = sum(r["answer_leak"] for r in results)

    print(f"\n{'-' * 70}")
    print(f"Structural leaks: {n_struct}/{len(results)} ({n_struct / len(results) * 100:.1f}%)")
    print(f"Answer leaks:     {n_answer}/{len(results)} ({n_answer / len(results) * 100:.1f}%)")
    print(f"{'-' * 70}")

    save_results(
        {
            "phase": "1_afr_baseline",
            "epsilon": 0.0,
            "n_queries": len(queries),
            "results": results,
        },
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
