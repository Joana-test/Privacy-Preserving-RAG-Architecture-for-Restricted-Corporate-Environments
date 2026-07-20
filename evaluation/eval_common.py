"""
Shared utilities for the evaluation scripts: deterministic misclassification
injection, per-query runners for both pipeline variants, Security-Layer
parse-failure detection, and result serialization with run metadata.

Author: J. Fermin, Master's thesis, LMU München, 2026.
Uses evaluation infrastructure adapted from Namboothiri et al. (2026),
licensed under the Apache License, Version 2.0.

Imported by eval_afr_baseline.py and eval_misclassification.py; not run
directly.
"""

import copy
import hashlib
import json
import os
import random
import subprocess
from datetime import datetime
from typing import Optional

from afr.ingest import get_ingester, reset_ingester
from afr.rag import SECURITY_LAYER_USER
from evaluation.eval_combined_baseline import check_structural_leak, check_answer_leak

__all__ = [
    "get_ingester", "reset_ingester",
    "inject_misclassification", "prompt_hash", "git_commit",
    "is_sl_parse_failure", "run_single_query_d6", "run_single_query_d6_sl",
    "save_results", "summarize_confusion",
]

# ── Misclassification injection (unchanged from stress_test_mini.py) ─────────

DOWNGRADE_MAP = {
    "confidential": ["restricted", "public"],
    "restricted":   ["public"],
    "public":       [],
}


def inject_misclassification(chunks, error_rate: float, seed: int = 42):
    """Randomly downgrades chunk sensitivity to simulate AFR tagging errors.

    Kept functionally equivalent to stress_test_mini.py's logic so that a
    given (error_rate, seed) pair produces the exact same corrupted corpus
    in every phase -- results stay comparable across phases and across your
    earlier stresstest reports.

    Returns:
        corrupted: the corrupted chunk list
        n: number of chunks that were downgraded
        detail: list of {"chunk_id", "original_sensitivity", "corrupted_to"}
                for each downgraded chunk -- lets you trace exactly which
                chunk was responsible for a given leak example, instead of
                only knowing a leak occurred somewhere. Phase 2 callers can
                safely ignore this third return value.
    """
    random.seed(seed)
    corrupted = copy.deepcopy(chunks)
    n = 0
    detail = []
    for chunk in corrupted:
        possible = DOWNGRADE_MAP.get(chunk.sensitivity, [])
        if possible and random.random() < error_rate:
            original = chunk.sensitivity
            new_sensitivity = random.choice(possible)
            chunk.sensitivity = new_sensitivity
            n += 1
            detail.append({
                "chunk_id": chunk.chunk_id,
                "original_sensitivity": original,
                "corrupted_to": new_sensitivity,
            })
    return corrupted, n, detail


# ── Run metadata (so results are traceable to the exact prompt/code state) ───

def prompt_hash() -> str:
    """Short hash of the active SECURITY_LAYER_USER prompt text.
    Lets you tell apart two result files that both claim the same
    'prompt version' but were actually generated with slightly
    different prompt text."""
    return hashlib.md5(SECURITY_LAYER_USER.encode()).hexdigest()[:8]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"


def is_sl_parse_failure(sl_result) -> bool:
    """chat_json() in llm_client.py silently falls back to
    {"verdict": "UNSAFE", "reason": "Parse error: ..."} when Llama does not
    return valid JSON."""
    return bool(
        sl_result and sl_result.reason
        and sl_result.reason.startswith("Parse error:")
    )


# ── Per-query runners ──────────────────────────────────────────────────────

def run_single_query_d6(pipeline, q, access_map, leak_keywords, k: int = 10) -> dict:
    """Runs one query through AFR alone (no Security Layer)."""
    r = pipeline.strict_afr_rag(q.query, q.test_role, k=k)
    ctx = [c.chunk_id for c in r.sources]
    unauth = check_structural_leak(ctx, q.test_role, access_map)
    leak_info = check_answer_leak(r.answer, q, leak_keywords, unauth)
    real_leak = leak_info["grounded_keywords"] + leak_info["ungrounded_keywords"]

    return {
        "query_id": q.query_id,
        "query": q.query,
        "category": q.category,
        "role": q.test_role,
        "expected_decision": q.expected_decision,
        "expected_leak": q.expected_leak,
        "answer": r.answer,
        "is_refusal": r.is_refusal,
        "context_chunk_ids": ctx,
        "unauthorized_chunk_ids": unauth,
        "structural_leak": len(unauth) > 0,
        "answer_leak": len(real_leak) > 0,
        "leak_classification": leak_info["classification"],
        "total_time_ms": r.metrics.total_time_ms,
    }


def run_single_query_d6_sl(pipeline, q, access_map, leak_keywords, k: int = 10) -> dict:
    """Runs one query through AFR + Security Layer (the combined model)."""
    r = pipeline.afr_with_security_layer(q.query, q.test_role, k=k)
    ctx = [c.chunk_id for c in r.sources]
    unauth = check_structural_leak(ctx, q.test_role, access_map)
    leak_info = check_answer_leak(r.answer, q, leak_keywords, unauth)
    real_leak = leak_info["grounded_keywords"] + leak_info["ungrounded_keywords"]
    sl = r.security_layer_result

    return {
        "query_id": q.query_id,
        "query": q.query,
        "category": q.category,
        "role": q.test_role,
        "expected_decision": q.expected_decision,
        "expected_leak": q.expected_leak,
        "answer": r.answer,
        "is_refusal": r.is_refusal,
        "context_chunk_ids": ctx,
        "unauthorized_chunk_ids": unauth,
        "structural_leak": len(unauth) > 0,
        "answer_leak": len(real_leak) > 0,
        "leak_classification": leak_info["classification"],
        "sl_verdict": sl.verdict if sl else None,
        "sl_reason": sl.reason if sl else None,
        "sl_violated_rule": sl.violated_rule if sl else None,
        "sl_parse_failed": is_sl_parse_failure(sl),
        "sl_time_ms": r.metrics.security_layer_time_ms,
        "total_time_ms": r.metrics.total_time_ms,
    }


# ── Confusion-matrix summary (TP/FP/FN/TN for the Security Layer) ────────────

def summarize_confusion(results: list) -> dict:
    """Only meaningful for D6_SL results (needs sl_verdict + expected_decision).
    Mirrors the definitions used in your original stresstest reports:
      TP = deny query, real leak in D6 would have occurred, SL blocked it
      FP = allow query, SL blocked it anyway (utility loss)
      FN = deny query, real leak occurred, SL did NOT block it (security gap)
      TN = allow query, SL correctly did not block it
    Note: TP/FN require knowing whether D6 (unfiltered) would have leaked --
    pass results that already include an "afr_answer_leak" field if you want
    exact TP/FN; otherwise this falls back to expected_decision only.
    """
    tp = fp = fn = tn = 0
    for r in results:
        blocked = r.get("sl_verdict") == "UNSAFE"
        expect_deny = r.get("expected_decision") == "deny"
        if expect_deny and blocked:
            tp += 1
        elif expect_deny and not blocked:
            fn += 1
        elif not expect_deny and blocked:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def to_flat_records(epsilon, seed, d6_results, d6_sl_results, corruption_detail=None):
    """Converts one (epsilon, seed) run's paired D6 / D6_SL results into the
    flat record schema expected by aggregate_stress_results.py.

    confusion_class reconstruction matches your original reports:
      allow query, SL blocked          -> FP
      allow query, SL did not block    -> TN
      deny query, D6 actually leaked, SL blocked     -> TP
      deny query, D6 actually leaked, SL did not block -> FN
      deny query, D6 did NOT actually leak -> None (excluded from the matrix,
        consistent with eps=0 reports where TP+FP+FN+TN == n_allow x n_seeds)

    corruption_detail (optional): the list returned by inject_misclassification()
    for this (epsilon, seed) -- attached to every record so you can trace,
    for any individual query, exactly which chunks were downgraded in the
    run that produced it (helpful for citing concrete examples in the thesis).
    """
    flat = []
    for d6, d6_sl in zip(d6_results, d6_sl_results):
        assert d6["query_id"] == d6_sl["query_id"], "D6/D6_SL query order mismatch"
        expect_deny = d6["expected_decision"] == "deny"
        blocked = d6_sl.get("sl_verdict") == "UNSAFE"
        d6_actual_leak = d6["answer_leak"]
        if not expect_deny:
            confusion_class = "FP" if blocked else "TN"
        elif d6_actual_leak:
            confusion_class = "TP" if blocked else "FN"
        else:
            confusion_class = None
        flat.append({
            "error_rate": epsilon,
            "seed": seed,
            "query_id": d6["query_id"],
            "query": d6.get("query", ""),
            "category": d6["category"],
            "role": d6["role"],
            "expected_decision": d6["expected_decision"],
            "confusion_class": confusion_class,
            "d6_structural_leak": d6["structural_leak"],
            "d6_sl_structural_leak": d6_sl["structural_leak"],
            "d6_answer_leak": d6["answer_leak"],
            "d6_sl_answer_leak": d6_sl["answer_leak"],
            "d6_answer_text": d6["answer"],
            "d6_context_chunk_ids": d6.get("context_chunk_ids", []),
            "d6_unauthorized_chunk_ids": d6.get("unauthorized_chunk_ids", []),
            "d6_sl_context_chunk_ids": d6_sl.get("context_chunk_ids", []),
            "d6_sl_unauthorized_chunk_ids": d6_sl.get("unauthorized_chunk_ids", []),
            "d6_sl_security_verdict": d6_sl.get("sl_verdict"),
            "d6_sl_security_reason": d6_sl.get("sl_reason"),
            "d6_sl_security_violated_rule": d6_sl.get("sl_violated_rule"),
            "d6_sl_security_parse_failed": d6_sl.get("sl_parse_failed", False),
            "d6_total_time_ms": d6["total_time_ms"],
            "d6_sl_total_time_ms": d6_sl["total_time_ms"],
            "d6_sl_security_time_ms": d6_sl.get("sl_time_ms", 0.0),
            "corrupted_chunks_this_seed": corruption_detail or [],
        })
    return flat


# ── Saving ────────────────────────────────────────────────────────────────

def save_results(payload: dict, output_dir: str, filename: str = "raw_results.json") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    full = {
        "meta": {
            "prompt_hash": prompt_hash(),
            "git_commit": git_commit(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        **payload,
    }
    with open(path, "w") as f:
        json.dump(full, f, indent=2)
    print(f"  Saved -> {path}")
    return path
