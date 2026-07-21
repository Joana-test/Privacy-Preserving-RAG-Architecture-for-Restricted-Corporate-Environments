"""
Re-classifies existing D6 answers with a different Security-Layer model
(filter regime of the model comparison).

Author: J. Fermin, Master's thesis, LMU München, 2026.
Uses evaluation infrastructure adapted from Namboothiri et al. (2026),
licensed under the Apache License, Version 2.0.

Usage:
    python -m evaluation.model_comparison \\
        --model-path ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \\
        --model-name "Llama-3.1-8B-Q8_0" \\
        --eps-dirs eps00 eps10 eps20 eps30 eps40 eps50 \\
        --results-root evaluation/results/misclassification \\
        --output evaluation/results/model_compare/llamaQ8/Q8.json
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from llama_cpp import Llama

# Prompt templates and the policy lookup are imported from the production
# code so that the re-classification uses byte-identical prompts. This also
# keeps the prompt_hash recorded in eval_common.py meaningful across regimes.
from afr.rag import SECURITY_LAYER_SYSTEM, SECURITY_LAYER_USER
from afr.policies import get_policy

CROSS_DOMAIN_ALLOW_CATEGORY = "cross_domain_allow"


def format_policy_fields(role: str) -> tuple[str, str]:
    """Returns (allowed_sensitivity, allowed_domains) as strings, exactly as
    they appear in the Security-Layer prompt. Handles both list and string
    representations returned by get_policy()."""
    policy = get_policy(role)
    allowed_sensitivity = getattr(policy, "allowed_sensitivity", None)
    allowed_domains = getattr(policy, "allowed_domains", None)
    if isinstance(allowed_sensitivity, (list, tuple)):
        allowed_sensitivity = ", ".join(allowed_sensitivity)
    if isinstance(allowed_domains, (list, tuple)):
        allowed_domains = ", ".join(allowed_domains)
    return str(allowed_sensitivity), str(allowed_domains)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def classify_with_sl(llm: Llama, answer: str, role: str,
                     max_tokens: int = 256) -> dict:
    """Builds the Security-Layer prompt (system + user) and calls the given
    model via chat completion, mirroring the invocation in rag.py with the
    same sampling parameters."""
    allowed_sensitivity, allowed_domains = format_policy_fields(role)

    user_prompt = SECURITY_LAYER_USER.format(
        role=role,
        allowed_sensitivity=allowed_sensitivity,
        allowed_domains=allowed_domains,
        answer=answer,
    )

    t0 = time.time()
    raw = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SECURITY_LAYER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    elapsed_ms = (time.time() - t0) * 1000

    text = raw["choices"][0]["message"]["content"].strip()

    # Identical to chat_json() in the production client: strip markdown
    # fences, then parse JSON. On a parse error the same fail-secure
    # fallback applies (Saltzer & Schroeder, 1975): UNSAFE, since a parse
    # error is a system failure, not a content decision.
    parse_failed = False
    try:
        clean = text
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        parsed = json.loads(clean.strip())
        verdict = parsed.get("verdict", "UNSAFE")
        violated_rule = parsed.get("violated_rule")
        reason = parsed.get("reason", text)
    except Exception:
        parse_failed = True
        verdict = "UNSAFE"
        violated_rule = None
        reason = f"Parse error: {text[:100]}"

    return {
        "sl_verdict": verdict,
        "sl_violated_rule": violated_rule,
        "sl_reason": reason,
        "sl_parse_failed": parse_failed,
        "sl_time_ms": elapsed_ms,
    }


def process_epsilon(llm: Llama, eps_dir: str, results_root: str, model_name: str,
                    checkpoint_path: Optional[str] = None):
    path = os.path.join(results_root, eps_dir, "raw_results.json")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping.")
        return None

    data = load_json(path)
    eps = data["epsilon"]

    per_seed_out = {}
    confusion = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    fp_violated_rule = defaultdict(int)
    fp_cross_domain = 0
    fp_rest = 0
    allow_cd_n = 0
    allow_rest_n = 0
    latencies = []

    # RESUME: reload seeds already completed in an earlier checkpoint to
    # avoid recomputing them, which is expensive for slower models.
    already_done_seeds = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path) as f:
                prev = json.load(f)
            prev_seeds = prev.get("per_seed_detail_so_far", {})
            for seed, seed_out in prev_seeds.items():
                per_seed_out[seed] = seed_out
                already_done_seeds.add(seed)
                for entry in seed_out:
                    # Reconstruct the confusion matrix and FP statistics from
                    # the loaded results so that the totals remain correct.
                    expect_deny = entry["expected_decision"] == "deny"
                    blocked = entry["new_sl_verdict"] == "UNSAFE"
                    d6_leak = entry.get("d6_answer_leak", False)
                    is_cross_domain = entry.get("category") == CROSS_DOMAIN_ALLOW_CATEGORY
                    if not expect_deny:
                        if is_cross_domain:
                            allow_cd_n += 1
                        else:
                            allow_rest_n += 1
                        if blocked:
                            confusion["FP"] += 1
                            rule = entry.get("new_sl_violated_rule") or "null"
                            fp_violated_rule[rule] += 1
                            if is_cross_domain:
                                fp_cross_domain += 1
                            else:
                                fp_rest += 1
                        else:
                            confusion["TN"] += 1
                    elif d6_leak:
                        confusion["TP" if blocked else "FN"] += 1
                    if entry.get("new_sl_time_ms"):
                        latencies.append(entry["new_sl_time_ms"])
            if already_done_seeds:
                print(f"  RESUME: loaded {len(already_done_seeds)} completed "
                      f"seed(s) from checkpoint: {sorted(already_done_seeds)}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  WARNING: could not load checkpoint ({e!r}), "
                  f"restarting this epsilon level from scratch.")

    for seed, seed_data in data["by_seed"].items():
        if seed in already_done_seeds:
            print(f"  Seed {seed} already done (resume), skipping.")
            continue

        d6_results = seed_data["d6_results"]
        d6_sl_results = seed_data["d6_sl_results"]

        seed_out = []
        for r in d6_sl_results:
            # IMPORTANT: the answer text must NOT come from r["answer"]. For
            # blocked queries, d6_sl_results["answer"] holds only the generic
            # block message, not the answer the Security Layer originally
            # judged. The real, unblocked answer is in d6_results (never
            # blocked) under the same query_id -- the same pairing that is
            # used for answer_leak in the confusion-matrix aggregation.
            d6_match = next((d for d in d6_results if d["query_id"] == r["query_id"]), None)
            if d6_match is None:
                print(f"WARNING: no d6_results counterpart for {r['query_id']}, "
                      f"skipping.")
                continue
            answer_text = d6_match["answer"]

            if "role" not in r:
                raise KeyError(
                    f"Query {r.get('query_id', '?')}: field 'role' missing in "
                    f"d6_sl_results."
                )
            role = r["role"]

            new_verdict = classify_with_sl(llm, answer_text, role)
            latencies.append(new_verdict["sl_time_ms"])
            print(f"    [{r['query_id']}] {new_verdict['sl_verdict']:>7s}  "
                  f"({new_verdict['sl_time_ms']/1000:.1f}s)"
                  f"{'  [PARSE FAIL]' if new_verdict['sl_parse_failed'] else ''}")

            expect_deny = r["expected_decision"] == "deny"
            blocked = new_verdict["sl_verdict"] == "UNSAFE"
            d6_leak = d6_match["answer_leak"]

            is_cross_domain = r.get("category") == CROSS_DOMAIN_ALLOW_CATEGORY

            if not expect_deny:
                if is_cross_domain:
                    allow_cd_n += 1
                else:
                    allow_rest_n += 1

                if blocked:
                    confusion["FP"] += 1
                    rule = new_verdict["sl_violated_rule"] or "null"
                    fp_violated_rule[rule] += 1
                    if is_cross_domain:
                        fp_cross_domain += 1
                    else:
                        fp_rest += 1
                else:
                    confusion["TN"] += 1
            elif d6_leak:
                confusion["TP" if blocked else "FN"] += 1
            # deny query without an actual leak -> excluded from the matrix

            seed_out.append({
                "query_id": r["query_id"],
                "category": r.get("category"),
                "role": role,
                "expected_decision": r["expected_decision"],
                "d6_answer": answer_text,       # the answer actually evaluated
                "d6_answer_leak": d6_leak,      # ground-truth leak status of that answer
                "new_sl_verdict": new_verdict["sl_verdict"],
                "new_sl_violated_rule": new_verdict["sl_violated_rule"],
                "new_sl_reason": new_verdict["sl_reason"],
                "new_sl_parse_failed": new_verdict["sl_parse_failed"],
                "new_sl_time_ms": new_verdict["sl_time_ms"],
                "old_sl_verdict": r.get("sl_verdict"),
                "old_sl_violated_rule": r.get("sl_violated_rule"),
                "old_sl_reason": r.get("sl_reason"),
            })

        per_seed_out[seed] = seed_out
        print(f"  Seed {seed} done ({len(d6_sl_results)} queries)")

        if checkpoint_path:
            with open(checkpoint_path, "w") as f:
                json.dump({
                    "model_name": model_name, "epsilon": eps,
                    "status": "in_progress", "seeds_done": list(per_seed_out.keys()),
                    "per_seed_detail_so_far": per_seed_out,
                }, f, indent=2)

    tp, fp, fn, tn = confusion["TP"], confusion["FP"], confusion["FN"], confusion["TN"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    cd_fpr = fp_cross_domain / allow_cd_n if allow_cd_n else 0.0
    rest_fpr = fp_rest / allow_rest_n if allow_rest_n else 0.0

    print(f"\neps={eps:.0%} [{model_name}]  TP={tp} FP={fp} FN={fn} TN={tn}  "
          f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}")
    print(f"  Cross-domain FPR: {fp_cross_domain}/{allow_cd_n} = {cd_fpr:.1%}   "
          f"Rest FPR: {fp_rest}/{allow_rest_n} = {rest_fpr:.1%}")
    if latencies:
        print(f"  SL latency: mean={statistics.mean(latencies):.0f}ms  "
              f"median={statistics.median(latencies):.0f}ms")

    return {
        "epsilon": eps,
        "model_name": model_name,
        "confusion": confusion,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_violated_rule": dict(fp_violated_rule),
        "fp_cross_domain": fp_cross_domain,
        "fp_rest": fp_rest,
        "allow_cross_domain_n": allow_cd_n,
        "allow_rest_n": allow_rest_n,
        "cross_domain_fpr": cd_fpr,
        "rest_fpr": rest_fpr,
        "sl_latency_mean_ms": statistics.mean(latencies) if latencies else None,
        "per_seed_detail": per_seed_out,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Re-classifies existing D6 answers with a different model"
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True,
                        help="Short name for the output, e.g. 'Llama-3.1-8B-Q8_0'")
    parser.add_argument("--eps-dirs", nargs="+",
                        default=["eps00", "eps10", "eps20", "eps30", "eps40", "eps50"],
                        help="Which epsilon directories to process")
    parser.add_argument("--results-root", type=str,
                        default="evaluation/results/misclassification")
    parser.add_argument("--output", type=str, required=True,
                        help="Path for the result JSON")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=12)
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    t0 = time.time()
    llm = Llama(
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=0,  # CPU-only, matching the production client
        verbose=False,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    all_results = []
    for eps_dir in args.eps_dirs:
        print(f"\n{'=' * 70}\nProcessing {eps_dir}\n{'=' * 70}")
        result = process_epsilon(
            llm, eps_dir, args.results_root, args.model_name,
            checkpoint_path=args.output.replace(".json", f"_{eps_dir}_checkpoint.json"),
        )
        if result:
            all_results.append(result)

            # Save after every epsilon level: with runtimes of 15h and more,
            # an interruption (dropped SSH connection, reboot) must not
            # destroy the progress made so far.
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            partial_path = args.output.replace(".json", "_partial.json")
            with open(partial_path, "w") as f:
                json.dump({"model_name": args.model_name, "results": all_results}, f, indent=2)
            print(f"  [Intermediate state saved -> {partial_path}]")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"model_name": args.model_name, "results": all_results}, f, indent=2)

    print(f"\nResults saved -> {args.output}")


if __name__ == "__main__":
    main()
