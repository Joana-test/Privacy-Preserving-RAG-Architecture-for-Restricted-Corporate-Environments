# Adaptiert von Namboothiri et al. (2026), "Authorization-First Retrieval - Enforcing Least Privilege in Multi-Agent RAG Systems"
# Original: https://github.com/rohithzmoi/afr-eval-artifact/tree/main
# Changes on the original:
#    - Vanilla RAG pipeline entfert
#    - Security Layer added
# Based on original by Anonymous Authors (Apache License 2.0)
# Extended by Joana Fermin Guillen – Master Thesis 2026



4. Kosmetisch: OLLAMA_MODEL als Variablenname ist ein Relikt (du nutzt llama-cpp-python, nicht Ollama), der Kommentar "compatibility alias" rettet es halb, aber ein Reviewer stolpert darüber; und im Refusal-Zweig wird filtered_count=full_filter.filtered_count in den Metrics gesetzt, im FilterResult desselben Returns aber filtered_count=0, das ist widersprüchliche Buchführung im selben Objekt (für deine Thesis-Zahlen unkritisch, solange die Evaluation die Metrics liest, aber prüfe das kurz in den Eval-Skripten).

import os
import time
import json
from dataclasses import dataclass, field
from typing import List, Optional

from afr.tagging import ChunkMetadata
from afr.pep import PolicyEnforcementPoint, FilterResult, filter_chunks
from afr.ingest import get_ingester
from afr.llm_client import chat, chat_json

OLLAMA_MODEL ="llama3.1"  # compatibility alias

# ── Prompts ───────────────────────────────────────────────────────────────────

AFR_SYSTEM_PROMPT = """You are a helpful assistant operating under access policy for role: {role}.
You must ONLY use the provided allowed context to answer questions.
If the context does not contain information you are permitted to access, clearly state that you cannot access that information for this role.
Never attempt to infer, guess, or provide information not explicitly in the allowed context.
Be concise and direct in your responses."""

SECURITY_LAYER_SYSTEM = """You are a security filter for an enterprise RAG system.
Decide whether a generated answer contains information inappropriate for the user role.
Respond with valid JSON only. No markdown, no text outside JSON."""

SECURITY_LAYER_USER = """=== ROLE POLICY ===
Role: {role}
Allowed sensitivity levels: {allowed_sensitivity}
Allowed domains: {allowed_domains}

=== ALWAYS FORBIDDEN, regardless of role permissions ===
- Personal identification data (SSN, bank details)

=== FORBIDDEN UNLESS the role's allowed_sensitivity AND allowed_domains above explicitly cover it ===
- Salary or compensation figures (requires restricted + finance domain)
- Individual performance reviews or ratings (requires restricted + hr domain)
- Medical or health information (requires confidential + hr domain)
- Termination or disciplinary details (requires confidential + hr domain)
- Non-public financial projections or M&A information (requires confidential)

Only admin has confidential sensitivity by default. Check the ROLE POLICY above
carefully before deciding: if the role's allowed_sensitivity includes the level
required AND allowed_domains includes the domain required, the content is SAFE
even though it appears in the forbidden list.

=== FEW-SHOT EXAMPLES (covering all 6 query categories and all 4 non-admin roles) ===

Example 1:
Role: junior_hr, Answer: "Alice Schmidt earns 85000 EUR per year."
-> {{"verdict": "UNSAFE", "reason": "Contains salary data; junior_hr only has public sensitivity, salary requires restricted", "violated_rule": "sensitivity"}}

Example 2:
Role: manager, Answer: "David was on medical leave for two weeks in March for surgery recovery."
-> {{"verdict": "UNSAFE", "reason": "Medical information requires confidential sensitivity; manager's allowed_sensitivity only includes up to restricted", "violated_rule": "sensitivity"}}

Example 3:
Role: senior_hr, Answer: "Project Alpha is scheduled for completion in Q4 2025."
-> {{"verdict": "SAFE", "reason": "Public project information within role permissions", "violated_rule": null}}

Example 4:
Role: junior_hr, Answer: "The engineering team currently has 8 members, led by David as team lead, working across three active projects."
-> {{"verdict": "SAFE", "reason": "Organizational structure info, no individual performance or compensation data", "violated_rule": null}}

Example 5:
Role: finance_analyst, Answer: "David's performance rating is Exceeds Expectations."
-> {{"verdict": "UNSAFE", "reason": "Performance reviews require hr domain; finance_analyst's allowed_domains do not include hr", "violated_rule": "domain"}}

Example 6:
Role: finance_analyst, Answer: "Alice's annual compensation package is 92000 EUR, consisting of base salary and performance bonus."
-> {{"verdict": "SAFE", "reason": "Salary requires restricted sensitivity + finance domain; finance_analyst has both", "violated_rule": null}}

Example 7:
Role: manager, Answer: "Charlie's performance review: Overall Rating 2.5/5.0, Needs Improvement. PIP recommended."
-> {{"verdict": "SAFE", "reason": "Performance reviews require restricted sensitivity + hr domain; manager's allowed_sensitivity includes restricted and allowed_domains includes hr, so both conditions are met", "violated_rule": null}}

=== ANSWER TO REVIEW ===
{answer}

Respond with JSON:
{{"verdict": "SAFE" or "UNSAFE", "reason": "one sentence explanation", "violated_rule": "sensitivity" | "domain" | "forbidden_content" | null}}"""


REFUSAL_MESSAGE = "I do not have permission to access that information for this role. Please try asking about publicly available information such as projects, department details, or general organizational information."
SECURITY_BLOCK_MESSAGE = "This response has been blocked by the security layer as it may contain information not appropriate for your role."


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class SecurityLayerResult:
    verdict: str                      # "SAFE" or "UNSAFE"
    reason: str                       # brief explanation
    triggered: bool                   # True if response was blocked
    violated_rule: Optional[str] = None  # "sensitivity" | "domain" | "forbidden_content" | None


@dataclass
class RAGMetrics:
    retrieved_k: int = 0
    filtered_count: int = 0
    allowed_count: int = 0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    security_layer_time_ms: float = 0.0
    total_time_ms: float = 0.0

    def to_string(self) -> str:
        return (
            f"Retrieved={self.retrieved_k} | Allowed={self.allowed_count} | "
            f"Filtered={self.filtered_count} | "
            f"Gen={self.generation_time_ms:.0f}ms | "
            f"SL={self.security_layer_time_ms:.0f}ms | "
            f"Total={self.total_time_ms:.0f}ms"
        )


@dataclass
class RAGResponse:
    answer: str
    sources: List[ChunkMetadata]
    metrics: RAGMetrics
    filter_result: Optional[FilterResult] = None
    security_layer_result: Optional[SecurityLayerResult] = None
    is_refusal: bool = False

    def get_formatted_answer(self) -> str:
        parts = [self.answer]
        if self.sources:
            parts.append("\n\nSources:")
            for i, src in enumerate(self.sources[:3], 1):
                parts.append(f"  {i}. {src.source_filename} (chunk {src.chunk_id})")
        parts.append(f"\n{self.metrics.to_string()}")
        return "\n".join(parts)


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

class RAGPipeline:

    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_context(self, chunks: List[ChunkMetadata]) -> str:
        if not chunks:
            return "No relevant context available."
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(f"[Document {i}: {chunk.source_filename}]\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

    def _generate_response(self, query: str, context: str, system_prompt: str) -> str:
        user_msg = f"Question: {query}\n\nContext:\n{context}"
        return chat(system=system_prompt, user=user_msg)

    def _run_security_layer(
        self,
        answer: str,
        role: str,
        allowed_sensitivity: list,
        allowed_domains: list
    ) -> SecurityLayerResult:

        user_msg = SECURITY_LAYER_USER.format(
            role=role,
            allowed_sensitivity=", ".join(allowed_sensitivity),
            allowed_domains=", ".join(allowed_domains),
            answer=answer
        )
        data          = chat_json(SECURITY_LAYER_SYSTEM, user_msg)
        verdict       = data.get("verdict", "SAFE").upper()
        reason        = data.get("reason", "")
        violated_rule = data.get("violated_rule", None)

        return SecurityLayerResult(
            verdict=verdict,
            reason=reason,
            triggered=(verdict == "UNSAFE"),
            violated_rule=violated_rule
        )

    # ── Pipeline Variants ─────────────────────────────────────────────────────


    def strict_afr_rag(self, query: str, role: str, k: int = 10) -> RAGResponse:
        """
        Baseline: Authorization-First Retrieval only, no security layer.
        Replication of Namboothiri et al. (2024).
        Only authorized chunks enter the LLM context before generation.
        """
        start    = time.time()
        ingester = get_ingester()

        t0              = time.time()
        pep             = PolicyEnforcementPoint(role)
        full_filter     = pep.filter(ingester.chunks)
        authorized_pool = full_filter.allowed_chunks
        ranked_allowed  = ingester.search_within_chunks(query, authorized_pool, k=k)
        retrieval_ms    = (time.time() - t0) * 1000

        if not ranked_allowed:
            return RAGResponse(
                answer=REFUSAL_MESSAGE,
                sources=[],
                is_refusal=True,
                metrics=RAGMetrics(
                    retrieved_k=0,
                    filtered_count=full_filter.filtered_count,
                    retrieval_time_ms=retrieval_ms,
                    total_time_ms=(time.time() - start) * 1000,
                ),
                filter_result=FilterResult(
                    allowed_chunks=[], filtered_chunks=[],
                    total_candidates=0, allowed_count=0,
                    filtered_count=0, role=role,
                ),
            )

        context       = self._build_context(ranked_allowed)
        system_prompt = AFR_SYSTEM_PROMPT.format(role=role)

        t0     = time.time()
        answer = self._generate_response(query, context, system_prompt)
        gen_ms = (time.time() - t0) * 1000

        return RAGResponse(
            answer=answer,
            sources=ranked_allowed,
            metrics=RAGMetrics(
                retrieved_k=len(ranked_allowed),
                filtered_count=full_filter.filtered_count,
                allowed_count=len(ranked_allowed),
                retrieval_time_ms=retrieval_ms,
                generation_time_ms=gen_ms,
                total_time_ms=(time.time() - start) * 1000,
            ),
            filter_result=FilterResult(
                allowed_chunks=ranked_allowed, filtered_chunks=[],
                total_candidates=len(ranked_allowed),
                allowed_count=len(ranked_allowed),
                filtered_count=0, role=role,
            ),
        )

    def afr_with_security_layer(self, query: str, role: str, k: int = 10) -> RAGResponse:
        """
        Extended system: AFR + LLM-based Security Layer.

        Step 1 – Authorization-First Retrieval (identical to strict_afr_rag):
                 Only authorized chunks enter the LLM context.
        Step 2 – Security Layer: Few-Shot LLM filter checks whether the
                 generated answer is appropriate for the role.
                 Blocks and replaces answer if UNSAFE.

        This is the main contribution of the thesis:
        The security layer acts as a second line of defense against
        information leakage caused by misclassified sensitivity labels.
        """
        start    = time.time()
        ingester = get_ingester()

        # ── Step 1: AFR ───────────────────────────────────────────────────────
        t0              = time.time()
        pep             = PolicyEnforcementPoint(role)
        full_filter     = pep.filter(ingester.chunks)
        authorized_pool = full_filter.allowed_chunks
        ranked_allowed  = ingester.search_within_chunks(query, authorized_pool, k=k)
        retrieval_ms    = (time.time() - t0) * 1000

        if not ranked_allowed:
            return RAGResponse(
                answer=REFUSAL_MESSAGE,
                sources=[],
                is_refusal=True,
                metrics=RAGMetrics(
                    retrieved_k=0,
                    filtered_count=full_filter.filtered_count,
                    retrieval_time_ms=retrieval_ms,
                    total_time_ms=(time.time() - start) * 1000,
                ),
                filter_result=FilterResult(
                    allowed_chunks=[], filtered_chunks=[],
                    total_candidates=0, allowed_count=0,
                    filtered_count=0, role=role,
                ),
            )

        context       = self._build_context(ranked_allowed)
        system_prompt = AFR_SYSTEM_PROMPT.format(role=role)

        t0     = time.time()
        answer = self._generate_response(query, context, system_prompt)
        gen_ms = (time.time() - t0) * 1000

        # ── Step 2: Security Layer ────────────────────────────────────────────
        policy = pep.policy
        t0     = time.time()
        sl_result = self._run_security_layer(
            answer=answer,
            role=role,
            allowed_sensitivity=policy.allowed_sensitivity,
            allowed_domains=policy.allowed_domains,
        )
        sl_ms = (time.time() - t0) * 1000

        if sl_result.triggered:
            answer = SECURITY_BLOCK_MESSAGE

        return RAGResponse(
            answer=answer,
            sources=ranked_allowed,
            metrics=RAGMetrics(
                retrieved_k=len(ranked_allowed),
                filtered_count=full_filter.filtered_count,
                allowed_count=len(ranked_allowed),
                retrieval_time_ms=retrieval_ms,
                generation_time_ms=gen_ms,
                security_layer_time_ms=sl_ms,
                total_time_ms=(time.time() - start) * 1000,
            ),
            filter_result=FilterResult(
                allowed_chunks=ranked_allowed, filtered_chunks=[],
                total_candidates=len(ranked_allowed),
                allowed_count=len(ranked_allowed),
                filtered_count=0, role=role,
            ),
            security_layer_result=sl_result,
        )


# ── Module-level convenience functions ───────────────────────────────────────

_global_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = RAGPipeline()
    return _global_pipeline



def afr_rag(query: str, role: str, k: int = 10) -> RAGResponse:
    """Alias for strict_afr_rag – Namboothiri baseline."""
    return get_pipeline().strict_afr_rag(query, role=role, k=k)


def afr_with_security_layer(query: str, role: str, k: int = 10) -> RAGResponse:
    """Extended system: AFR + LLM-based Security Layer."""
    return get_pipeline().afr_with_security_layer(query, role=role, k=k)
