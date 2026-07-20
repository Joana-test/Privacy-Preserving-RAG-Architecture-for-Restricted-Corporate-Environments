"""
Policy Enforcement Point: applies the role policies from policies.py to each
retrieved chunk and splits them into an authorized and a filtered set. Also
provides filter statistics for the retrieval inspector.

Taken from Namboothiri et al. (2026),
licensed under the Apache License, Version 2.0.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from afr.policies import validate_access, PolicyDecision, get_policy
from afr.tagging import ChunkMetadata


@dataclass
class FilteredChunk:
    chunk: ChunkMetadata
    decision: PolicyDecision

    @property
    def reason(self) -> str:
        return self.decision.reason


@dataclass
class FilterResult:
    allowed_chunks: List[ChunkMetadata]
    filtered_chunks: List[FilteredChunk]
    total_candidates: int
    allowed_count: int
    filtered_count: int
    role: str

    @property
    def filter_rate(self) -> float:
        if self.total_candidates == 0:
            return 0.0
        return self.filtered_count / self.total_candidates

    def get_filter_summary(self) -> Dict[str, int]:
        reasons = {}
        for fc in self.filtered_chunks:
            reason_key = fc.reason.split(";")[0].strip() if fc.reason else "Unknown"
            reasons[reason_key] = reasons.get(reason_key, 0) + 1
        return reasons

    def to_inspector_data(self) -> List[Dict]:
        data = []

        for chunk in self.allowed_chunks:
            data.append({
                "score": f"{chunk.score:.3f}",
                "source": chunk.source_filename,
                "sensitivity": chunk.sensitivity,
                "domain": chunk.domain,
                "preview": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
                "status": "[Allowed]",
                "reason": "Access granted",
            })

        for fc in self.filtered_chunks:
            data.append({
                "score": f"{fc.chunk.score:.3f}",
                "source": fc.chunk.source_filename,
                "sensitivity": fc.chunk.sensitivity,
                "domain": fc.chunk.domain,
                "preview": fc.chunk.text[:100] + "..." if len(fc.chunk.text) > 100 else fc.chunk.text,
                "status": "[Filtered]",
                "reason": fc.reason,
            })

        return sorted(data, key=lambda x: float(x["score"]), reverse=True)


class PolicyEnforcementPoint:

    def __init__(self, role: str):
        self.role = role
        self.policy = get_policy(role)

    def check_access(self, chunk: ChunkMetadata) -> PolicyDecision:
        return validate_access(
            role=self.role,
            sensitivity=chunk.sensitivity,
            domain=chunk.domain,
            subject=chunk.subject
        )

    def filter(self, chunks: List[ChunkMetadata]) -> FilterResult:
        allowed = []
        filtered = []

        for chunk in chunks:
            decision = self.check_access(chunk)
            if decision.allowed:
                allowed.append(chunk)
            else:
                filtered.append(FilteredChunk(chunk=chunk, decision=decision))

        return FilterResult(
            allowed_chunks=allowed,
            filtered_chunks=filtered,
            total_candidates=len(chunks),
            allowed_count=len(allowed),
            filtered_count=len(filtered),
            role=self.role
        )


def filter_chunks(chunks: List[ChunkMetadata], role: str) -> FilterResult:
    pep = PolicyEnforcementPoint(role)
    return pep.filter(chunks)


def create_pep(role: str) -> PolicyEnforcementPoint:
    return PolicyEnforcementPoint(role)


def get_filter_explanation(result: FilterResult) -> str:
    if result.filtered_count == 0:
        return f"All {result.total_candidates} chunks are accessible for role '{result.role}'."

    explanation = [
        f"Role: {result.role}",
        f"Total candidates: {result.total_candidates}",
        f"Allowed: {result.allowed_count}",
        f"Filtered: {result.filtered_count}",
        "",
        "Filter reasons:"
    ]

    for reason, count in result.get_filter_summary().items():
        explanation.append(f"  - {reason}: {count} chunk(s)")

    return "\n".join(explanation)
