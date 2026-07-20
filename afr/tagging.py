# Adaptiert von Namboothiri et al. (2026), "Authorization-First Retrieval - Enforcing Least Privilege in Multi-Agent RAG Systems"
# Original: https://github.com/rohithzmoi/afr-eval-artifact/tree/main
# Anonymous Authors
# Licensed under the Apache License, Version 2.0
# See LICENSE file for details

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import re

SENSITIVITY_KEYWORDS = {
    "confidential": [
        "confidential", "secret", "top secret", "classified",
        "private key", "password", "credential", "ssn", "social security"
    ],
    "restricted": [
        "salary", "compensation", "payroll", "bonus", "stock option",
        "appraisal", "performance review", "disciplinary", "termination",
        "medical", "health record", "disability", "personal leave",
        "budget confidential", "financial projection", "merger", "acquisition"
    ],
}

DOMAIN_KEYWORDS = {
    "finance": [
        "salary", "payroll", "compensation", "budget", "revenue",
        "expense", "profit", "loss", "financial", "accounting",
        "invoice", "payment", "tax", "audit", "fiscal"
    ],
    "hr": [
        "employee", "staff", "personnel", "hire", "onboarding",
        "performance", "review", "promotion", "department", "team",
        "leave", "vacation", "benefits", "training", "hr"
    ],
    "engineering": [
        "code", "software", "development", "architecture", "api",
        "database", "server", "deployment", "git", "repository",
        "sprint", "agile", "technical", "engineering", "devops"
    ],
}

SUBJECT_PATTERNS = [
    r"\b(bob|alice|charlie|david|eve|frank|grace|henry|ivan|julia)\b",
    r"\bemployee[:\s]+(\w+)\b",
    r"\bsubject[:\s]+(\w+)\b",
    r"\bname[:\s]+(\w+)\b",
]


@dataclass
class ChunkMetadata:
    doc_id: str
    source_filename: str
    chunk_id: str
    text: str
    sensitivity: str = "public"
    domain: str = "general"
    subject: Optional[str] = None
    embedding: Optional[List[float]] = None
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_filename": self.source_filename,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "sensitivity": self.sensitivity,
            "domain": self.domain,
            "subject": self.subject,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkMetadata":
        return cls(
            doc_id=data.get("doc_id", ""),
            source_filename=data.get("source_filename", ""),
            chunk_id=data.get("chunk_id", ""),
            text=data.get("text", ""),
            sensitivity=data.get("sensitivity", "public"),
            domain=data.get("domain", "general"),
            subject=data.get("subject"),
            score=data.get("score", 0.0),
        )


def detect_sensitivity(text: str, filename: str) -> str:
    text_lower = text.lower()
    filename_lower = filename.lower()

    if "finance" in filename_lower or "payroll" in filename_lower:
        return "restricted"
    if "confidential" in filename_lower or "secret" in filename_lower:
        return "confidential"

    for keyword in SENSITIVITY_KEYWORDS["confidential"]:
        if keyword in text_lower:
            return "confidential"

    for keyword in SENSITIVITY_KEYWORDS["restricted"]:
        if keyword in text_lower:
            return "restricted"

    return "public"


def detect_domain(text: str, filename: str) -> str:
    text_lower = text.lower()
    filename_lower = filename.lower()

    domain_scores = {"finance": 0, "hr": 0, "engineering": 0}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        if domain in filename_lower:
            domain_scores[domain] += 5
        for keyword in keywords:
            if keyword in text_lower:
                domain_scores[domain] += 1

    max_domain = max(domain_scores, key=domain_scores.get)
    if domain_scores[max_domain] > 0:
        return max_domain

    return "general"


def detect_subject(text: str) -> Optional[str]:
    text_lower = text.lower()

    for pattern in SUBJECT_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            subject = match.group(1) if match.lastindex else match.group(0)
            return subject.lower()

    return None


def tag_chunk(
        text: str,
        filename: str,
        doc_id: str,
        chunk_id: str,
        override_sensitivity: Optional[str] = None,
        override_domain: Optional[str] = None,
) -> ChunkMetadata:
    sensitivity = override_sensitivity or detect_sensitivity(text, filename)
    domain = override_domain or detect_domain(text, filename)
    subject = detect_subject(text)

    return ChunkMetadata(
        doc_id=doc_id,
        source_filename=filename,
        chunk_id=chunk_id,
        text=text,
        sensitivity=sensitivity,
        domain=domain,
        subject=subject,
    )


def get_tagging_summary(chunks: List[ChunkMetadata]) -> Dict[str, Any]:
    summary = {
        "total_chunks": len(chunks),
        "by_sensitivity": {"public": 0, "restricted": 0, "confidential": 0},
        "by_domain": {"hr": 0, "finance": 0, "engineering": 0, "general": 0},
        "subjects_found": set(),
    }

    for chunk in chunks:
        summary["by_sensitivity"][chunk.sensitivity] = summary["by_sensitivity"].get(chunk.sensitivity, 0) + 1
        summary["by_domain"][chunk.domain] = summary["by_domain"].get(chunk.domain, 0) + 1
        if chunk.subject:
            summary["subjects_found"].add(chunk.subject)

    summary["subjects_found"] = list(summary["subjects_found"])
    return summary
