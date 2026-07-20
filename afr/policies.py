"""
Role definitions and access policies. Each role carries permitted sensitivity
levels, domains, and subjects; validate_access() evaluates a chunk against
them and returns a decision with a reason.

Five roles are defined: junior_hr, senior_hr, finance_analyst, manager, admin.

Taken from Namboothiri et al. (2026),
licensed under the Apache License, Version 2.0.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
from enum import Enum


class Sensitivity(Enum):
    PUBLIC = "public"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"


class Domain(Enum):
    HR = "hr"
    FINANCE = "finance"
    ENGINEERING = "engineering"
    GENERAL = "general"


@dataclass
class RolePolicy:
    role_name: str
    allowed_sensitivity: List[str]
    allowed_domains: List[str]
    allowed_subjects: Optional[List[str]] = None
    description: str = ""

    def can_access(self, sensitivity: str, domain: str, subject: Optional[str] = None) -> bool:
        if sensitivity not in self.allowed_sensitivity:
            return False
        if domain not in self.allowed_domains and "all" not in self.allowed_domains:
            return False
        if self.allowed_subjects is not None and subject is not None:
            if subject not in self.allowed_subjects and "all" not in self.allowed_subjects:
                return False
        return True


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str
    matched_rule: Optional[str] = None
    sensitivity_match: bool = True
    domain_match: bool = True
    subject_match: bool = True


ROLE_POLICIES: Dict[str, RolePolicy] = {
    "junior_hr": RolePolicy(
        role_name="junior_hr",
        allowed_sensitivity=["public"],
        allowed_domains=["hr", "general"],
        allowed_subjects=["all"],
        description="Junior HR staff - access to public HR information only"
    ),
    "senior_hr": RolePolicy(
        role_name="senior_hr",
        allowed_sensitivity=["public", "restricted"],
        allowed_domains=["hr", "general"],
        allowed_subjects=["all"],
        description="Senior HR staff - access to public and restricted HR information"
    ),
    "finance_analyst": RolePolicy(
        role_name="finance_analyst",
        allowed_sensitivity=["public", "restricted"],
        allowed_domains=["finance", "general"],
        allowed_subjects=["all"],
        description="Finance Analyst - access to finance data, no HR restricted access"
    ),
    "manager": RolePolicy(
        role_name="manager",
        allowed_sensitivity=["public", "restricted"],
        allowed_domains=["hr", "engineering", "general"],
        allowed_subjects=["all"],
        description="Manager - access to HR and engineering, no finance restricted"
    ),
    "admin": RolePolicy(
        role_name="admin",
        allowed_sensitivity=["public", "restricted", "confidential"],
        allowed_domains=["all"],
        allowed_subjects=["all"],
        description="Administrator - full access to all information"
    ),
}

ROLE_DISPLAY_NAMES: Dict[str, str] = {
    "junior_hr": "Junior HR",
    "senior_hr": "Senior HR",
    "finance_analyst": "Finance Analyst",
    "manager": "Manager",
    "admin": "Admin",
}


def get_policy(role_name: str) -> RolePolicy:
    role_key = role_name.lower().replace(" ", "_")
    if role_key not in ROLE_POLICIES:
        return ROLE_POLICIES["junior_hr"]
    return ROLE_POLICIES[role_key]


def list_roles() -> List[str]:
    return list(ROLE_DISPLAY_NAMES.values())


def get_role_key(display_name: str) -> str:
    for key, name in ROLE_DISPLAY_NAMES.items():
        if name == display_name:
            return key
    return "junior_hr"


def validate_access(
        role: str,
        sensitivity: str,
        domain: str,
        subject: Optional[str] = None
) -> PolicyDecision:
    policy = get_policy(role)

    sensitivity_ok = sensitivity in policy.allowed_sensitivity
    domain_ok = domain in policy.allowed_domains or "all" in policy.allowed_domains
    subject_ok = True
    if policy.allowed_subjects is not None and subject is not None:
        subject_ok = subject in policy.allowed_subjects or "all" in policy.allowed_subjects

    allowed = sensitivity_ok and domain_ok and subject_ok

    reasons = []
    if not sensitivity_ok:
        reasons.append(f"sensitivity '{sensitivity}' not allowed for role '{role}'")
    if not domain_ok:
        reasons.append(f"domain '{domain}' not allowed for role '{role}'")
    if not subject_ok:
        reasons.append(f"subject '{subject}' not allowed for role '{role}'")

    reason = "; ".join(reasons) if reasons else "Access granted"
    matched_rule = f"Role: {role}, Sensitivity: {policy.allowed_sensitivity}, Domains: {policy.allowed_domains}"

    return PolicyDecision(
        allowed=allowed,
        reason=reason,
        matched_rule=matched_rule,
        sensitivity_match=sensitivity_ok,
        domain_match=domain_ok,
        subject_match=subject_ok
    )


def get_policy_summary(role: str) -> str:
    policy = get_policy(role)
    return (
        f"Role: {policy.role_name}\n"
        f"Description: {policy.description}\n"
        f"Allowed Sensitivity: {', '.join(policy.allowed_sensitivity)}\n"
        f"Allowed Domains: {', '.join(policy.allowed_domains)}"
    )
