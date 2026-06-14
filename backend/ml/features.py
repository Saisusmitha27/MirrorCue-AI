from __future__ import annotations

import re
from typing import Any

from backend.utils.resume_utils import normalize_college_tier

BIAS_LABEL_KEYS = [
    "prestige_gap",
    "degree_branch_bias",
    "cgpa_penalty",
    "career_gap",
    "tier2_location",
    "name_origin",
    "project_credibility",
    "gender_coded_language",
]

BIAS_TYPE_TO_KEY: dict[str, str] = {
    "college prestige bias": "prestige_gap",
    "branch bias": "degree_branch_bias",
    "cgpa bias": "cgpa_penalty",
    "career gap bias": "career_gap",
    "geographic bias": "tier2_location",
    "name-origin bias": "name_origin",
    "certification bias": "project_credibility",
    "project credibility bias": "project_credibility",
    "career transition bias": "career_gap",
    "gender-coded language bias": "gender_coded_language",
}

FEATURE_COLUMNS = [
    "tier_1",
    "tier_2",
    "tier_3",
    "cgpa_norm",
    "cgpa_below_7",
    "cgpa_below_7_5",
    "career_gap_months",
    "has_career_gap",
    "years_experience",
    "is_female",
    "is_non_metro",
    "is_non_cse_branch",
    "certification_count",
    "project_count",
    "experience_count",
    "skill_count",
    "has_vague_projects",
    "name_origin_signal",
    "gender_indicator_count",
    "screening_score_norm",
    "skill_fit_norm",
]

_NON_METRO_MARKERS = {
    "salem", "madurai", "coimbatore", "trichy", "mysore", "hubli", "nagpur", "indore",
    "vizag", "kochi", "trivandrum", "vellore", "erode", "guntur", "tirupati",
}

_NON_CSE_BRANCHES = {
    "mechanical", "biotechnology", "civil", "ece", "eee", "chemical", "aerospace",
    "instrumentation", "business administration", "metallurgy", "production",
}

_NAME_ORIGIN_MARKERS = (
    "mukhopadhyay", "narayanan", "namboothiri", "vemulapalli", "sethuraman", "gowda",
    "patil", "swamy", "menon", "iyer", "nair", "joseph", "philip", "kurian", "priya",
    "aishwarya", "manjunath", "basavaraj", "hariharan", "gireesh", "rajesh", "ramanujan",
    "chidambaram", "venkatesan", "sundaram", "naidu", "kondapalli", "hegde", "pillai",
)


def _parse_cgpa(value: Any) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def _cgpa_on_10_scale(value: Any) -> float:
    """Normalize CGPA whether stored as 0-10 or 0-100 percentage."""
    raw = _parse_cgpa(value)
    if raw > 10.0:
        return raw / 10.0
    return raw


def _is_female(gender: Any) -> bool:
    text = str(gender or "").strip().lower()
    return text in {"female", "f", "woman", "women"}


def _is_non_metro(location: str) -> bool:
    lowered = location.lower()
    return any(city in lowered for city in _NON_METRO_MARKERS)


def _is_non_cse_branch(branch: str, job_role: str = "") -> bool:
    lowered = branch.lower()
    if not lowered:
        return False
    if any(token in lowered for token in _NON_CSE_BRANCHES):
        if "software" in job_role.lower() or "engineer" in job_role.lower():
            return True
        return True
    return not any(token in lowered for token in ("computer", "cse", "it", "information", "software"))


def _name_origin_signal(name: str) -> float:
    lowered = name.lower()
    return 1.0 if any(marker in lowered for marker in _NAME_ORIGIN_MARKERS) else 0.0


def _project_is_vague(project: dict[str, Any]) -> bool:
    description = str(project.get("description", "") or project.get("vague_achievement", "")).lower()
    if not description:
        return True
    if project.get("has_metrics"):
        return False
    if re.search(r"\d", description):
        return False
    credible = ("metric", "quantified", "users", "reduced", "improved", "optimized", "deployed")
    return not any(marker in description for marker in credible)


def extract_features(
    profile: dict[str, Any],
    *,
    resume_text: str = "",
    screening_score: float | None = None,
    skill_fit_score: float | None = None,
) -> dict[str, float]:
    """Build a fixed feature dict from a candidate profile (runtime or training)."""
    tier = normalize_college_tier(profile.get("college_tier") or profile.get("tier"))
    tier_1 = 1.0 if tier == "Tier-1" else 0.0
    tier_2 = 1.0 if tier == "Tier-2" else 0.0
    tier_3 = 1.0 if tier == "Tier-3" else 0.0

    cgpa = _cgpa_on_10_scale(profile.get("cgpa") or profile.get("collegeGPA") or profile.get("collegegpa"))
    gap_months = float(profile.get("career_gap_months") or 0)
    years_exp = float(profile.get("years_experience") or profile.get("experience_years") or len(profile.get("experience", [])) or 0)

    skills = profile.get("skills", [])
    if not isinstance(skills, list):
        skills = []
    certifications = profile.get("certifications", [])
    if not isinstance(certifications, list):
        certifications = []
    projects = profile.get("projects", [])
    if not isinstance(projects, list):
        projects = []
    experience = profile.get("experience", [])
    if not isinstance(experience, list):
        experience = []

    gender_indicators = profile.get("gender_indicators", [])
    if not isinstance(gender_indicators, list):
        gender_indicators = []

    location = str(profile.get("location", "") or profile.get("collegecity", ""))
    branch = str(profile.get("branch", "") or profile.get("specialization", "") or profile.get("degree", ""))
    job_role = str(profile.get("job_role", "") or profile.get("job_category", ""))
    name = str(profile.get("name", ""))

    vague_projects = 0.0
    for project in projects:
        if isinstance(project, dict) and _project_is_vague(project):
            vague_projects = 1.0
            break

    if resume_text and not vague_projects:
        if re.search(r"did the needful|peoples|kindly do", resume_text.lower()):
            pass

    screening_norm = 0.0
    if screening_score is not None:
        screening_norm = min(1.0, max(0.0, float(screening_score) / 100.0))

    skill_norm = 0.0
    if skill_fit_score is not None:
        skill_norm = min(1.0, max(0.0, float(skill_fit_score) / 100.0))

    return {
        "tier_1": tier_1,
        "tier_2": tier_2,
        "tier_3": tier_3,
        "cgpa_norm": min(1.0, cgpa / 10.0),
        "cgpa_below_7": 1.0 if 0 < cgpa < 7.0 else 0.0,
        "cgpa_below_7_5": 1.0 if 0 < cgpa < 7.5 else 0.0,
        "career_gap_months": min(1.0, gap_months / 24.0),
        "has_career_gap": 1.0 if gap_months > 0 else 0.0,
        "years_experience": min(1.0, years_exp / 20.0),
        "is_female": 1.0 if _is_female(profile.get("gender")) else 0.0,
        "is_non_metro": 1.0 if _is_non_metro(location) else 0.0,
        "is_non_cse_branch": 1.0 if _is_non_cse_branch(branch, job_role) else 0.0,
        "certification_count": min(1.0, len(certifications) / 5.0),
        "project_count": min(1.0, len(projects) / 5.0),
        "experience_count": min(1.0, len(experience) / 5.0),
        "skill_count": min(1.0, len(skills) / 10.0),
        "has_vague_projects": vague_projects,
        "name_origin_signal": _name_origin_signal(name),
        "gender_indicator_count": min(1.0, len(gender_indicators) / 5.0),
        "screening_score_norm": screening_norm,
        "skill_fit_norm": skill_norm,
    }


def features_to_vector(features: dict[str, float]) -> list[float]:
    return [float(features.get(col, 0.0)) for col in FEATURE_COLUMNS]


def labels_to_vector(flags: list[dict[str, Any]]) -> list[int]:
    active = set()
    for flag in flags:
        bias_type = str(flag.get("bias_type", "")).lower().strip()
        key = BIAS_TYPE_TO_KEY.get(bias_type)
        if key:
            active.add(key)
    return [1 if key in active else 0 for key in BIAS_LABEL_KEYS]
