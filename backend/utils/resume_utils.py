from __future__ import annotations

from typing import Any


def flatten_resume_text_field(*values: Any) -> str:
    """Coalesce LLM text fields that may arrive as strings, lists, or nested dicts."""
    parts: list[str] = []

    def _collect(value: Any) -> None:
        if value is None or value == "":
            return
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                parts.append(stripped)
            return
        if isinstance(value, list):
            for item in value:
                _collect(item)
            return
        if isinstance(value, dict):
            for key in ("text", "bullet", "content", "description", "value"):
                if key in value and value[key]:
                    _collect(value[key])
                    return
            for nested in value.values():
                if isinstance(nested, (str, list, dict)):
                    _collect(nested)
            return
        stripped = str(value).strip()
        if stripped and stripped not in {"{}", "[]"}:
            parts.append(stripped)

    for value in values:
        _collect(value)
    return " ".join(parts)


def extract_experience_description(exp: dict[str, Any]) -> str:
    return flatten_resume_text_field(
        exp.get("description"),
        exp.get("bullets"),
        exp.get("responsibilities"),
        exp.get("highlights"),
        exp.get("summary"),
    )


def extract_project_description(proj: dict[str, Any]) -> str:
    return flatten_resume_text_field(
        proj.get("description"),
        proj.get("bullets"),
        proj.get("summary"),
    )


def build_experience_section_text(experience: list[Any]) -> str:
    parts: list[str] = []
    for exp in experience:
        if not isinstance(exp, dict):
            continue
        entry_parts = [
            exp.get("title", ""),
            exp.get("company", ""),
            extract_experience_description(exp),
        ]
        parts.append(" ".join(str(p).strip() for p in entry_parts if p))
    return " ".join(parts)


def build_projects_section_text(projects: list[Any]) -> str:
    parts: list[str] = []
    for proj in projects:
        if not isinstance(proj, dict):
            continue
        techs = proj.get("tech", []) or proj.get("tech_stack", []) or []
        tech_str = " ".join(str(t) for t in techs) if isinstance(techs, list) else str(techs)
        entry_parts = [
            proj.get("name", ""),
            extract_project_description(proj),
            tech_str,
        ]
        parts.append(" ".join(str(p).strip() for p in entry_parts if p))
    return " ".join(parts)


def normalize_college_tier(tier: str | None) -> str:
    """Convert parser tiers (tier1/tier2/tier3) to training-data format (Tier-1/2/3)."""
    normalized = str(tier or "tier3").strip().lower().replace("-", "").replace(" ", "")
    if normalized in {"tier1", "1"}:
        return "Tier-1"
    if normalized in {"tier2", "2"}:
        return "Tier-2"
    return "Tier-3"


def stable_candidate_id(resume_json: dict[str, Any], analysis_id: str | None = None) -> int:
    """Derive a stable numeric ID from resume identity instead of a hardcoded placeholder."""
    seed = (
        analysis_id
        or resume_json.get("email")
        or resume_json.get("phone")
        or resume_json.get("name")
        or "candidate"
    )
    return abs(hash(str(seed))) % 90000 + 10000


_BIAS_TYPE_ALIASES: dict[str, str] = {
    "college prestige bias": "prestige_gap",
    "prestige gap": "prestige_gap",
    "prestige": "prestige_gap",
    "geographic bias": "tier2_location",
    "non-metro geographic bias": "tier2_location",
    "tier2 location": "tier2_location",
    "project credibility bias": "project_credibility",
    "project credibility discount": "project_credibility",
    "cgpa hard filter": "cgpa_penalty",
    "cgpa penalty": "cgpa_penalty",
    "gender-coded language": "gender_coded_language",
    "gender coded language": "gender_coded_language",
    "name-origin bias": "name_origin",
    "name origin": "name_origin",
    "career gap": "career_gap",
    "unexplained career gap": "career_gap",
    "indian english phrasing": "vernacular_english",
    "vernacular english": "vernacular_english",
    "degree/branch bias": "degree_branch_bias",
    "degree branch bias": "degree_branch_bias",
    "masculine-coded language": "masculine_language_bias",
    "masculine language bias": "masculine_language_bias",
}


def map_bias_type(bias_type_name: str, patterns: dict[str, Any]) -> str:
    """Map LLM/tuned-model bias labels to internal pattern keys."""
    raw = str(bias_type_name or "").strip()
    if not raw:
        return "project_credibility"

    lowered = raw.lower().replace("_", " ").strip()
    if lowered in _BIAS_TYPE_ALIASES:
        return _BIAS_TYPE_ALIASES[lowered]

    for key, pattern in patterns.items():
        label = str(pattern.get("label", "")).lower()
        if label == lowered or key.replace("_", " ") == lowered:
            return key

    slug = lowered.replace(" ", "_").replace("_bias", "")
    if slug in patterns:
        return slug

    for key in patterns:
        if key.replace("_", " ") in lowered or lowered in key.replace("_", " "):
            return key

    return "project_credibility"


def normalize_llm_flag(flag: Any) -> dict[str, Any] | None:
    """Coerce LLM flag payloads into a dict shape the pipeline can consume."""
    if isinstance(flag, dict):
        return flag
    if isinstance(flag, str) and flag.strip():
        return {"bias_type": flag.strip(), "severity": "medium"}
    return None
