from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.core.logging_config import log_event
from backend.utils.llm_client import call_llm_json
from backend.utils.pdf_utils import extract_text_from_pdf
from backend.utils.prompts import PARSER_PROMPT
from backend.utils.resume_utils import extract_experience_description, extract_project_description

DEFAULT_RESUME_JSON: dict[str, Any] = {
    "name": "",
    "email": "",
    "phone": "",
    "college": "",
    "tier": "tier3",
    "cgpa": "",
    "branch": "",
    "graduation_year": "",
    "skills": [],
    "experience": [],
    "projects": [],
    "certifications": [],
    "languages_known": [],
    "gender_indicators": [],
    "name_origin_hints": "",
    "career_gaps": [],
    "location": "",
}


def _ensure_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, "", {}):
        return []
    return [value]


def _ensure_resume_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(DEFAULT_RESUME_JSON)
    normalized.update({key: payload.get(key, default) for key, default in DEFAULT_RESUME_JSON.items()})

    # Clean and flatten skills list
    raw_skills = _ensure_list(normalized.get("skills"))
    cleaned_skills = []
    for s in raw_skills:
        if not isinstance(s, str):
            continue
        s = s.strip()
        if not s:
            continue
        if s.endswith(":"):
            continue
        if "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            cleaned_skills.extend(parts)
        else:
            cleaned_skills.append(s)
    normalized["skills"] = cleaned_skills

    # Clean and structure experience
    raw_exp = _ensure_list(normalized.get("experience"))
    cleaned_exp = []
    for exp in raw_exp:
        if not isinstance(exp, dict):
            continue
        desc = extract_experience_description(exp)
        
        cleaned_exp.append({
            "title": str(exp.get("title") or ""),
            "company": str(exp.get("company") or ""),
            "duration": str(exp.get("duration") or ""),
            "description": desc.strip(),
            "is_internship": bool(exp.get("is_internship", False))
        })
    normalized["experience"] = cleaned_exp

    # Clean and structure projects
    raw_proj = _ensure_list(normalized.get("projects"))
    cleaned_proj = []
    for proj in raw_proj:
        if not isinstance(proj, dict):
            continue
        desc = extract_project_description(proj)
            
        raw_tech = proj.get("tech") or proj.get("tech_stack") or []
        tech_list = []
        if isinstance(raw_tech, list):
            for t in raw_tech:
                if isinstance(t, str):
                    tech_list.append(t.strip())
        elif isinstance(raw_tech, str):
            tech_list = [t.strip() for t in raw_tech.split(",") if t.strip()]
            
        cleaned_proj.append({
            "name": str(proj.get("name") or ""),
            "description": desc.strip(),
            "tech": tech_list,
            "has_metrics": bool(proj.get("has_metrics", False))
        })
    normalized["projects"] = cleaned_proj

    normalized["certifications"] = _ensure_list(normalized.get("certifications"))
    normalized["languages_known"] = _ensure_list(normalized.get("languages_known"))
    normalized["gender_indicators"] = _ensure_list(normalized.get("gender_indicators"))
    normalized["career_gaps"] = _ensure_list(normalized.get("career_gaps"))

    tier = str(normalized.get("tier") or "tier3").lower()
    normalized["tier"] = tier if tier in {"tier1", "tier2", "tier3"} else "tier3"
    normalized["cgpa"] = str(normalized.get("cgpa") or "")
    return normalized


def _extract_resume_text(file_path: str | Path) -> str:
    return extract_text_from_pdf(file_path)


def parse_resume_pdf(file_path: str | Path, user_id: str | None = None, analysis_id: str | None = None) -> dict[str, Any]:
    path = Path(file_path)
    log_event(
        agent="A2_PARSER",
        user_id=user_id,
        analysis_id=analysis_id,
        event="agent_start",
        details={"file_path": str(path), "file_name": path.name},
    )

    resume_text = _extract_resume_text(path)
    parsed: dict[str, Any]

    try:
        parsed = call_llm_json(
            prompt=f"{PARSER_PROMPT}\n\nResume text:\n{resume_text}",
            system_instruction="You are a strict JSON-only resume parser.",
            schema_hint=json.dumps(DEFAULT_RESUME_JSON),
            agent="A2_PARSER",
            user_id=user_id,
            analysis_id=analysis_id,
        )
    except Exception:
        parsed = {}

    normalized = _ensure_resume_defaults(parsed)
    exp_descriptions = [
        len(extract_experience_description(exp))
        for exp in _ensure_list(normalized.get("experience"))
        if isinstance(exp, dict)
    ]
    log_event(
        agent="A2_PARSER",
        user_id=user_id,
        analysis_id=analysis_id,
        event="agent_complete",
        details={
            "resume_text_chars": len(resume_text),
            "fields_populated": sum(1 for value in normalized.values() if value not in ("", [], {})),
            "experience_entry_count": len(exp_descriptions),
            "experience_description_chars": exp_descriptions,
        },
    )
    return {
        "resume_text": resume_text,
        "resume_json": normalized,
    }
