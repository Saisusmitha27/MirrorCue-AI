from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from backend.core.config import settings
from backend.core.logging_config import log_event
from backend.ml.bias_classifier import get_bias_classifier
from backend.utils.llm_client import call_llm, call_llm_json
from backend.utils.prompts import BIAS_MIRROR_PROMPT, SKILL_ALIGNMENT_PROMPT
from backend.utils.resume_utils import normalize_college_tier, stable_candidate_id

BIAS_PATTERNS_PATH = Path(__file__).parent.parent / "data" / "bias_patterns.json"


def _load_bias_patterns() -> dict[str, Any]:
    if not BIAS_PATTERNS_PATH.exists():
        return {}
    return json.loads(BIAS_PATTERNS_PATH.read_text(encoding="utf-8"))


def _candidate_wrote_for_pattern(pattern_key: str, resume_json: dict[str, Any], resume_text: str) -> tuple[str, str]:
    college = str(resume_json.get("college", "")).strip()
    cgpa = str(resume_json.get("cgpa", "")).strip()
    location = str(resume_json.get("location", "")).strip()
    name = str(resume_json.get("name", "")).strip()
    career_gaps = resume_json.get("career_gaps", [])
    experience = resume_json.get("experience", [])
    projects = resume_json.get("projects", [])
    gender_indicators = resume_json.get("gender_indicators", [])
    name_origin_hints = str(resume_json.get("name_origin_hints", "")).strip()

    if pattern_key == "prestige_gap" and college:
        return college, f"{college} may be read as a non-premium institute by an overworked recruiter"
    if pattern_key == "name_origin" and name:
        return name, f"{name} may trigger a regional, caste, or religious assumption"
    if pattern_key == "gender_coded_language" and gender_indicators:
        return ", ".join(map(str, gender_indicators)), "Soft-skills-heavy wording can trigger gender inference"
    if pattern_key == "career_gap" and career_gaps:
        return ", ".join(map(str, career_gaps)), "An unexplained gap can be read as a risk signal"
    if pattern_key == "cgpa_penalty" and cgpa:
        return f"CGPA {cgpa}", "Sub-7 or borderline CGPA often becomes a hard filter"
    if pattern_key == "vernacular_english":
        return resume_text[:240], "Indian English phrasing can reduce perceived polish"
    if pattern_key == "tier2_location" and location:
        return location, f"{location} may be mentally bucketed as non-metro"
    if pattern_key == "project_credibility" and projects:
        project_name = str(projects[0].get("name", "")) if isinstance(projects[0], dict) else str(projects[0])
        return project_name or "project details", "A project without company context or metrics can look lightweight"
    if pattern_key == "project_credibility" and experience:
        exp_title = str(experience[0].get("title", "")) if isinstance(experience[0], dict) else str(experience[0])
        return exp_title or "experience details", "A bullet without metrics can look underpowered"
    if name_origin_hints:
        return name_origin_hints, "Name-origin hints can trigger unconscious identity assumptions"
    return "resume content", "This detail may be read in a biased 7-second scan"


_NON_METRO_CITIES = {
    "salem", "madurai", "coimbatore", "trichy", "tiruchirappalli", "mysore", "mysuru",
    "nagpur", "indore", "bhopal", "vizag", "visakhapatnam", "kochi", "trivandrum",
    "thiruvananthapuram", "hubli", "belgaum", "vellore", "tirunelveli", "erode",
}


def _parse_cgpa(value: str) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _heuristic_pattern_triggered(
    pattern_key: str,
    resume_json: dict[str, Any],
    resume_text: str,
    candidate_wrote: str,
) -> bool:
    if not candidate_wrote or candidate_wrote == "resume content":
        return False

    if pattern_key == "prestige_gap":
        tier = normalize_college_tier(resume_json.get("tier"))
        return tier in {"Tier-2", "Tier-3"} and bool(resume_json.get("college"))

    if pattern_key == "cgpa_penalty":
        cgpa = _parse_cgpa(str(resume_json.get("cgpa", "")))
        return cgpa is not None and cgpa < 7.0

    if pattern_key == "career_gap":
        return bool(resume_json.get("career_gaps")) or int(resume_json.get("career_gap_months") or 0) > 0

    if pattern_key == "gender_coded_language":
        return bool(resume_json.get("gender_indicators"))

    if pattern_key == "tier2_location":
        location = str(resume_json.get("location", "")).strip().lower()
        return any(city in location for city in _NON_METRO_CITIES)

    if pattern_key == "vernacular_english":
        markers = ("did the needful", "peoples", "kindly do", "worked in a team of peoples")
        lowered = resume_text.lower()
        return any(marker in lowered for marker in markers)

    if pattern_key == "project_credibility":
        credible_markers = (
            "metric", "quantified", "measured", "kpi", "impact", "users",
            "reduced", "improved", "increased", "optimized", "scaled", "deployed",
        )
        projects = resume_json.get("projects", [])
        experience = resume_json.get("experience", [])
        for item in projects + experience:
            if not isinstance(item, dict):
                continue
            if item.get("has_metrics"):
                continue
            description = str(item.get("description", "")).strip().lower()
            if not description:
                continue
            if re.search(r"\d", description):
                continue
            if any(marker in description for marker in credible_markers):
                continue
            return True
        return False

    if pattern_key == "name_origin":
        # Check name origin hints if present
        return bool(resume_json.get("name_origin_hints"))

    return True


def _severity_from_weight(weight: float, triggered: bool) -> str:
    if not triggered:
        return "low"
    if weight >= 0.85:
        return "high"
    if weight >= 0.7:
        return "medium"
    return "low"


def _estimate_bias_score(flags: list[dict[str, Any]], pattern_weights: dict[str, float]) -> float:
    contributions = []
    for flag in flags:
        multiplier = 5
        if flag.get("severity") == "high":
            multiplier = 20
        elif flag.get("severity") == "medium":
            multiplier = 10
        weight = float(pattern_weights.get(flag.get("bias_type", ""), 0.5))
        contributions.append(weight * multiplier)
    
    # Calculate score decay for multiple flags
    contributions.sort(reverse=True)
    total = 0.0
    for idx, contrib in enumerate(contributions):
        decay = 0.85 ** idx
        total += contrib * decay
    return min(100.0, round(total, 2))


def _infer_gender(resume_json: dict[str, Any], resume_text: str) -> str:
    gender_ind = resume_json.get("gender_indicators", [])
    if gender_ind:
        text = " ".join(map(str, gender_ind)).lower()
        if any(w in text for w in ["female", "she", "her", "women", "girl"]):
            return "Female"
        if any(w in text for w in ["male", "he", "him", "his"]):
            return "Male"
    text_lower = resume_text.lower()
    if any(w in text_lower for w in ["she/her", "member of women in tech", "society of women engineers"]):
        return "Female"
    if any(w in text_lower for w in ["he/him"]):
        return "Male"
    return "Unspecified"


def _infer_job_role(resume_json: dict[str, Any], jd_text: str | None) -> str:
    if jd_text:
        jd_lower = jd_text.lower()
        if "marketing" in jd_lower:
            return "Marketing Manager"
        if "hr" in jd_lower or "human resources" in jd_lower or "talent acquisition" in jd_lower:
            return "HR Specialist"
        if "data" in jd_lower or "analyst" in jd_lower:
            return "Data Analyst"
    return "Software Engineer"


def detect_bias(
    resume_json: dict[str, Any],
    resume_text: str,
    user_id: str | None = None,
    analysis_id: str | None = None,
    jd_text: str | None = None,
    ats_score: float | None = None,
    other_analyses: list[Any] | None = None,
) -> dict[str, Any]:
    log_event(
        agent="A4_BIAS",
        user_id=user_id,
        analysis_id=analysis_id,
        event="agent_start",
        details={
            "resume_chars": len(resume_text),
            "college": str(resume_json.get("college", ""))[:80],
            "cgpa_present": bool(str(resume_json.get("cgpa", "")).strip()),
        },
    )

    patterns = _load_bias_patterns()
    pattern_weights = {key: float(value.get("severity_weight", 0.5)) for key, value in patterns.items()}
    candidate_flags: list[dict[str, Any]] = []

    
    candidate_profile = {
        "candidate_id": stable_candidate_id(resume_json, analysis_id),
        "name": resume_json.get("name", "Candidate"),
        "gender": _infer_gender(resume_json, resume_text),
        "location": resume_json.get("location", ""),
        "college": resume_json.get("college", ""),
        "college_tier": normalize_college_tier(resume_json.get("tier")),
        "branch": resume_json.get("branch", ""),
        "cgpa": str(resume_json.get("cgpa", "")),
        "graduation_year": str(resume_json.get("graduation_year", "")),
        "years_experience": resume_json.get("years_experience", 0) or len(resume_json.get("experience", [])),
        "career_gap_months": resume_json.get("career_gap_months", 0),
        "skills": resume_json.get("skills", []),
        "certifications": resume_json.get("certifications", []),
        "projects": resume_json.get("projects", []),
        "experience": resume_json.get("experience", []),
        "salary": resume_json.get("salary", 0),
        "job_role": _infer_job_role(resume_json, jd_text),
    }

    # Classify bias using XGBoost model
    classifier = get_bias_classifier()
    ml_flags: list[dict[str, Any]] = []
    stage1_source = "heuristics"

    if settings.use_ml_bias_classifier and classifier.is_ready:
        ml_flags = classifier.predict_flags(candidate_profile, resume_text=resume_text, patterns=patterns)
        if ml_flags:
            stage1_source = "xgboost"

    if ml_flags:
        for flag in ml_flags:
            mapped_key = str(flag.get("bias_type", ""))
            candidate_wrote, recruiter_decoded = _candidate_wrote_for_pattern(mapped_key, resume_json, resume_text)
            candidate_flags.append({
                "bias_type": mapped_key,
                "label": flag.get("label") or patterns.get(mapped_key, {}).get("label", mapped_key.replace("_", " ").title()),
                "candidate_wrote": candidate_wrote,
                "recruiter_decoded": flag.get("recruiter_decoded") or flag.get("evidence") or recruiter_decoded,
                "severity": str(flag.get("severity", "medium")).lower(),
                "fix": _build_fix(mapped_key, resume_json),
                "line_context": _context_label(mapped_key, resume_json),
                "confidence": flag.get("confidence"),
                "model": flag.get("model", stage1_source),
            })
    else:
        for pattern_key, pattern in patterns.items():
            if pattern_key in {"degree_branch_bias", "masculine_language_bias"}:
                continue

            candidate_wrote, recruiter_decoded = _candidate_wrote_for_pattern(pattern_key, resume_json, resume_text)
            triggered = _heuristic_pattern_triggered(pattern_key, resume_json, resume_text, candidate_wrote)
            if not triggered:
                continue

            candidate_flags.append({
                "bias_type": pattern_key,
                "label": pattern.get("label", pattern_key.replace("_", " ").title()),
                "candidate_wrote": candidate_wrote,
                "recruiter_decoded": recruiter_decoded,
                "severity": _severity_from_weight(float(pattern.get("severity_weight", 0.5)), True),
                "fix": _build_fix(pattern_key, resume_json),
                "line_context": _context_label(pattern_key, resume_json),
                "model": "heuristics",
            })

    log_event(
        agent="A4_BIAS",
        user_id=user_id,
        analysis_id=analysis_id,
        event="stage1_complete",
        details={"source": stage1_source, "flags_count": len(candidate_flags)},
    )

    
    branch_bias = None
    masculine_bias = None
    if jd_text:
        branch_bias = _evaluate_branch_bias(resume_json, resume_text, jd_text, ats_score, other_analyses)
        masculine_bias = _evaluate_masculine_bias(jd_text)

        if branch_bias and branch_bias.get("risk_level") != "Low":
            if not any(f.get("bias_type") == "degree_branch_bias" for f in candidate_flags):
                candidate_flags.append({
                    "bias_type": "degree_branch_bias",
                    "label": "Degree/Branch Bias",
                    "candidate_wrote": f"Degree/Branch: {resume_json.get('branch', 'N/A')}",
                    "recruiter_decoded": "Recruiters may penalize non-CSE branch despite skills.",
                    "severity": branch_bias.get("severity", "medium"),
                    "fix": branch_bias.get("recommendations", ["Highlight software projects first; use 'Software Engineer' in the title."])[0],
                    "line_context": "Education",
                    "confidence": branch_bias.get("confidence"),
                    "evidence": branch_bias.get("evidence"),
                    "skill_alignment_score": branch_bias.get("skill_alignment_score"),
                    "rankings_influenced": branch_bias.get("rankings_influenced"),
                })

        if masculine_bias and masculine_bias.get("risk_level") != "Low":
            if not any(f.get("bias_type") == "masculine_language_bias" for f in candidate_flags):
                candidate_flags.append({
                    "bias_type": "masculine_language_bias",
                    "label": "Masculine-Coded Language",
                    "candidate_wrote": ", ".join([item["term"] for item in masculine_bias.get("matched_terms", [])]),
                    "recruiter_decoded": "Masculine-coded phrasing in JD may discourage diverse candidates.",
                    "severity": masculine_bias.get("severity", "medium"),
                    "fix": masculine_bias.get("recommendation", ""),
                    "line_context": "Job Description",
                    "confidence": masculine_bias.get("confidence"),
                    "evidence": masculine_bias.get("evidence"),
                    "masculine_bias_density": masculine_bias.get("density_score"),
                    "matched_terms": masculine_bias.get("matched_terms"),
                })

    
    india_specific_count = sum(
        1 for flag in candidate_flags if patterns.get(str(flag.get("bias_type", "")), {}).get("india_specific")
    )
    high_severity_count = sum(1 for flag in candidate_flags if str(flag.get("severity", "")).lower() == "high")
    bias_score = _estimate_bias_score(candidate_flags, pattern_weights)

    # Generate explainable bias report using LLM
    summary = ""
    if candidate_flags:
        stage2_prompt = (
            "You are an expert HR auditor. Read the raw bias flags and risk metrics. "
            "Write a highly professional, encouraging, and detailed Unconscious Bias Audit Report summary (max 3 sentences). "
            "Explain what categories were triggered and how the candidate can address them. "
            "Do NOT add any new flags or change the risk ratings.\n\n"
            f"Bias Flags:\n{json.dumps(candidate_flags, indent=2)}\n"
            f"Bias Score: {bias_score}\n"
        )
        try:
            summary = call_llm(
                prompt=stage2_prompt,
                system_instruction="You are a professional HR report writer.",
                temperature=0.3,
                agent="A4_BIAS",
                user_id=user_id,
                analysis_id=analysis_id,
            )
            summary = summary.strip()
        except Exception:
            summary = "Visible bias risk increases when prestige, gaps, weak metrics, or identity cues are easy to infer in the first scan."
    else:
        summary = "No significant unconscious bias risks detected. The resume presents a balanced, metrics-driven overview."

    result = {
        "flags": candidate_flags,
        "bias_score": bias_score,
        "summary": summary,
        "clean_signals": _clean_signals(resume_json),
        "india_specific_count": india_specific_count,
        "high_severity_count": high_severity_count,
        "branch_bias": branch_bias,
        "masculine_bias": masculine_bias,
    }

    log_event(
        agent="A4_BIAS",
        user_id=user_id,
        analysis_id=analysis_id,
        event="agent_complete",
        details={
            "bias_score": result["bias_score"],
            "flags_count": len(result["flags"]),
            "india_specific_count": result["india_specific_count"],
            "high_severity_count": result["high_severity_count"],
        },
    )
    return result



def _build_fix(pattern_key: str, resume_json: dict[str, Any]) -> str:
    if pattern_key == "prestige_gap":
        return "Lead with skills, impact, and projects before mentioning college; surface metrics early."
    if pattern_key == "name_origin":
        return "Keep the resume focused on results and technical value; avoid extra identity cues in the header."
    if pattern_key == "gender_coded_language":
        return "Replace soft-skill-heavy phrasing with action verbs, tools, and measurable outcomes."
    if pattern_key == "career_gap":
        return "Add a one-line context note for the gap and emphasize recent upskilling, projects, or freelance work."
    if pattern_key == "cgpa_penalty":
        return "Place strong project outcomes and technical stack above CGPA; if the CGPA is solid, format it clearly."
    if pattern_key == "vernacular_english":
        return "Rewrite awkward phrasing into concise professional English without changing the facts."
    if pattern_key == "tier2_location":
        return "Do not foreground the location; make skills, projects, and outcomes more visible."
    if pattern_key == "project_credibility":
        return "Add technologies, scope, users, and metrics to prove project seriousness."
    return "Make the point clearer and more measurable without inventing anything."


def _context_label(pattern_key: str, resume_json: dict[str, Any]) -> str:
    if pattern_key in {"prestige_gap", "cgpa_penalty"}:
        return "Education"
    if pattern_key in {"name_origin", "gender_coded_language"}:
        return "Header / Summary"
    if pattern_key == "career_gap":
        return "Experience Timeline"
    if pattern_key == "vernacular_english":
        return "Body Copy"
    if pattern_key == "tier2_location":
        return "Location"
    if pattern_key == "project_credibility":
        return "Projects / Experience"
    return "Resume"


def _clean_signals(resume_json: dict[str, Any]) -> list[str]:
    signals: list[str] = []
    skills = resume_json.get("skills", [])
    projects = resume_json.get("projects", [])
    experience = resume_json.get("experience", [])

    if skills:
        signals.append("Clear technical skill list is present")
    if projects:
        signals.append("Projects section gives substance beyond education")
    if experience:
        signals.append("Experience section shows applied work")
    if any(isinstance(item, dict) and item.get("has_metrics") for item in projects):
        signals.append("At least one project includes measurable outcomes")
    if str(resume_json.get("cgpa", "")).strip():
        signals.append("CGPA is explicitly stated rather than hidden")
    return signals


MASCULINE_DICT_PATH = Path(__file__).parent.parent / "data" / "masculine_bias_dictionary.json"


def _load_masculine_dictionary() -> dict[str, str]:
    if not MASCULINE_DICT_PATH.exists():
        return {
            "aggressive": "proactive",
            "dominant": "market-leading",
            "competitive": "collaborative",
            "fearless": "bold",
            "assertive": "clear-communicating",
            "rockstar": "skilled professional",
            "ninja": "expert",
            "champion": "advocate",
            "killer instinct": "results-oriented focus",
            "strong leader": "effective leader",
            "driven": "motivated",
        }
    try:
        return json.loads(MASCULINE_DICT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_github_and_assessments(resume_text: str) -> dict[str, Any]:
    text_lower = resume_text.lower()
    github_matches = re.findall(r"github\.com/[a-zA-Z0-9_-]+", text_lower)

    platforms = []
    for platform in ["leetcode", "hackerrank", "codechef", "codeforces", "hackerearth", "geeksforgeeks", "kaggle"]:
        if platform in text_lower:
            platforms.append(platform.capitalize())

    return {
        "github_profile": github_matches[0] if github_matches else None,
        "has_github": len(github_matches) > 0,
        "assessment_platforms": platforms,
        "has_assessments": len(platforms) > 0,
    }


def _calculate_skill_alignment(resume_json: dict[str, Any], resume_text: str, jd_text: str) -> dict[str, Any]:
    extra_features = _extract_github_and_assessments(resume_text)

    hidden_json = dict(resume_json)
    hidden_json.pop("college", None)
    hidden_json.pop("tier", None)
    hidden_json.pop("branch", None)
    hidden_json.pop("graduation_year", None)

    prompt = SKILL_ALIGNMENT_PROMPT.format(
        jd_text=jd_text,
        hidden_json=json.dumps(hidden_json, ensure_ascii=False),
        extra_features=json.dumps(extra_features, ensure_ascii=False),
    )

    schema_hint = (
        '{"skill_alignment_score":0,"skills_rating":0,"projects_rating":0,"experience_rating":0,'
        '"certifications_rating":0,"github_assessment_rating":0,"reasoning":""}'
    )

    try:
        alignment_payload = call_llm_json(
            prompt=prompt,
            system_instruction="You are a strict JSON-only skill alignment assessor.",
            schema_hint=schema_hint,
            agent="A4_BIAS",
        )
        return alignment_payload
    except Exception:
        skills = resume_json.get("skills", [])
        score = min(
            100.0,
            len(skills) * 8.0 + (10.0 if extra_features["has_github"] else 0.0) + (10.0 if extra_features["has_assessments"] else 0.0),
        )
        return {
            "skill_alignment_score": score,
            "skills_rating": min(40.0, len(skills) * 4.0),
            "projects_rating": min(20.0, len(resume_json.get("projects", [])) * 5.0),
            "experience_rating": min(20.0, len(resume_json.get("experience", [])) * 5.0),
            "certifications_rating": min(10.0, len(resume_json.get("certifications", [])) * 5.0),
            "github_assessment_rating": 10.0 if (extra_features["has_github"] or extra_features["has_assessments"]) else 0.0,
            "reasoning": "Fallback heuristic calculation due to LLM error.",
        }


def _evaluate_branch_bias(
    resume_json: dict[str, Any],
    resume_text: str,
    jd_text: str,
    current_ats_score: float | None = None,
    other_analyses: list[Any] | None = None,
) -> dict[str, Any]:
    branch = str(resume_json.get("branch", "")).strip().lower()

    non_cse_branches = ["mechanical", "ece", "eee", "civil", "chemical", "aerospace", "metallurgy", "production", "instrumentation", "biotech"]
    is_non_cse = any(b in branch for b in non_cse_branches) or (
        len(branch) > 0 and not any(cse in branch for cse in ["computer", "cse", "it", "information", "software"])
    )

    alignment = _calculate_skill_alignment(resume_json, resume_text, jd_text)
    skill_score = alignment.get("skill_alignment_score", 0.0)

    rankings_influenced = False
    evidence = ""
    risk_level = "Low"
    severity = "low"
    confidence = "Medium"
    recommendations = []

    if is_non_cse:
        bias_by_comparison = False
        compared_evidence = []

        if other_analyses and current_ats_score is not None:
            for other in other_analyses:
                other_parsed = getattr(other, "parsed_json", None) or {}
                other_ats = getattr(other, "ats_result", None) or {}

                other_branch = str(other_parsed.get("branch", "")).strip().lower()
                is_other_cse = any(cse in other_branch for cse in ["computer", "cse", "it", "information", "software"])

                if is_other_cse:
                    other_bias = getattr(other, "bias_result", None) or {}
                    other_branch_bias = other_bias.get("branch_bias") or {}
                    other_skill_score = other_branch_bias.get("skill_alignment_score")

                    if other_skill_score is None:
                        other_alignment = _calculate_skill_alignment(other_parsed, "", jd_text)
                        other_skill_score = other_alignment.get("skill_alignment_score", 0.0)

                    other_ats_score = other_ats.get("score", 0.0)

                    if skill_score > other_skill_score and current_ats_score < other_ats_score:
                        bias_by_comparison = True
                        compared_evidence.append(
                            f"Candidate ({resume_json.get('branch', 'Non-CS')}) has a higher Skill Alignment Score ({skill_score:.0f}%) "
                            f"than CSE/IT candidate ({other_parsed.get('branch', 'CSE')}) who has {other_skill_score:.0f}%, "
                            f"but was ranked lower (ATS Score {current_ats_score:.0f}% vs {other_ats_score:.0f}%)."
                        )

        if bias_by_comparison:
            risk_level = "High"
            severity = "high"
            confidence = "High"
            rankings_influenced = True
            evidence = "Direct ranking discrepancy found: " + " | ".join(compared_evidence)
        else:
            if skill_score >= 75.0:
                risk_level = "High"
                severity = "high"
                confidence = "Medium"
                evidence = (
                    f"Candidate is from the '{resume_json.get('branch', 'Non-CS')}' branch but has strong skill alignment ({skill_score:.0f}%). "
                    "In typical tech recruitment pipelines, candidates from non-CSE/IT backgrounds face a high risk of being filtered out "
                    "or ranked lower, even when possessing superior technical skills."
                )
            elif skill_score >= 50.0:
                risk_level = "Medium"
                severity = "medium"
                confidence = "Medium"
                evidence = (
                    f"Candidate has moderate skill alignment ({skill_score:.0f}%) from branch '{resume_json.get('branch', 'Non-CS')}'. "
                    "They may experience medium risk of being deprioritized relative to CSE/IT peers."
                )
            else:
                risk_level = "Low"
                severity = "low"
                confidence = "Low"
                evidence = f"Candidate has low skill alignment ({skill_score:.0f}%), so branch-based priority differences are secondary."

        recommendations = [
            "Restructure the resume to put core technical skills and projects at the very top.",
            "Include links to live project demos, GitHub repositories, and coding platform profiles (e.g. LeetCode, HackerRank) to provide degree-independent proof of skills.",
            "If applying for software roles, highlight coursework or certifications in Data Structures, Algorithms, and Software Engineering to bridge the branch gap.",
        ]
    else:
        risk_level = "Low"
        severity = "low"
        confidence = "High"
        evidence = f"Candidate is from branch '{resume_json.get('branch', 'CSE/IT')}' which is highly preferred for software/IT roles."
        recommendations = ["Keep leveraging the branch advantage while ensuring skills match the JD."]

    return {
        "risk_level": risk_level,
        "skill_alignment_score": skill_score,
        "severity": severity,
        "confidence": confidence,
        "evidence": evidence,
        "recommendations": recommendations,
        "rankings_influenced": rankings_influenced,
    }


def _evaluate_masculine_bias(jd_text: str) -> dict[str, Any]:
    bias_dict = _load_masculine_dictionary()

    matched_terms = []
    total_matches = 0

    words = [w for w in re.findall(r"\w+", jd_text) if w]
    word_count = max(1, len(words))

    jd_lower = jd_text.lower()
    for term, alternative in bias_dict.items():
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        matches = len(re.findall(pattern, jd_lower))
        if matches > 0:
            matched_terms.append({"term": term, "replacement": alternative, "count": matches})
            total_matches += matches

    density_score = round((total_matches / word_count) * 100, 2)

    if total_matches == 0:
        risk_level = "Low"
        severity = "low"
    elif total_matches <= 2:
        risk_level = "Medium"
        severity = "medium"
    else:
        risk_level = "High"
        severity = "high"

    if total_matches > 0:
        terms_str = ", ".join([f"'{item['term']}'" for item in matched_terms])
        evidence = (
            f"Found {total_matches} masculine-coded terms ({terms_str}) in the job description, "
            f"resulting in a Masculine Bias Density of {density_score:.2f}%."
        )
        recommendations = [
            f"Replace '{item['term']}' with '{item['replacement']}' to appeal to a more diverse candidate pool."
            for item in matched_terms
        ]
    else:
        evidence = "No masculine-coded terms from the dictionary were detected in the job description."
        recommendations = ["The job description language is inclusive and balanced."]

    return {
        "risk_level": risk_level,
        "density_score": density_score,
        "matched_terms": matched_terms,
        "severity": severity,
        "confidence": "High",
        "evidence": evidence,
        "recommendation": "; ".join(recommendations) if recommendations else "No changes needed.",
    }
