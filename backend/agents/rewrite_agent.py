from __future__ import annotations

import json
import re
from typing import Any

from backend.core.logging_config import log_event
from backend.ml.rewrite_mapper import map_rewrite_stage1
from backend.utils.llm_client import call_llm_json
from backend.utils.prompts import REWRITE_PROMPT
from backend.agents.ats_matcher import match_ats

SYSTEM_INSTRUCTION = """You are MirrorCue's rewrite engine. Your inviolable rules:

1. NEVER invent, extrapolate, or assume any metric. Use ONLY numbers explicitly listed
   in VERIFIED METRICS. If no metric exists for a bullet, rewrite without any number.

2. NEVER add a technology, tool, or framework to a bullet unless it ALREADY EXISTS
   in that item's original content (mapped_facts / technologies) OR it appears in
   CONFIRMED KEYWORDS below. Technologies listed in UNCONFIRMED KEYWORDS must NOT
   appear anywhere in the rewritten output — not in bullets, not in tech stacks,
   not as parenthetical notes, not as "future plans".

3. Preserve all original specific metrics exactly (e.g. "92% accuracy",
   "200+ query types", "30% reduction"). Do NOT replace them with generic numbers.

4. Remove every bias-triggering phrase identified in BIAS REMOVAL INSTRUCTIONS.

5. Use powerful action verbs: Engineered, Architected, Deployed, Optimized,
   Automated, Reduced, Scaled, Developed, Implemented, Built.

6. Each bullet format: Action Verb + What You Did + Technology Used + [Metric if available].

7. Return ONLY valid JSON."""


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _flatten_text_list(values: Any) -> str:
    """Flatten certifications/achievements (list of strings or dicts) into a
    single space-joined string so this content is preserved when rescoring.

    FIX (NEW): the old rescore text dropped certifications/achievements entirely,
    so any JD-keyword matches coming from those sections were lost on rescore
    even though nothing about those sections changed — making the "after" score
    look artificially lower than it should be.
    """
    parts: list[str] = []
    for v in _as_list(values):
        if isinstance(v, dict):
            parts.append(" ".join(str(x) for x in v.values() if isinstance(x, (str, int, float))))
        elif v is not None:
            text = str(v).strip()
            if text:
                parts.append(text)
    return " ".join(parts)


def _collect_keywords_added(stage1_payload: dict[str, Any]) -> int:
    return len(_as_list(stage1_payload.get("confirmed_keywords", [])))


def _collect_bias_phrases_removed(bias_result: dict[str, Any]) -> int:
    count = 0
    for flag in _as_list(bias_result.get("flags", [])):
        if not isinstance(flag, dict):
            continue
        wrote = str(flag.get("candidate_wrote", "")).strip()
        if wrote and wrote not in ("resume content", "project details", "experience details", ""):
            count += 1
    return count


def _format_scanned_experience(resume_json: dict[str, Any], ats_result: dict[str, Any]) -> str:
    years = resume_json.get("years_experience")
    if isinstance(years, (int, float)) and years > 0:
        label = "year" if years == 1 else "years"
        return f"{int(years)} {label}"
    exp_count = len(_as_list(resume_json.get("experience", [])))
    if exp_count:
        label = "year" if exp_count == 1 else "years"
        return f"{exp_count} {label}"
    seniority = str(ats_result.get("jd_seniority_level", "")).strip()
    return seniority or "0 years"


def _build_original_experience(resume_json: dict[str, Any]) -> list[dict[str, Any]]:
    original_items: list[dict[str, Any]] = []
    for item in _as_list(resume_json.get("experience", [])):
        if not isinstance(item, dict):
            continue
        duration = (
            str(item.get("duration", "")).strip()
            or str(item.get("dates", "")).strip()
            or str(item.get("date_range", "")).strip()
            or str(item.get("period", "")).strip()
        )
        original_items.append(
            {
                "title": str(item.get("title", "")),
                "company": str(item.get("company", "")),
                "duration": duration,
                "bullets": [str(item.get("description", ""))] if item.get("description") else [],
            }
        )
    return original_items


def _build_original_projects(resume_json: dict[str, Any]) -> list[dict[str, Any]]:
    original_items: list[dict[str, Any]] = []
    for item in _as_list(resume_json.get("projects", [])):
        if not isinstance(item, dict):
            continue
        original_items.append(
            {
                "name": str(item.get("name", "")),
                "tech_stack": [str(tech) for tech in _as_list(item.get("tech", []))],
                "bullets": [str(item.get("description", ""))] if item.get("description") else [],
            }
        )
    return original_items


def _build_keyword_guardrails(stage1_payload: dict[str, Any]) -> str:
    confirmed = _as_list(stage1_payload.get("confirmed_keywords", []))
    unconfirmed = _as_list(stage1_payload.get("unconfirmed_keywords", []))

    block = "\n\nKEYWORD GUARDRAILS (strictly enforced):\n"

    if confirmed:
        block += (
            "CONFIRMED KEYWORDS (candidate verified using these in Q&A — safe to include "
            "where they naturally fit the bullet context):\n"
            + "  " + ", ".join(confirmed) + "\n"
        )
    else:
        block += "CONFIRMED KEYWORDS: none — do not add any missing JD keywords.\n"

    if unconfirmed:
        block += (
            "UNCONFIRMED KEYWORDS (candidate did NOT confirm using these — "
            "NEVER add any of these to any bullet, tech stack, or summary):\n"
            + "  " + ", ".join(unconfirmed) + "\n"
        )

    return block


def _build_question_map(
    qa_questions: list[dict[str, Any]] | None,
    qa_answers: dict[str, str],
) -> dict[str, dict[str, str]]:
    """Map question IDs to their full context — item_name, section, answer_type, answer."""
    question_map: dict[str, dict[str, str]] = {}
    if not qa_questions:
        return question_map
    for q in qa_questions:
        if not isinstance(q, dict):
            continue
        qid = str(q.get("id", "")).strip()
        if not qid:
            continue
        question_map[qid] = {
            "item_name": str(q.get("item_name", "")).strip(),
            "section": str(q.get("section", "")).strip(),
            "answer_type": str(q.get("answer_type", "")).strip(),
            "answer": str(qa_answers.get(qid, "")).strip(),
        }
    return question_map


def _is_negative_answer(text: str) -> bool:
    return bool(re.search(
        r"\b(no[,.]?\s|haven'?t|have not|not yet|don'?t have|never used|no experience)\b",
        text, re.IGNORECASE,
    ))


def _build_qa_context_block(question_map: dict[str, dict[str, str]]) -> str:
    metric_lines: list[str] = []
    no_metric_lines: list[str] = []

    for qid, meta in question_map.items():
        answer = meta["answer"]
        if not answer or _is_negative_answer(answer):
            continue

        item_name = meta["item_name"] or qid
        section = meta["section"]
        label = f"[{section}] {item_name}" if section else item_name

        if re.search(r"\d", answer):
            metric_lines.append(f"  - {label}: {answer}")
        else:
            no_metric_lines.append(f"  - {label}: {answer}")

    block = ""
    if metric_lines:
        block += (
            "\n\nVERIFIED METRICS FROM Q&A (use ONLY these numbers — do not invent any others):\n"
            + "\n".join(metric_lines) + "\n"
        )
    if no_metric_lines:
        block += (
            "\nADDITIONAL CONTEXT FROM Q&A (no metrics — use for richer bullet phrasing only):\n"
            + "\n".join(no_metric_lines) + "\n"
        )
    if not metric_lines and not no_metric_lines:
        block += "\n\nVERIFIED METRICS FROM Q&A: none provided — rewrite with action verbs only, zero invented numbers.\n"

    return block


def _derive_confirmed_keywords(
    question_map: dict[str, dict[str, str]],
    missing_keywords: list[str],
) -> list[str]:
    confirmed: list[str] = []
    for qid, meta in question_map.items():
        answer = meta["answer"]
        if not answer or _is_negative_answer(answer):
            continue
        if meta["answer_type"] != "technology":
            continue
        for keyword in missing_keywords:
            if keyword.lower() in answer.lower() and keyword not in confirmed:
                confirmed.append(keyword)
    return confirmed


def rewrite_resume(
    resume_json: dict[str, Any],
    jd_text: str,
    ats_result: dict[str, Any],
    bias_result: dict[str, Any],
    qa_answers: dict[str, str],
    qa_questions: list[dict[str, Any]] | None = None,
    user_id: str | None = None,
    analysis_id: str | None = None,
) -> dict[str, Any]:
    log_event(
        agent="A6_REWRITE",
        user_id=user_id,
        analysis_id=analysis_id,
        event="agent_start",
        details={
            "resume_sections": {
                "experience_count": len(_as_list(resume_json.get("experience", []))),
                "project_count": len(_as_list(resume_json.get("projects", []))),
            },
            "ats_missing_keywords_count": len(_as_list(ats_result.get("missing_keywords", []))),
            "bias_flags_count": len(_as_list(bias_result.get("flags", []))),
            "qa_answers_count": len(qa_answers),
            "qa_questions_count": len(qa_questions or []),
        },
    )

    question_map = _build_question_map(qa_questions, qa_answers)

    missing_keywords: list[str] = [
        str(k) for k in _as_list(ats_result.get("missing_keywords", [])) if str(k).strip()
    ]

    confirmed_from_qa = _derive_confirmed_keywords(question_map, missing_keywords)

    stage1_payload = map_rewrite_stage1(resume_json, ats_result, qa_answers, bias_result)

    existing_confirmed = _as_list(stage1_payload.get("confirmed_keywords", []))
    merged_confirmed = list(dict.fromkeys(existing_confirmed + confirmed_from_qa))
    stage1_payload["confirmed_keywords"] = merged_confirmed
    stage1_payload["unconfirmed_keywords"] = [
        k for k in missing_keywords if k not in merged_confirmed
    ]

    resume_has_experience = bool(_as_list(resume_json.get("experience")))
    resume_has_projects = bool(_as_list(resume_json.get("projects")))

    if (
        (resume_has_experience and not _as_list(stage1_payload.get("experience")))
        or (resume_has_projects and not _as_list(stage1_payload.get("projects")))
    ):
        fallback_exp = []
        for exp in _as_list(resume_json.get("experience", [])):
            if not isinstance(exp, dict):
                continue
            fallback_exp.append({
                "title": str(exp.get("title", "")),
                "company": str(exp.get("company", "")),
                "mapped_facts": [str(exp.get("description", ""))] if exp.get("description") else [],
                "technologies": [],
                "metrics": [],
                "keywords_to_add": [],
                "bias_phrases_to_remove": [],
            })
        fallback_proj = []
        for proj in _as_list(resume_json.get("projects", [])):
            if not isinstance(proj, dict):
                continue
            fallback_proj.append({
                "name": str(proj.get("name", "")),
                "mapped_facts": [str(proj.get("description", ""))] if proj.get("description") else [],
                "technologies": [str(t) for t in _as_list(proj.get("tech", []))],
                "metrics": [],
                "keywords_to_add": [],
            })
        stage1_payload = {
            "experience": fallback_exp,
            "projects": fallback_proj,
            "confirmed_keywords": merged_confirmed,
            "unconfirmed_keywords": [k for k in missing_keywords if k not in merged_confirmed],
        }

    log_event(
        agent="A6_REWRITE",
        user_id=user_id,
        analysis_id=analysis_id,
        event="stage1_complete",
        details={
            "source": "rule_mapper",
            "experience_count": len(_as_list(stage1_payload.get("experience"))),
            "projects_count": len(_as_list(stage1_payload.get("projects"))),
            "confirmed_keywords": stage1_payload.get("confirmed_keywords", []),
            "unconfirmed_keywords": stage1_payload.get("unconfirmed_keywords", []),
        },
    )

    detailed_schema = """{
  "rewritten_experience": [
    {"title": "string", "company": "string", "duration": "string", "bullets": ["string"]}
  ],
  "rewritten_projects": [
    {"name": "string", "tech_stack": ["string"], "bullets": ["string"]}
  ],
  "rewritten_summary": "string",
  "changes_summary": "string"
}"""

    bias_flags = _as_list(bias_result.get("flags", []))
    bias_removal_lines = []
    for flag in bias_flags:
        wrote = str(flag.get("candidate_wrote", "")).strip()
        fix = str(flag.get("fix", "")).strip()
        if wrote and wrote not in ("resume content", "project details", "experience details", ""):
            if fix:
                bias_removal_lines.append(f'  - Replace or rephrase "{wrote}" → {fix}')
            else:
                bias_removal_lines.append(f'  - Remove or rephrase: "{wrote}"')

    bias_removal_block = ""
    if bias_removal_lines:
        bias_removal_block = (
            "\n\nBIAS REMOVAL INSTRUCTIONS (apply to rewritten bullets):\n"
            + "\n".join(bias_removal_lines)
            + "\n"
        )

    qa_context_block = _build_qa_context_block(question_map)
    keyword_guardrails = _build_keyword_guardrails(stage1_payload)

    stage2_prompt = (
        f"Polishing instructions:\n{REWRITE_PROMPT}\n\n"
        f"Raw mapped facts and metrics:\n{json.dumps(stage1_payload, indent=2)}\n"
        f"{qa_context_block}"
        f"{keyword_guardrails}"
        f"{bias_removal_block}\n"
        "FINAL REMINDER: Do NOT add any technology from UNCONFIRMED KEYWORDS to any bullet. "
        "Do NOT invent any percentage, user count, or metric not listed in VERIFIED METRICS. "
        "Preserve all original specific numbers exactly as they appear in mapped_facts.\n\n"
        "Generate the final polished resume bullets matching the requested JSON schema."
    )

    stage2_system_instruction = (
        "You are an elite executive resume writer. "
        "Rewrite the mapped facts into high-impact professional bullets using strong action verbs. "
        "Keep all original technical terms and metrics exactly as provided — change nothing. "
        "If a duration is empty, output an empty string \"\" — NEVER write 'Not specified' or 'N/A'. "
        "Return ONLY valid JSON."
    )

    try:
        llm_payload = call_llm_json(
            prompt=stage2_prompt,
            system_instruction=stage2_system_instruction,
            schema_hint=detailed_schema,
            agent="A6_REWRITE",
            user_id=user_id,
            analysis_id=analysis_id,
        )

        log_event(
            agent="A6_REWRITE",
            user_id=user_id,
            analysis_id=analysis_id,
            event="llm_response_received",
            details={
                "has_rewritten_experience": "rewritten_experience" in llm_payload,
                "experience_count": len(_as_list(llm_payload.get("rewritten_experience", []))),
                "has_rewritten_projects": "rewritten_projects" in llm_payload,
                "projects_count": len(_as_list(llm_payload.get("rewritten_projects", []))),
            },
        )
    except Exception as exc:
        log_event(
            level=40,
            agent="A6_REWRITE",
            user_id=user_id,
            analysis_id=analysis_id,
            event="llm_error",
            details={"error": str(exc)},
            exc_info=True,
        )
        llm_payload = {}

    rewritten_experience = llm_payload.get("rewritten_experience") if isinstance(llm_payload, dict) else None
    rewritten_projects = llm_payload.get("rewritten_projects") if isinstance(llm_payload, dict) else None

    is_valid_experience = isinstance(rewritten_experience, list)
    is_valid_projects = isinstance(rewritten_projects, list)

    log_event(
        agent="A6_REWRITE",
        user_id=user_id,
        analysis_id=analysis_id,
        event="validation_check",
        details={
            "is_valid_experience": is_valid_experience,
            "is_valid_projects": is_valid_projects,
            "will_use_original": not (is_valid_experience or is_valid_projects),
        },
    )

    final_rewritten_experience = rewritten_experience if is_valid_experience else _build_original_experience(resume_json)
    final_rewritten_projects = rewritten_projects if is_valid_projects else _build_original_projects(resume_json)
    final_rewritten_summary = llm_payload.get("rewritten_summary", "") if isinstance(llm_payload, dict) else ""
    if not final_rewritten_summary:
        final_rewritten_summary = "MirrorCue refined the resume to foreground measurable impact, cleaner wording, and stronger JD alignment."

    rewritten_skills = list(_as_list(resume_json.get("skills", [])))
    for kw in merged_confirmed:
        if kw and kw not in rewritten_skills:
            rewritten_skills.append(kw)

    rewritten_bullets = []
    for exp in final_rewritten_experience:
        rewritten_bullets.extend(exp.get("bullets", []))
    for proj in final_rewritten_projects:
        rewritten_bullets.extend(proj.get("bullets", []))
        rewritten_bullets.extend([str(tech) for tech in proj.get("tech_stack", [])])

    # ── FIX (NEW): preserve certifications/achievements for the rescore ──
    certifications_text = _flatten_text_list(resume_json.get("certifications", []))
    achievements_text = _flatten_text_list(resume_json.get("achievements", []))

    rewritten_text = "\n".join([
        final_rewritten_summary,
        " ".join(rewritten_skills),
        " ".join(rewritten_bullets),
        certifications_text,
        achievements_text,
    ])

    rewritten_resume_json = {
        "name": resume_json.get("name", ""),
        "college": resume_json.get("college", ""),
        "tier": resume_json.get("tier", ""),
        "branch": resume_json.get("branch", ""),
        "cgpa": resume_json.get("cgpa", ""),
        "skills": rewritten_skills,
        "experience": final_rewritten_experience,
        "projects": final_rewritten_projects,
        "certifications": resume_json.get("certifications", []),
        "achievements": resume_json.get("achievements", []),
    }

    recalculated_ats = match_ats(
        resume_json=rewritten_resume_json,
        resume_text=rewritten_text,
        jd_text=jd_text,
        user_id=user_id,
        analysis_id=analysis_id,
    )
    computed_ats_after = round(
        recalculated_ats.get("final_score") or recalculated_ats.get("score") or 0, 2
    )
    ats_before = round(
        ats_result.get("final_score") or ats_result.get("score") or 0, 2
    )

    # ── FIX (NEW): never let the rewrite make the ATS score worse. ──
    # If the LLM rewrite scores below the original, fall back to the
    # candidate's ORIGINAL experience/project bullets (keeping only the
    # confirmed-keyword skill additions) and rescore that instead. Whichever
    # version scores higher becomes the final result, so ats_score_after
    # is never lower than ats_score_before unless even the original bullets
    # plus confirmed keywords can't match it.
    used_fallback = False
    if computed_ats_after < ats_before:
        fallback_experience = _build_original_experience(resume_json)
        fallback_projects = _build_original_projects(resume_json)

        fallback_bullets: list[str] = []
        for exp in fallback_experience:
            fallback_bullets.extend(exp.get("bullets", []))
        for proj in fallback_projects:
            fallback_bullets.extend(proj.get("bullets", []))
            fallback_bullets.extend(proj.get("tech_stack", []))

        fallback_text = "\n".join([
            final_rewritten_summary,
            " ".join(rewritten_skills),
            " ".join(fallback_bullets),
            certifications_text,
            achievements_text,
        ])

        fallback_resume_json = {
            "name": resume_json.get("name", ""),
            "college": resume_json.get("college", ""),
            "tier": resume_json.get("tier", ""),
            "branch": resume_json.get("branch", ""),
            "cgpa": resume_json.get("cgpa", ""),
            "skills": rewritten_skills,
            "experience": fallback_experience,
            "projects": fallback_projects,
            "certifications": resume_json.get("certifications", []),
            "achievements": resume_json.get("achievements", []),
        }

        fallback_ats = match_ats(
            resume_json=fallback_resume_json,
            resume_text=fallback_text,
            jd_text=jd_text,
            user_id=user_id,
            analysis_id=analysis_id,
        )
        fallback_score = round(
            fallback_ats.get("final_score") or fallback_ats.get("score") or 0, 2
        )

        log_event(
            agent="A6_REWRITE",
            user_id=user_id,
            analysis_id=analysis_id,
            event="rewrite_regression_check",
            details={
                "ats_before": ats_before,
                "llm_rewrite_score": computed_ats_after,
                "fallback_score": fallback_score,
            },
        )

        if fallback_score >= computed_ats_after:
            final_rewritten_experience = fallback_experience
            final_rewritten_projects = fallback_projects
            computed_ats_after = fallback_score
            used_fallback = True
            log_event(
                agent="A6_REWRITE",
                user_id=user_id,
                analysis_id=analysis_id,
                event="rewrite_regression_fallback_applied",
                details={
                    "reason": (
                        "LLM rewrite scored below original; kept original bullets "
                        "with confirmed-keyword skill additions only"
                    ),
                    "final_ats_after": computed_ats_after,
                },
            )

    result = {
        "original_experience": _build_original_experience(resume_json),
        "original_projects": _build_original_projects(resume_json),
        "rewritten_experience": final_rewritten_experience,
        "rewritten_projects": final_rewritten_projects,
        "rewritten_summary": final_rewritten_summary,
        "ats_score_before": ats_before,
        "ats_score_after": computed_ats_after,
        "ats_score_delta": round(computed_ats_after - ats_before, 2),
        "total_keywords_added": _collect_keywords_added(stage1_payload),
        "total_bias_phrases_removed": 0 if used_fallback else _collect_bias_phrases_removed(bias_result),
        "confirmed_keywords": stage1_payload.get("confirmed_keywords", []),
        "unconfirmed_keywords": stage1_payload.get("unconfirmed_keywords", []),
        "changes_summary": llm_payload.get("changes_summary", "") if isinstance(llm_payload, dict) else "",
        "validation_error": not (is_valid_experience or is_valid_projects),
        "used_fallback": used_fallback,
    }

    if not result["rewritten_summary"]:
        result["rewritten_summary"] = (
            "MirrorCue refined the resume to foreground measurable impact, cleaner wording, and stronger JD alignment."
        )

    if used_fallback:
        result["changes_summary"] = (
            "The AI-generated rewrite did not improve ATS alignment for this job description, "
            "so your original experience and project bullets were kept unchanged. "
            "Any confirmed keywords were still added to your skills section."
        )
    elif not result["changes_summary"]:
        result["changes_summary"] = (
            "The rewrite strengthens action verbs and preserves all original metrics. "
            "Only keywords confirmed in the Q&A session were added."
        )

    log_event(
        agent="A6_REWRITE",
        user_id=user_id,
        analysis_id=analysis_id,
        event="agent_complete",
        details={
            "ats_score_before": result["ats_score_before"],
            "ats_score_after": result["ats_score_after"],
            "ats_score_delta": result["ats_score_delta"],
            "keywords_added": result["total_keywords_added"],
            "confirmed_keywords": result["confirmed_keywords"],
            "bias_phrases_removed": result["total_bias_phrases_removed"],
            "used_fallback": result["used_fallback"],
        },
    )
    return result