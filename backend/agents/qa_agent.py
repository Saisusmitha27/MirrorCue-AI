from __future__ import annotations

import re
from typing import Any

from backend.core.logging_config import log_event
from backend.utils.llm_client import call_llm_json
from backend.utils.prompts import QA_PROMPT


def _has_metric(text: str) -> bool:
    patterns = [
        r"\b\d+%\b",
        r"\b\d+\+?\b",
        r"\b\d+\s?(users|users/day|daily users|DAU|projects|clients|teams|people|hrs|hours|days|weeks|months|transactions)\b",
        r"\b(improved|reduced|increased|decreased|optimized|scaled)\b",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _question_id(index: int) -> str:
    return f"q{index + 1}"


def _build_question(
    *,
    section: str,
    item_name: str,
    question_text: str,
    why_needed: str,
    example_answer: str,
    answer_type: str,
    index: int,
) -> dict[str, str]:
    return {
        "id": _question_id(index),
        "section": section,
        "item_name": item_name,
        "question": question_text,
        "why_needed": why_needed,
        "example_answer": example_answer,
        "answer_type": answer_type,
    }


def _collect_vague_experiences(resume_json: dict[str, Any]) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []
    for index, item in enumerate(resume_json.get("experience", [])[:5]):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "Experience")).strip() or "Experience"
        company = str(item.get("company", "")).strip()
        description = str(item.get("description", "")).strip()
        duration = str(item.get("duration", "")).strip()
        if not description:
            continue
        if _has_metric(description):
            continue
        questions.append(
            _build_question(
                section="experience",
                item_name=title,
                question_text=(
                    f"During your {title} role at {company}, what was the most concrete outcome you "
                    f"can point to — a percentage improvement, user count, or time saved?"
                ),
                why_needed="A specific, defensible metric makes this bullet stand out in a 7-second recruiter scan.",
                example_answer="e.g., Reduced model inference time by 35%, or Handled 500 requests/day",
                answer_type="metric",
                index=index,
            )
        )
        if len(questions) >= 5:
            break
        if duration and not _has_metric(duration):
            questions.append(
                _build_question(
                    section="experience",
                    item_name=title,
                    question_text=(
                        f"For your {title} work at {company}, how large was the team and what was "
                        f"your specific responsibility within it?"
                    ),
                    why_needed="Team size and ownership scope help recruiters gauge seniority and collaboration style.",
                    example_answer="e.g., 3-person ML team; I owned the data pipeline and model evaluation",
                    answer_type="scope",
                    index=len(questions),
                )
            )
        if len(questions) >= 5:
            break
    return questions[:5]


def _collect_vague_projects(resume_json: dict[str, Any]) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []
    for index, item in enumerate(resume_json.get("projects", [])[:5]):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "Project")).strip() or "Project"
        description = str(item.get("description", "")).strip()
        tech = item.get("tech", [])
        has_metrics = bool(item.get("has_metrics"))
        if not description:
            continue
        if has_metrics or _has_metric(description):
            continue
        if not tech:
            questions.append(
                _build_question(
                    section="project",
                    item_name=name,
                    question_text=f"What specific technologies and tools did you use to build {name}?",
                    why_needed="A concrete tech stack makes the project credible and surfaces ATS keywords.",
                    example_answer="e.g., Python, Flask, PostgreSQL, React — deployed on Render",
                    answer_type="technology",
                    index=index,
                )
            )
        else:
            questions.append(
                _build_question(
                    section="project",
                    item_name=name,
                    question_text=(
                        f"For {name}, what was the real-world outcome — how many users, "
                        f"what performance improvement, or what problem did it solve?"
                    ),
                    why_needed="Concrete outcomes reduce 'toy project' perception and show real impact.",
                    example_answer="e.g., Used by 20 classmates during a demo; cut report generation from 10 min to 45 sec",
                    answer_type="metric",
                    index=index,
                )
            )
        if len(questions) >= 5:
            break
    return questions[:5]


def _collect_depth_questions(resume_json: dict[str, Any]) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []

    for item in resume_json.get("projects", [])[:5]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "Project")).strip() or "Project"
        description = str(item.get("description", "")).strip()
        if not description or not _has_metric(description):
            continue
        has_deployment = bool(re.search(
            r"\b(deployed|live|production|github|hosted|render|vercel|heroku|netlify|aws|gcp|azure|huggingface|streamlit cloud)\b",
            description, re.IGNORECASE,
        ))
        if not has_deployment:
            questions.append(
                _build_question(
                    section="project",
                    item_name=name,
                    question_text=(
                        f"Is {name} deployed or hosted anywhere — Render, Streamlit Cloud, GitHub, "
                        f"Hugging Face Spaces? If so, has anyone outside your team used it?"
                    ),
                    why_needed=(
                        "Deployed projects signal production-readiness. Even a college demo or "
                        "GitHub link makes a project significantly more credible."
                    ),
                    example_answer=(
                        "e.g., Hosted on Streamlit Cloud, used by 40 peers during a demo day — "
                        "or: Not deployed, ran locally only"
                    ),
                    answer_type="scope",
                    index=len(questions),
                )
            )
        if len(questions) >= 3:
            break

    for item in resume_json.get("experience", [])[:5]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        company = str(item.get("company", "")).strip()
        description = str(item.get("description", "")).strip()
        if not description or not _has_metric(description):
            continue
        has_team_mention = bool(re.search(
            r"\b(team|collaborat|colleague|cross.functional|group|member|alongside|together)\b",
            description, re.IGNORECASE,
        ))
        if not has_team_mention:
            questions.append(
                _build_question(
                    section="experience",
                    item_name=title,
                    question_text=(
                        f"At {company}, were you working solo or as part of a team? "
                        f"What was your specific ownership within the project?"
                    ),
                    why_needed=(
                        "Collaboration context and ownership scope show recruiters how you work "
                        "within a team and what level of responsibility you carried."
                    ),
                    example_answer=(
                        "e.g., Built the ML pipeline independently under a senior mentor's review — "
                        "or: Part of a 4-person team, I owned model evaluation and hyperparameter tuning"
                    ),
                    answer_type="scope",
                    index=len(questions),
                )
            )
        if len(questions) >= 3:
            break

    return questions[:3]


def _collect_jd_keyword_questions(atts_missing_keywords: list[str]) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []
    for index, keyword in enumerate(atts_missing_keywords[:2]):
        keyword_text = str(keyword).strip()
        if not keyword_text:
            continue
        questions.append(
            _build_question(
                section="project" if index % 2 else "experience",
                item_name=keyword_text,
                question_text=(
                    f"Have you worked with {keyword_text} in any project, internship, or self-study, "
                    f"even briefly? If yes, describe what you built or configured with it."
                ),
                why_needed=(
                    f"{keyword_text} appears in the job description. If you have genuine hands-on "
                    f"experience — even a small personal project — we can surface it in your resume."
                ),
                example_answer=(
                    f"e.g., Set up a basic {keyword_text} pipeline for a college project last semester "
                    f"— or: No, I haven't used it yet but I understand the concepts."
                ),
                answer_type="technology",
                index=index,
            )
        )
        if len(questions) >= 2:
            break
    return questions[:2]


def _build_generic_fallback_question(
    resume_json: dict[str, Any],
    missing_keywords: list[str],
    index: int = 0,
) -> dict[str, str]:
    if missing_keywords:
        keyword = str(missing_keywords[0]).strip()
        if keyword:
            return _build_question(
                section="experience",
                item_name=keyword,
                question_text=(
                    f"Have you worked with {keyword} in any project, internship, or self-study, "
                    f"even briefly? If yes, describe what you built or configured with it."
                ),
                why_needed=(
                    f"{keyword} appears in the job description and is currently missing from your resume."
                ),
                example_answer=(
                    f"e.g., Used {keyword} in a small personal project — or: No, I haven't used it yet."
                ),
                answer_type="technology",
                index=index,
            )

    return _build_question(
        section="general",
        item_name="Resume",
        question_text=(
            "What is the single most impressive, measurable outcome from your projects or "
            "internships that isn't fully captured in your current resume?"
        ),
        why_needed="A standout, quantified achievement helps your resume rise above other candidates.",
        example_answer="e.g., Reduced processing time by 40%, or built a tool used by 50+ students.",
        answer_type="metric",
        index=index,
    )


def _reindex_questions(questions: list[dict[str, str]]) -> list[dict[str, str]]:
    """Re-assign sequential IDs. Deduplication happens before this call."""
    return [
        {**q, "id": _question_id(i)}
        for i, q in enumerate(questions[:5])
    ]


def _resume_needs_qa(resume_json: dict[str, Any]) -> bool:
    strict_metric_patterns = [
        r"\b\d+\s*%",
        r"\b\d+\s?(users|DAU|clients|transactions|requests|hrs|hours|days|weeks|months)\b",
        r"\b(improved|reduced|increased|decreased|optimized|scaled)\b",
    ]

    def has_strict_metric(text: str) -> bool:
        return any(re.search(p, text, re.IGNORECASE) for p in strict_metric_patterns)

    for item in resume_json.get("experience", [])[:5]:
        if isinstance(item, dict):
            desc = str(item.get("description", "")).strip()
            if desc and not has_strict_metric(desc):
                return True

    for item in resume_json.get("projects", [])[:5]:
        if isinstance(item, dict):
            desc = str(item.get("description", "")).strip()
            if desc and not has_strict_metric(desc):
                return True

    return False


def _merge_unique(
    base: list[dict[str, str]],
    candidates: list[dict[str, str]],
    seen_keys: set[str],
    limit: int,
) -> list[dict[str, str]]:
    """Append candidates to base, skipping duplicates, until len(base) == limit."""
    result = list(base)
    for q in candidates:
        if len(result) >= limit:
            break
        key = str(q.get("question", ""))[:60].lower().strip()
        if key and key not in seen_keys:
            seen_keys.add(key)
            result.append(q)
    return result


def generate_questions(
    resume_json: dict[str, Any],
    ats_result: dict[str, Any] | None = None,
    user_id: str | None = None,
    analysis_id: str | None = None,
    previously_asked: list[str] | None = None,
) -> dict[str, Any]:
    log_event(
        agent="A5_QA",
        user_id=user_id,
        analysis_id=analysis_id,
        event="agent_start",
        details={
            "has_ats_result": bool(ats_result),
            "experience_count": len(resume_json.get("experience", [])),
            "project_count": len(resume_json.get("projects", [])),
            "previously_asked_count": len(previously_asked or []),
        },
    )

    # Track all observed question texts for deduplication
    seen_keys: set[str] = {q[:60].lower().strip() for q in (previously_asked or [])}

    # ── BYPASS: only skip if BOTH ATS >= 90 AND no metric gaps ──
    if ats_result and isinstance(ats_result, dict):
        current_score = ats_result.get("score") or ats_result.get("final_score") or 0
        if float(current_score) >= 90 and not _resume_needs_qa(resume_json):
            log_event(
                agent="A5_QA",
                user_id=user_id,
                analysis_id=analysis_id,
                event="bypass_strong_resume",
                details={"reason": f"ATS score {current_score} >= 90 and no metric gaps — skipping Q&A"},
            )
            return {"questions": [], "bypassed": True}

    missing_keywords: list[str] = []
    if ats_result and isinstance(ats_result, dict):
        missing_keywords = [
            str(keyword) for keyword in ats_result.get("missing_keywords", [])
            if str(keyword).strip()
        ]

    # ── STEP 1: LLM — ask for exactly 5 questions ──
    questions: list[dict[str, str]] = []
    try:
        llm_payload = call_llm_json(
            prompt=(
                f"{QA_PROMPT}\n\n"
                f"Resume JSON:\n{resume_json}\n\n"
                f"ATS missing keywords:\n{missing_keywords}\n\n"
                f"Previously asked questions (DO NOT repeat these):\n{list(seen_keys)}"
            ),
            system_instruction=(
                "You are a strict JSON-only clarification question assistant. "
                "Generate exactly 5 smart, varied questions to improve this resume. "
                "Each question MUST include the exact item_name matching the experience title "
                "or project name from the resume — this is critical for mapping answers back. "
                "Focus on: missing metrics, deployment status, team context, user impact, "
                "and any missing JD keywords the candidate may genuinely have experience with. "
                "Do NOT repeat any question from the 'Previously asked questions' list. "
                "Do NOT ask leading questions that assume the candidate used a technology "
                "they have not mentioned. Never ask about facts already in the resume. "
                "For keyword questions, always allow a 'No' answer — never pressure fabrication. "
                "Return JSON: "
                '{"questions":[{"id":"q1","section":"experience","item_name":"EXACT title or project name",'
                '"question":"","why_needed":"","example_answer":"","answer_type":"metric"}]}'
            ),
            schema_hint='{"questions":[{"id":"q1","section":"experience","item_name":"","question":"","why_needed":"","example_answer":"","answer_type":"metric"}]}',
            agent="A5_QA",
            user_id=user_id,
            analysis_id=analysis_id,
        )

        llm_questions = llm_payload.get("questions", []) if isinstance(llm_payload, dict) else []

        if isinstance(llm_questions, list):
            for item in llm_questions:
                if not isinstance(item, dict):
                    continue
                key = str(item.get("question", ""))[:60].lower().strip()
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                questions.append(item)
                if len(questions) >= 5:
                    break

    except Exception:
        pass

    log_event(
        agent="A5_QA",
        user_id=user_id,
        analysis_id=analysis_id,
        event="llm_questions",
        details={"llm_questions_count": len(questions)},
    )

    # ── STEP 2: Rule-based top-up — fill remaining slots to reach exactly 5 ──
    #
    # FIX: Previously this block used `max(0, 4 - len(fallback))` as the slice
    # cap, meaning it could only ever build a fallback list of 4 — not 5. Also
    # it maintained a separate `existing_q_keys` set instead of sharing
    # `seen_keys`, causing dedup to miss cross-set collisions.
    #
    # Now: all candidates are fed through `_merge_unique` which checks the
    # single shared `seen_keys` set and stops when we hit 5.
    if len(questions) < 5:
        remaining = 5 - len(questions)
        log_event(
            agent="A5_QA",
            user_id=user_id,
            analysis_id=analysis_id,
            event="rule_based_topup",
            details={"reason": f"LLM returned {len(questions)} — filling {remaining} slots with rules"},
        )

        # Collect all rule-based candidates (more than we need; _merge_unique will stop at 5)
        exp_questions   = _collect_vague_experiences(resume_json)
        depth_questions = _collect_depth_questions(resume_json)
        proj_questions  = _collect_vague_projects(resume_json)
        kw_questions    = _collect_jd_keyword_questions(missing_keywords)

        # Priority order: experience gaps → depth probes → project gaps → keyword gaps
        questions = _merge_unique(questions, exp_questions,   seen_keys, limit=5)
        questions = _merge_unique(questions, depth_questions, seen_keys, limit=5)
        questions = _merge_unique(questions, proj_questions,  seen_keys, limit=5)
        questions = _merge_unique(questions, kw_questions,    seen_keys, limit=5)

        log_event(
            agent="A5_QA",
            user_id=user_id,
            analysis_id=analysis_id,
            event="rule_based_questions",
            details={
                "exp_questions": len(exp_questions),
                "depth_questions": len(depth_questions),
                "proj_questions": len(proj_questions),
                "kw_questions": len(kw_questions),
                "total_after_topup": len(questions),
            },
        )

    # ── STEP 3: Hard guarantee — if all rule-based paths also returned nothing,
    # inject generic fallback questions until we reach 5.
    # This only fires on truly degenerate resumes (no experience, no projects,
    # no missing keywords) that somehow passed the ATS >= 90 bypass check.
    while len(questions) < 5:
        # Rotate through remaining missing keywords so each fallback is distinct
        kw_index = len(questions)
        kw_slice = missing_keywords[kw_index:] if kw_index < len(missing_keywords) else []
        fallback_q = _build_generic_fallback_question(resume_json, kw_slice, index=kw_index)
        key = str(fallback_q.get("question", ""))[:60].lower().strip()
        if key not in seen_keys:
            seen_keys.add(key)
            questions.append(fallback_q)
        else:
            # Avoid infinite loop if somehow even the fallback text collides
            break

    if not questions:
        # Absolute last resort — should be unreachable
        questions = [_build_generic_fallback_question(resume_json, missing_keywords, index=0)]

    # Re-assign sequential IDs after all merging is done
    questions = _reindex_questions(questions)

    log_event(
        agent="A5_QA",
        user_id=user_id,
        analysis_id=analysis_id,
        event="agent_complete",
        details={"questions_count": len(questions), "bypassed": False},
    )
    return {"questions": questions, "bypassed": False}


def validate_answers(
    qa_answers: dict[str, str],
    user_id: str | None = None,
    analysis_id: str | None = None,
) -> dict[str, Any]:
    log_event(
        agent="A5_QA",
        user_id=user_id,
        analysis_id=analysis_id,
        event="agent_start",
        details={"answers_count": len(qa_answers)},
    )

    validated_answers: dict[str, str] = {}
    warnings: list[str] = []

    for question_id, answer in qa_answers.items():
        clean_answer = str(answer).strip()
        validated_answers[question_id] = clean_answer

        if not clean_answer:
            warnings.append(f"{question_id} is empty.")
            continue

        if re.search(r"\b100%\b", clean_answer) or re.search(r"\b(\d{3,}|9\d|8\d)\s?%\b", clean_answer):
            warnings.append(f"{question_id} may contain exaggerated improvement claims.")

        is_negative_answer = bool(re.search(
            r"\b(no[,.]?\s|haven'?t|have not|not yet|don'?t have|never used|no experience)\b",
            clean_answer, re.IGNORECASE,
        ))
        if not is_negative_answer:
            has_substance = (
                re.search(r"\d", clean_answer)
                or re.search(
                    r"\b(Python|Java|React|Node|FastAPI|SQL|AWS|Docker|Kubernetes|TensorFlow|PyTorch|"
                    r"Flask|Streamlit|LangChain|RAG|Hugging Face|Scikit-learn|Firebase|Supabase|"
                    r"Git|REST|NLP|ML|AI)\b",
                    clean_answer, re.IGNORECASE,
                )
                or len(clean_answer) > 30
            )
            if not has_substance:
                warnings.append(
                    f"{question_id} is too brief — add a technology name, number, or a short description."
                )

    ready_to_rewrite = len(warnings) == 0 and bool(validated_answers)

    try:
        llm_payload = call_llm_json(
            prompt=f"{QA_PROMPT}\n\nQA answers:\n{validated_answers}",
            system_instruction="You are a strict JSON-only answer validator.",
            schema_hint='{"validated_answers":{},"warnings":[],"ready_to_rewrite":true}',
            agent="A5_QA",
            user_id=user_id,
            analysis_id=analysis_id,
        )
        if isinstance(llm_payload, dict):
            warnings.extend([str(item) for item in llm_payload.get("warnings", []) if str(item).strip()])
            llm_ready = bool(llm_payload.get("ready_to_rewrite", ready_to_rewrite))
            # Never let the LLM mark ready_to_rewrite=True if there are no actual answers
            ready_to_rewrite = llm_ready and bool(validated_answers)
    except Exception:
        pass

    result = {
        "validated_answers": validated_answers,
        "warnings": list(dict.fromkeys(warnings)),
        "ready_to_rewrite": ready_to_rewrite,
    }

    log_event(
        agent="A5_QA",
        user_id=user_id,
        analysis_id=analysis_id,
        event="agent_complete",
        details={"warnings_count": len(result["warnings"]), "ready_to_rewrite": result["ready_to_rewrite"]},
    )
    return result