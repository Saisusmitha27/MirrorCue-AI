from __future__ import annotations

import re
from typing import Any


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _extract_metric_from_answer(answer: str) -> str:
    """Pull the most defensible metric phrase from a QA answer."""
    text = str(answer).strip()
    if not text:
        return ""

    # Reject negative answers — "No, I haven't used it" should never produce a metric
    if re.search(
        r"\b(no[,.]?\s|haven'?t|have not|not yet|don'?t have|never used|no experience)\b",
        text, re.IGNORECASE,
    ):
        return ""

    patterns = [
        r"\b\d+(?:\.\d+)?\s*%",
        r"\b\d[\d,]*\+?\s*(?:users|clients|transactions|hours|days|weeks|months|DAU|INR|k|K)\b",
        r"\b(?:reduced|improved|increased|decreased|optimized|scaled|saved|supported)\b[^.]{0,80}\d+[^.]{0,40}",
        r"\b\d[\d,]*\+?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip()

    if re.search(r"\d", text):
        return text[:120].strip()
    return ""


def _extract_confirmed_keywords(qa_answers: dict[str, str], missing_kws: list[str]) -> set[str]:
    """Return ONLY missing keywords the candidate explicitly confirmed in their QA answers.

    A keyword is 'confirmed' if:
    - It appears verbatim (case-insensitive) in a positive QA answer, AND
    - The answer is not a negative/denial response.

    This is the anti-fabrication gate — keywords NOT in this set must NEVER
    be added to resume bullets by the rewrite agent.
    """
    confirmed: set[str] = set()
    for key, value in qa_answers.items():
        answer = str(value).strip()
        if not answer:
            continue
        # Skip negative answers — "No, I haven't used FastAPI" doesn't confirm FastAPI
        if re.search(
            r"\b(no[,.]?\s|haven'?t|have not|not yet|don'?t have|never used|no experience)\b",
            answer, re.IGNORECASE,
        ):
            continue
        answer_lower = answer.lower()
        for kw in missing_kws:
            if kw.lower() in answer_lower:
                confirmed.add(kw)
    return confirmed


def _map_qa_answers_to_experiences(
    qa_answers: dict[str, str],
    experience_items: list[dict[str, Any]],
) -> dict[int, str]:
    """Map QA answers to specific experience items by matching item_name in answer keys.

    FIX: The old approach used sequential index mapping (q1 → exp[0], q2 → exp[1]).
    This was wrong because QA answers are about missing keywords (FastAPI, PyTorch etc.),
    not about specific internships. A keyword answer getting mapped to the Elewayte
    internship caused "40%" to be injected into every ML bullet.

    New approach: only map an answer to an experience if the answer key explicitly
    references that experience by title, OR if the answer contains a metric and no
    other experience has claimed it yet.
    """
    exp_metrics: dict[int, str] = {}

    for key, value in qa_answers.items():
        metric = _extract_metric_from_answer(str(value))
        if not metric:
            continue
        key_lower = key.lower()
        # Try to match by experience title in the answer key
        matched = False
        for idx, exp in enumerate(experience_items):
            title_lower = str(exp.get("title", "")).lower().replace(" ", "_")
            company_lower = str(exp.get("company", "")).lower().replace(" ", "_")
            if title_lower and title_lower in key_lower:
                if idx not in exp_metrics:
                    exp_metrics[idx] = metric
                matched = True
                break
            if company_lower and company_lower in key_lower:
                if idx not in exp_metrics:
                    exp_metrics[idx] = metric
                matched = True
                break
        # Do NOT fall back to sequential mapping if no name match found.
        # Better to add no metric than to add the wrong one.

    return exp_metrics


def _map_qa_answers_to_projects(
    qa_answers: dict[str, str],
    project_items: list[dict[str, Any]],
) -> dict[int, str]:
    """Map QA answers to specific projects by matching project name in answer keys."""
    proj_metrics: dict[int, str] = {}

    for key, value in qa_answers.items():
        metric = _extract_metric_from_answer(str(value))
        if not metric:
            continue
        key_lower = key.lower()
        for idx, proj in enumerate(project_items):
            name_lower = str(proj.get("name", "")).lower().replace(" ", "_").replace("-", "_")
            if name_lower and name_lower in key_lower:
                if idx not in proj_metrics:
                    proj_metrics[idx] = metric
                break
        # Same principle — no sequential fallback, no metric injection guessing

    return proj_metrics


def _bias_phrases_to_remove(bias_result: dict[str, Any]) -> list[str]:
    """Return only actual candidate-wrote phrases from resume — never recruiter explanations."""
    phrases: list[str] = []
    for flag in _as_list(bias_result.get("flags", [])):
        if not isinstance(flag, dict):
            continue
        wrote = str(flag.get("candidate_wrote", "")).strip()
        if wrote and wrote not in ("resume content", "project details", "experience details", ""):
            phrases.append(wrote)
    return phrases


def map_rewrite_stage1(
    resume_json: dict[str, Any],
    ats_result: dict[str, Any],
    qa_answers: dict[str, str],
    bias_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Rule-based stage-1 rewrite mapper.

    Maps QA metrics and confirmed ATS keywords into structured experience/project blocks.

    ANTI-FABRICATION CONTRACT (enforced here before the LLM sees anything):
    - keywords_to_add: EMPTY unless the candidate explicitly confirmed that keyword in a QA answer
    - metrics: EMPTY unless a metric was extracted from a QA answer matched to that item
    - The 'confirmed_keywords' and 'unconfirmed_keywords' fields are passed through
      so the rewrite agent can enforce the same contract at the prompt level.

    The old mapper used `missing_kws[idx % len(missing_kws)]` which cycled ALL
    missing keywords (FastAPI, PyTorch, Docker, AWS, GCP) across ALL items regardless
    of whether the candidate confirmed using any of them — the root cause of fabrication.
    """
    missing_kws = [str(k).strip() for k in _as_list(ats_result.get("missing_keywords", [])) if str(k).strip()]
    bias_phrases = _bias_phrases_to_remove(bias_result)

    # Gate: only keywords the candidate confirmed in QA answers
    confirmed_kws = _extract_confirmed_keywords(qa_answers, missing_kws)
    unconfirmed_kws = [kw for kw in missing_kws if kw not in confirmed_kws]

    exp_items = [item for item in _as_list(resume_json.get("experience", [])) if isinstance(item, dict)]
    proj_items = [item for item in _as_list(resume_json.get("projects", [])) if isinstance(item, dict)]

    # Map metrics to specific items by name-matching, not sequential index
    exp_metrics = _map_qa_answers_to_experiences(qa_answers, exp_items)
    proj_metrics = _map_qa_answers_to_projects(qa_answers, proj_items)

    experience_out: list[dict[str, Any]] = []
    for idx, exp in enumerate(exp_items):
        title = str(exp.get("title", "")).strip()
        company = str(exp.get("company", "")).strip()
        desc = str(exp.get("description", "")).strip()

        # Use name-matched metric only — no sequential fallback
        metric = exp_metrics.get(idx, "")

        # Preserve the original description; only add metric context if confirmed
        if metric:
            mapped_facts = [f"{desc} ({metric})"] if desc else [f"{title} at {company}: {metric}"]
        elif desc:
            # Keep original text — do not invent enhancements
            mapped_facts = [desc]
        else:
            mapped_facts = [f"Executed deliverables as {title} at {company}"]

        # FIX: Use experience-specific tech from description, NOT global skills[:2].
        # The old code used resume_json.get("skills", [])[:2] which added random
        # global skills (e.g. "LangChain") to an unrelated internship bullet.
        # Extract tech from the description using the existing skills list as a lookup.
        global_skills_lower = {s.lower(): s for s in _as_list(resume_json.get("skills", []))}
        exp_specific_tech = [
            global_skills_lower[skill_lower]
            for skill_lower in global_skills_lower
            if skill_lower in desc.lower()
        ][:5]  # cap at 5 to avoid overwhelming the LLM context

        experience_out.append({
            "title": title,
            "company": company,
            "mapped_facts": mapped_facts,
            "technologies": exp_specific_tech,
            "metrics": [metric] if metric else [],
            # CRITICAL: only inject a keyword if the candidate confirmed it in Q&A
            "keywords_to_add": [kw for kw in confirmed_kws if kw.lower() in desc.lower() or not desc],
            "bias_phrases_to_remove": bias_phrases,
        })

    projects_out: list[dict[str, Any]] = []
    for idx, proj in enumerate(proj_items):
        name = str(proj.get("name", "")).strip() or f"Project {idx + 1}"
        desc = str(proj.get("description", "")).strip()
        tech = [str(t) for t in _as_list(proj.get("tech", []))]

        metric = proj_metrics.get(idx, "")

        if metric:
            mapped_facts = [f"{desc} ({metric})"] if desc else [f"{name}: {metric}"]
        elif desc:
            mapped_facts = [desc]
        else:
            mapped_facts = [f"Built {name} using {', '.join(tech) if tech else 'Python'}"]

        projects_out.append({
            "name": name,
            "mapped_facts": mapped_facts,
            "technologies": tech,  # only what was actually in the project
            "metrics": [metric] if metric else [],
            # Only add a confirmed keyword if it's not already in the project's tech stack
            "keywords_to_add": [
                kw for kw in confirmed_kws
                if kw not in tech and kw.lower() not in desc.lower()
            ],
            "bias_phrases_to_remove": bias_phrases,
        })

    return {
        "experience": experience_out,
        "projects": projects_out,
        # Pass these through so the rewrite agent can enforce the contract at prompt level
        "confirmed_keywords": sorted(confirmed_kws),
        "unconfirmed_keywords": unconfirmed_kws,
    }