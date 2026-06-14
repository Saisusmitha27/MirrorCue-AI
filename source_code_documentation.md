<div align="center">

# MetaCortex

***

### Capgemini Exceller AgentifAI Buildathon 2026 &bull; Use Case 06
## MirrorCue AI
### Bias-Aware Resume Intelligence System

***

### Source Code Documentation

| Team Member | Role |
| :--- | :--- |
| **Saisusmitha P** | Team Lead & AI Architect |
| **Sharmitha Sri M** | Backend & Agent Developer |
| **Tanvi R** | Prompt Engineer |
| **Sarveshwaran C** | Frontend & UX |
| **Raghul R** | Data & Bias Research |

<br>

**Panimalar Engineering College**  
B.Tech Artificial Intelligence and Data Science &bull; Batch 2023–2027

</div>

<div style="page-break-after: always;"></div>

# Table of Contents

## AI and Machine Learning Modules

* [backend/agents/orchestrator.py](#backendagentsorchestratorpy)
* [backend/agents/ats_matcher.py](#backendagentsats_matcherpy)
* [backend/agents/bias_mirror.py](#backendagentsbias_mirrorpy)
* [backend/agents/qa_agent.py](#backendagentsqa_agentpy)
* [backend/agents/rewrite_agent.py](#backendagentsrewrite_agentpy)
* [backend/agents/parser.py](#backendagentsparserpy)
* [backend/ml/bias_classifier.py](#backendmlbias_classifierpy)
* [backend/ml/features.py](#backendmlfeaturespy)

## Backend Core Modules

* [backend/main.py](#backendmainpy)
* [backend/core/config.py](#backendcoreconfigpy)

## Database Models

* [backend/models/resume.py](#backendmodelsresumepy)
* [backend/models/analysis.py](#backendmodelsanalysispy)

## Frontend Components

* [frontend/src/components/analysis/ATSTab.tsx](#frontendsrccomponentsanalysisatstabtsx)
* [frontend/src/components/analysis/BiasMirrorTab.tsx](#frontendsrccomponentsanalysisbiasmirrortabtsx)
* [frontend/src/components/analysis/RewriteTab.tsx](#frontendsrccomponentsanalysisrewritetabtsx)

## Testing Framework

* [backend/tests/conftest.py](#backendtestsconftestpy)
* [backend/tests/test_ats_matcher.py](#backendteststest_ats_matcherpy)
* [backend/tests/test_bias_mirror.py](#backendteststest_bias_mirrorpy)
* [backend/tests/test_resume_utils.py](#backendteststest_resume_utilspy)

<div style="page-break-after: always;"></div>

<div id="backendagentsorchestratorpy"></div>

## backend/agents/orchestrator.py

```python
from __future__ import annotations

import asyncio
import traceback
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.ats_matcher import match_ats
from backend.agents.bias_mirror import detect_bias
from backend.agents.parser import parse_resume_pdf
from backend.agents.qa_agent import generate_questions, validate_answers
from backend.agents.rewrite_agent import rewrite_resume
from backend.core.logging_config import log_event
from backend.models.analysis import Analysis
from backend.models.resume import Resume


async def _update_analysis(session: AsyncSession, analysis_id: str, **fields: Any) -> None:
    await session.execute(update(Analysis).where(Analysis.id == UUID(analysis_id)).values(**fields))
    await session.commit()


async def _load_resume_path(session: AsyncSession, analysis_id: str) -> str:
    result = await session.execute(select(Analysis.resume_id).where(Analysis.id == UUID(analysis_id)))
    resume_id = result.scalar_one_or_none()
    if not resume_id:
        raise ValueError("Analysis not found")
    resume_result = await session.get(Resume, resume_id)
    if not resume_result:
        raise ValueError("Resume not found")
    return resume_result.file_path


async def error_handler_node(session: AsyncSession, analysis_id: str, user_id: str, error: str, stage: str) -> None:
    await _update_analysis(session, analysis_id, status="failed")
    log_event(
        level=40,
        agent="A1_ORCHESTRATOR",
        user_id=user_id,
        analysis_id=analysis_id,
        event="error",
        details={"error": error, "stage": stage, "traceback": traceback.format_exc()},
        exc_info=True,
    )


async def run_pipeline(
    session: AsyncSession,
    analysis_id: str,
    user_id: str,
    jd_text: str,
    qa_answers: dict[str, str] | None = None,
) -> None:
    analysis = await session.get(Analysis, UUID(analysis_id))
    if not analysis:
        raise ValueError("Analysis not found")

    resume_path = await _load_resume_path(session, analysis_id)

    resume_text = ""
    resume_json: dict[str, Any] = {}
    ats_result: dict[str, Any] = {}
    bias_result: dict[str, Any] = {}
    qa_questions: dict[str, Any] = {}
    rewrite_result: dict[str, Any] = {}

    has_existing_results = bool(
        analysis.parsed_json or analysis.ats_result or analysis.bias_result or analysis.qa_questions
    )

    log_event(
        agent="A1_ORCHESTRATOR",
        user_id=user_id,
        analysis_id=analysis_id,
        event="pipeline_entry",
        details={
            "qa_answers_provided": qa_answers is not None,
            "has_existing_results": has_existing_results,
            "current_status": analysis.status,
            "parsed_json_exists": analysis.parsed_json is not None,
            "ats_result_exists": analysis.ats_result is not None,
            "bias_result_exists": analysis.bias_result is not None,
            "qa_questions_exists": analysis.qa_questions is not None,
        },
    )

    try:
        # ── BRANCH 1: QA answers submitted — candidate has answered, now rewrite ──
        if qa_answers is not None and has_existing_results:
            if not analysis.parsed_json or not analysis.ats_result or not analysis.bias_result:
                raise ValueError(
                    "Incomplete analysis state: missing parsed_json, ats_result, or bias_result. "
                    "Cannot proceed with rewrite."
                )

            resume_json = analysis.parsed_json
            ats_result = analysis.ats_result
            bias_result = analysis.bias_result

            qa_questions = analysis.qa_questions or {}
            saved_questions: list[dict[str, Any]] = qa_questions.get("questions", [])

            log_event(
                agent="A1_ORCHESTRATOR",
                user_id=user_id,
                analysis_id=analysis_id,
                event="rewrite_branch_start",
                details={
                    "qa_answer_count": len(qa_answers),
                    "qa_questions_count": len(saved_questions),
                },
            )

            await _update_analysis(session, analysis_id, status="qa_validate")
            log_event(agent="A1_ORCHESTRATOR", user_id=user_id, analysis_id=analysis_id, event="status_set", details={"status": "qa_validate"})

            validation = await asyncio.to_thread(
                validate_answers,
                qa_answers,
                user_id,
                analysis_id,
            )

            validated = validation.get("validated_answers", qa_answers)
            warnings = validation.get("warnings", [])
            ready = validation.get("ready_to_rewrite", True)

            log_event(
                agent="A1_ORCHESTRATOR",
                user_id=user_id,
                analysis_id=analysis_id,
                event="qa_validated",
                details={"warnings": warnings, "ready": ready},
            )

            # ── GATE: block rewrite if answers failed validation ──
            if not ready:
                await _update_analysis(
                    session,
                    analysis_id,
                    qa_answers=validated,
                    status="qa_pending",  # send back to pending so frontend re-prompts
                )
                log_event(
                    agent="A1_ORCHESTRATOR",
                    user_id=user_id,
                    analysis_id=analysis_id,
                    event="qa_gate_blocked",
                    details={"reason": "Answers failed validation", "warnings": warnings},
                )
                return
            # ── END GATE ──

            await _update_analysis(
                session,
                analysis_id,
                qa_answers=validated,
                status="qa_validated",
            )

            await _update_analysis(session, analysis_id, status="rewrite")
            log_event(agent="A1_ORCHESTRATOR", user_id=user_id, analysis_id=analysis_id, event="status_set", details={"status": "rewrite"})

            rewrite_result = await asyncio.to_thread(
                rewrite_resume,
                resume_json,
                jd_text,
                ats_result or {},
                bias_result or {},
                validated,
                saved_questions,
                user_id,
                analysis_id,
            )

            log_event(
                agent="A1_ORCHESTRATOR",
                user_id=user_id,
                analysis_id=analysis_id,
                event="rewrite_complete",
                details={
                    "has_rewritten_experience": bool(rewrite_result.get("rewritten_experience")),
                    "has_rewritten_projects": bool(rewrite_result.get("rewritten_projects")),
                    "ats_score_before": rewrite_result.get("ats_score_before", 0),
                    "ats_score_after": rewrite_result.get("ats_score_after", 0),
                    "ats_score_delta": rewrite_result.get("ats_score_delta", 0),
                    "used_fallback": rewrite_result.get("used_fallback", False),
                },
            )

            await _update_analysis(session, analysis_id, rewrite_result=rewrite_result, status="complete")

            updated = await session.get(Analysis, UUID(analysis_id))
            if updated:
                log_event(
                    agent="A1_ORCHESTRATOR",
                    user_id=user_id,
                    analysis_id=analysis_id,
                    event="rewrite_saved_verification",
                    details={
                        "status": updated.status,
                        "has_rewrite_result": updated.rewrite_result is not None,
                        "rewrite_result_keys": list(updated.rewrite_result.keys()) if updated.rewrite_result else [],
                    },
                )
            return

        # ── BRANCH 2: Fresh analysis — parse, ATS, bias, then generate Q&A and STOP ──
        await _update_analysis(session, analysis_id, status="parse")
        parsed = await asyncio.to_thread(parse_resume_pdf, resume_path, user_id, analysis_id)
        resume_text = parsed["resume_text"]
        resume_json = parsed["resume_json"]
        await _update_analysis(session, analysis_id, parsed_json=resume_json, status="parse_complete")

        await _update_analysis(session, analysis_id, status="ats_match")
        ats_result = await asyncio.to_thread(
            match_ats,
            resume_json,
            resume_text,
            jd_text,
            user_id,
            analysis_id,
        )
        await _update_analysis(session, analysis_id, ats_result=ats_result, status="ats_complete")

        other_analyses = []
        try:
            stmt = select(Analysis).where(
                Analysis.user_id == UUID(user_id),
                Analysis.status == "complete",
                Analysis.id != UUID(analysis_id)
            )
            db_res = await session.execute(stmt)
            all_other = db_res.scalars().all()
            for other in all_other:
                if other.jd_text and other.jd_text.strip().lower() == jd_text.strip().lower():
                    other_analyses.append(other)
        except Exception as query_exc:
            log_event(
                level=30,
                agent="A1_ORCHESTRATOR",
                user_id=user_id,
                analysis_id=analysis_id,
                event="other_analyses_query_warning",
                details={"error": str(query_exc)}
            )

        await _update_analysis(session, analysis_id, status="bias_detect")
        bias_result = await asyncio.to_thread(
            detect_bias,
            resume_json,
            resume_text,
            user_id,
            analysis_id,
            jd_text=jd_text,
            ats_score=ats_result.get("score"),
            other_analyses=other_analyses,
        )
        await _update_analysis(session, analysis_id, bias_result=bias_result, status="bias_complete")

        # ── Generate Q&A questions and ALWAYS stop here for candidate input ──
        await _update_analysis(session, analysis_id, status="qa_generate")
        qa_questions = await asyncio.to_thread(
            generate_questions,
            resume_json,
            ats_result or {},
            user_id,
            analysis_id,
        )
        await _update_analysis(session, analysis_id, qa_questions=qa_questions, status="qa_pending")

        questions_list = qa_questions.get("questions", [])

        # ── FIX (NEW): branch on the explicit "bypassed" flag from
        # generate_questions instead of bare list-emptiness.
        #
        # Previously, ANY empty questions_list (including ones caused by an
        # LLM failure combined with edge-case rule fallbacks returning zero
        # items — e.g. no missing keywords + every bullet already had a
        # metric) was treated as "resume is genuinely strong, skip Q&A".
        # That assumption was never actually verified against the ATS score,
        # so a 77% resume could be silently pushed straight into rewrite
        # with zero candidate input — which is exactly what was happening.
        #
        # Now:
        #   - bypassed=True  -> generate_questions itself verified ATS >= 90
        #                        and no metric gaps. Safe to rewrite directly.
        #   - bypassed=False -> questions_list is now GUARANTEED non-empty
        #                        (generate_questions always supplies at least
        #                        one fallback question). Gate and wait.
        #   - bypassed=False and questions_list empty -> should not happen;
        #                        treat as an error and stay at qa_pending
        #                        rather than risk degrading the resume.
        bypassed = bool(qa_questions.get("bypassed", False))

        if bypassed:
            log_event(
                agent="A1_ORCHESTRATOR",
                user_id=user_id,
                analysis_id=analysis_id,
                event="qa_bypass",
                details={
                    "reason": (
                        "generate_questions reported bypassed=True "
                        "(ATS >= 90 and no metric gaps) — proceeding to rewrite"
                    )
                },
            )
            await _update_analysis(session, analysis_id, status="rewrite")

            rewrite_result = await asyncio.to_thread(
                rewrite_resume,
                resume_json,
                jd_text,
                ats_result or {},
                bias_result or {},
                {},           # no qa_answers — resume was already strong
                [],           # no qa_questions
                user_id,
                analysis_id,
            )
            await _update_analysis(session, analysis_id, rewrite_result=rewrite_result, status="complete")

            updated = await session.get(Analysis, UUID(analysis_id))
            if updated:
                log_event(
                    agent="A1_ORCHESTRATOR",
                    user_id=user_id,
                    analysis_id=analysis_id,
                    event="rewrite_saved_verification",
                    details={
                        "status": updated.status,
                        "has_rewrite_result": updated.rewrite_result is not None,
                        "rewrite_result_keys": list(updated.rewrite_result.keys()) if updated.rewrite_result else [],
                    },
                )

        elif not questions_list:
            # Guard against empty questions list
            log_event(
                level=40,
                agent="A1_ORCHESTRATOR",
                user_id=user_id,
                analysis_id=analysis_id,
                event="qa_questions_empty_not_bypassed",
                details={
                    "reason": (
                        "generate_questions returned no questions and bypassed=False. "
                        "This should not happen. Staying at qa_pending instead of "
                        "auto-rewriting to avoid degrading the resume."
                    )
                },
            )
            # Pipeline pauses at qa_pending state

        else:
            # ── GATE: questions exist — stop here, wait for candidate answers ──
            # The frontend reads status="qa_pending" and shows questions to the candidate.
            # The pipeline resumes only when the candidate submits answers,
            # which triggers Branch 1 above.
            log_event(
                agent="A1_ORCHESTRATOR",
                user_id=user_id,
                analysis_id=analysis_id,
                event="qa_gate_hold",
                details={
                    "reason": "Questions generated — waiting for candidate answers before rewrite",
                    "questions_count": len(questions_list),
                },
            )
            # Wait for candidate answers before resuming rewrite

    except Exception as exc:
        await error_handler_node(session, analysis_id, user_id, str(exc), "pipeline")
```

<div style="page-break-after: always;"></div>

<div id="backendagentsats_matcherpy"></div>

## backend/agents/ats_matcher.py

```python
from __future__ import annotations

import os
import datetime

# Suppress TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import re
import json
import time
from typing import Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sentence_transformers import SentenceTransformer
from rapidfuzz import process, fuzz

from backend.core.logging_config import log_event
from backend.utils.llm_client import call_llm_json
from backend.utils.resume_utils import build_experience_section_text, build_projects_section_text, extract_experience_description

# Cache SentenceTransformer embedder instance
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

ATS_SCHEMA_HINT = """{
  "keywords": ["string"]
}"""

RECOMMENDATION_SCHEMA_HINT = """{
  "recommendation": "string",
  "related_recommended_keywords": [{"keyword": "string", "reason": "string"}],
  "additional_resume_strengths": [{"item": "string", "category": "certification|tool|achievement|skill"}]
}"""

# Stopwords for LLM fallback filtering

LLM_FALLBACK_STOPWORDS: set[str] = {
    "the", "and", "for", "with", "this", "that", "are", "from", "have", "been",
    "will", "can", "your", "our", "you", "its", "their", "which", "should",
    "must", "any", "all", "was", "not", "but", "has", "had", "they", "them",
    "also", "more", "such", "each", "into", "than", "then", "when", "who",
    "what", "how", "use", "work", "team", "role", "able", "new", "good",
    "well", "both", "very", "some", "most", "other", "like", "may", "need",
}

# Map JD keywords to candidate synonyms

CONCEPT_SYNONYMS: dict[str, list[str]] = {
    "nosql": ["firebase", "supabase", "mongodb", "dynamodb", "firestore", "couchdb", "cassandra"],
    "vector databases": ["faiss", "chroma", "weaviate", "pinecone", "qdrant", "milvus",
                         "vector store", "embeddings store", "rag", "retrieval augmented"],
    "text classification": ["intent recognition", "sentiment", "named entity", "text categorization",
                             "nlp classification", "nlp"],
    "semantic search": ["embeddings", "cosine similarity", "rag", "retrieval", "vector search",
                        "similarity search"],
    "docker": ["containerize", "containerized", "containerization", "containerizing",
               "container", "dockerfile", "docker-compose"],
    "cloud deployment": ["aws", "gcp", "azure", "heroku", "render", "vercel", "cloud"],
    "deep learning": ["neural network", "tensorflow", "pytorch", "keras", "cnn", "lstm", "transformer"],
    "fastapi": ["fast api", "api framework", "async api", "uvicorn"],
    "pytorch": ["torch", "autograd", "torchvision"],
    "aws": ["amazon web services", "ec2", "s3", "lambda", "sagemaker"],
    "gcp": ["google cloud", "bigquery", "vertex ai", "cloud run"],
    "mlops": ["model deployment", "model serving", "mlflow", "kubeflow", "bentoml", "airflow"],
    "ci/cd": ["github actions", "jenkins", "gitlab ci", "circleci", "devops pipeline"],
    "data engineering": ["etl", "pipeline", "spark", "airflow", "dbt", "kafka"],
    "computer vision": ["opencv", "yolo", "image classification", "object detection", "cnn"],
    "microservices": ["docker", "kubernetes", "k8s", "api gateway"],
    "agile": ["scrum", "sprint", "kanban", "jira", "standup"],
    "sql": ["mysql", "postgresql", "sqlite", "database query", "relational database"],
    "generative ai": ["genai", "gen ai", "generative artificial intelligence",
                      "llm application", "llm-based", "large language model"],
    "natural language processing": ["nlp", "natural language", "text processing",
                                     "language model", "computational linguistics"],
    "speech recognition": ["stt", "speech to text", "whisper", "voice recognition",
                            "audio transcription"],
    "llm deployment": ["ollama", "model serving", "llm serving", "model deployment",
                       "llm inference", "vllm"],
    "backend development": ["flask", "fastapi", "node.js", "express", "django",
                             "rest api", "api development"],
    "embedded systems": ["arduino", "raspberry pi", "rtos", "microcontroller", "stm32",
                          "esp32", "firmware", "embedded c", "bare metal"],
    "signal processing": ["dsp", "fft", "filter design", "matlab", "simulink"],
    "version control": ["git", "github", "gitlab", "bitbucket", "svn", "source control"],
    "testing": ["unit test", "pytest", "jest", "selenium", "mocha", "cypress", "tdd",
                "test driven", "integration test", "test automation"],
    "linux": ["ubuntu", "bash", "shell script", "unix", "debian", "linux command", "cli"],
    "rest api": ["restful", "rest endpoints", "http api", "api integration", "requests library",
                 "postman", "swagger", "openapi"],
    "object oriented programming": ["oop", "oops", "encapsulation", "inheritance",
                                     "polymorphism", "design patterns", "solid principles"],
    "data structures": ["algorithms", "dsa", "leetcode", "competitive programming",
                        "binary tree", "linked list", "graph", "sorting", "hashing"],
    "networking": ["tcp/ip", "http", "socket programming", "network protocols", "wireshark",
                   "dns", "network layer"],
    "android": ["kotlin", "android studio", "java android", "mobile app", "apk", "jetpack"],
    "web development": ["html", "css", "javascript", "react", "angular", "vue", "frontend",
                        "backend", "full stack", "django", "flask", "node.js"],
    "database design": ["schema design", "normalization", "erd", "entity relationship",
                        "relational model", "stored procedure", "indexing"],
}

# Check alternative keys for professional summary

SUMMARY_KEYS: list[str] = ["summary", "professional_summary", "objective", "profile", "about"]

# Thresholds for semantic section matching
SEMANTIC_THRESHOLDS: dict[str, float] = {
    "skills": 0.60,
    "experience": 0.52,
    "projects": 0.52,
    "education": 0.65,
    "summary": 0.55,
}

# Sections scanned for keyword matching
MATCH_SECTIONS: list[str] = ["skills", "experience", "projects", "education", "summary"]

# Sections displayed in UI breakdown
UI_SECTIONS: list[str] = ["skills", "experience", "projects", "education"]


SECTION_LABELS: dict[str, str] = {
    "skills": "skills",
    "experience": "experience",
    "projects": "projects",
    "education": "education",
    "summary": "professional summary",
}


def _call_llm_with_retry(retries: int = 2, delay: float = 1.5, **kwargs: Any) -> dict[str, Any]:
    for attempt in range(retries):
        try:
            return call_llm_json(**kwargs)
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _calculate_formatting_flags(resume_json: dict[str, Any], resume_text: str) -> list[str]:
    flags = []

    words = resume_text.split()
    if len(words) < 300:
        flags.append("Resume text is under 300 words, which may lack detail for ATS scanners.")

    email_present = bool(resume_json.get("email") or "@" in resume_text)
    phone_present = bool(resume_json.get("phone") or re.search(r"\+?\d[\d -]{6,15}\d", resume_text))
    if not email_present or not phone_present:
        issues = []
        if not email_present:
            issues.append("email")
        if not phone_present:
            issues.append("phone number")
        flags.append(f"Missing contact information: {' and '.join(issues)} contact information should be in the header.")

    if not resume_json.get("projects") or not isinstance(resume_json.get("projects"), list):
        flags.append("Missing projects section; including personal or academic projects demonstrates practical experience.")

    if not resume_json.get("skills") or not isinstance(resume_json.get("skills"), list):
        flags.append("Missing skills section; a dedicated skills block is crucial for ATS keyword matching.")

    experience = resume_json.get("experience", [])
    if experience and isinstance(experience, list):
        all_short = True
        has_bullets = False
        for exp in experience:
            if not isinstance(exp, dict):
                continue
            desc = extract_experience_description(exp)
            bullets = [b.strip() for b in re.split(r"[\n•*-]", desc) if b.strip()]
            if bullets:
                has_bullets = True
                for b in bullets:
                    if len(b) > 90:
                        all_short = False
                        break
            if not all_short:
                break
        if has_bullets and all_short:
            flags.append("All experience bullet points are one line or less; expand on details and context.")

    if experience and isinstance(experience, list):
        has_numbers = False
        for exp in experience:
            if not isinstance(exp, dict):
                continue
            desc = extract_experience_description(exp)
            if re.search(r"\b\d+%?\b", desc):
                has_numbers = True
                break
        if not has_numbers:
            flags.append("No quantified achievements detected; add metrics, numbers, or percentages to describe your impact.")

    return flags


def fuzzy_match_keyword(keyword: str, section_text: str, threshold: int = 85) -> tuple[bool, float]:
    if not keyword or not section_text:
        return False, 0.0

    kw_lower = keyword.lower()
    sec_lower = section_text.lower()

    # Avoid partial matching on short keywords
    if len(kw_lower) < 4:
        pattern = r"\b" + re.escape(kw_lower) + r"\b"
        matched = bool(re.search(pattern, sec_lower))
        return matched, 100.0 if matched else 0.0

    partial_score = float(fuzz.partial_ratio(kw_lower, sec_lower))

    words = [w for w in re.findall(r"[a-zA-Z0-9+#.\-/]+", sec_lower) if w]
    kw_words = [w for w in re.findall(r"[a-zA-Z0-9+#.\-/]+", kw_lower) if w]
    n = len(kw_words)

    highest_ngram_score = 0.0
    if n > 0 and len(words) >= n:
        for size in [n, n + 1]:
            if len(words) < size:
                continue
            for i in range(len(words) - size + 1):
                ngram = " ".join(words[i:i + size])
                score = float(fuzz.token_sort_ratio(ngram, kw_lower))
                if score > highest_ngram_score:
                    highest_ngram_score = score

    max_score = max(partial_score, highest_ngram_score)
    return (max_score >= threshold), max_score


def synonym_match(keyword: str, section_text: str) -> bool:
    """Word-boundary synonym check — prevents false positives inside unrelated words."""
    kw_lower = keyword.lower()
    sec_lower = section_text.lower()
    synonyms = CONCEPT_SYNONYMS.get(kw_lower, [])
    for syn in synonyms:
        pattern = r"\b" + re.escape(syn) + r"\b"
        if re.search(pattern, sec_lower):
            return True
    return False


def semantic_keyword_match(keyword: str, section_text: str, threshold: float = 0.55) -> bool:
    """Sentence-level embedding match — avoids dilution from long mixed-content sections."""
    if not keyword or not section_text:
        return False

    sentences = [s.strip() for s in re.split(r"[.\n]", section_text) if len(s.strip()) > 20]
    chunks = sentences[:20] if sentences else [section_text]

    emb_kw = _embedder.encode([keyword], normalize_embeddings=True)[0]
    emb_chunks = _embedder.encode(chunks, normalize_embeddings=True)
    sims = sklearn_cosine_similarity(emb_chunks, emb_kw.reshape(1, -1))
    return float(np.max(sims)) >= threshold


def encode_section_chunked(section_text: str, chunk_size: int = 3) -> np.ndarray:
    """Sentence-group chunks — best-match chunk wins, mixed-domain sections don't dilute."""
    sentences = [s.strip() for s in re.split(r"[.\n]", section_text) if len(s.strip()) > 20]
    if not sentences:
        return _embedder.encode([section_text], normalize_embeddings=True)
    chunks = [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    return _embedder.encode(chunks, normalize_embeddings=True)


def match_ats(
    resume_json: dict[str, Any],
    resume_text: str,
    jd_text: str,
    user_id: str | None = None,
    analysis_id: str | None = None,
) -> dict[str, Any]:
    log_event(
        agent="A3_ATS",
        user_id=user_id,
        analysis_id=analysis_id,
        event="agent_start",
        details={"resume_chars": len(resume_text), "jd_chars": len(jd_text)},
    )

    # Extract job description keywords via LLM
    try:
        jd_keywords_payload = _call_llm_with_retry(
            prompt=f"Job Description:\n{jd_text}",
            system_instruction=(
                "You are the JD Keyword Extraction Agent. Extract a flat list of important keywords, tools, skills, "
                "technologies, and certifications from the Job Description.\n"
                "CRITICAL: Normalize ALL verbose forms to their short canonical industry term:\n"
                "  'Natural Language Processing' → 'NLP'\n"
                "  'Backend Development' or 'server-side' → use the specific tool instead (Flask, FastAPI, Node.js)\n"
                "  'Speech Recognition' or 'Speech-to-Text' → 'STT' or the specific tool (Whisper)\n"
                "  'storing and retrieving embeddings' → 'vector databases'\n"
                "  'building REST endpoints' → 'REST APIs'\n"
                "  'generative AI' or 'GenAI' → 'Generative AI'\n"
                "  'Large Language Models' → 'LLMs'\n"
                "Normalize verbose descriptions but KEEP specific tool names as separate keywords.\n"
                "Example: 'SQL and NoSQL databases (SQLite, Firebase, Supabase)' → "
                "extract ALL of: ['SQL', 'NoSQL', 'SQLite', 'Firebase', 'Supabase']. "
                "Do NOT collapse specific tools into their category name only.\n"
                "Never extract generic role descriptions (e.g. 'Backend Development', 'Full-stack Development', "
                "'LLM Deployment') as standalone keywords — extract the specific tools/technologies instead.\n"
                "Return ONLY short canonical terms, not phrases.\n"
                '{"keywords": ["Python", "FastAPI", "Docker", "AWS", "NLP"]}'
            ),
            schema_hint=ATS_SCHEMA_HINT,
            temperature=0,
            agent="A3_ATS",
            user_id=user_id,
            analysis_id=analysis_id,
        )
        jd_keywords = jd_keywords_payload.get("keywords", [])
        if not isinstance(jd_keywords, list):
            jd_keywords = []
    except Exception as exc:
        log_event(
            level=30,
            agent="A3_ATS",
            user_id=user_id,
            analysis_id=analysis_id,
            event="jd_keyword_extraction_fallback",
            details={"error": str(exc)},
        )
        jd_keywords = list(set(
            w for w in re.findall(r"[A-Za-z][A-Za-z0-9+#.\-/]{2,}", jd_text.lower())
            if w not in LLM_FALLBACK_STOPWORDS
        ))

    
    raw_skills = resume_json.get("skills", [])
    skills_text = " ".join(raw_skills)
    if skills_text:
        # Wrap lists in sentences for better embeddings
        skills_text = f"Technical skills include: {skills_text}."

    experience_text = build_experience_section_text(resume_json.get("experience", []))
    projects_text = build_projects_section_text(resume_json.get("projects", []))

    edu_text_parts = []
    for k in ["college", "branch", "cgpa", "graduation_year", "degree", "education"]:
        val = resume_json.get(k)
        if val:
            edu_text_parts.append(
                " ".join(str(v) for v in val) if isinstance(val, list) else str(val)
            )
    education_text = " ".join(edu_text_parts)

    # Professional summary — included in MATCH_SECTIONS so keywords only in the
    summary_text = ""
    for k in SUMMARY_KEYS:
        val = resume_json.get(k)
        if val:
            summary_text = str(val)
            break

    section_texts: dict[str, str] = {
        "skills": skills_text,
        "experience": experience_text,
        "projects": projects_text,
        "education": education_text,
        "summary": summary_text,
    }

    log_event(
        agent="A3_ATS",
        user_id=user_id,
        analysis_id=analysis_id,
        event="section_text_preview",
        details={
            "skills_text": section_texts["skills"][:200],
            "experience_text": section_texts["experience"][:500],
            "projects_text": section_texts["projects"][:500],
            "summary_text": section_texts["summary"][:300],
            "experience_chars": len(section_texts["experience"]),
            "projects_chars": len(section_texts["projects"]),
            "experience_entry_count": len(resume_json.get("experience", [])),
        },
    )

    log_event(
        agent="A3_ATS",
        user_id=user_id,
        analysis_id=analysis_id,
        event="section_text_lengths",
        details={
            "skills_len": len(section_texts["skills"]),
            "experience_len": len(section_texts["experience"]),
            "projects_len": len(section_texts["projects"]),
            "education_len": len(section_texts["education"]),
            "summary_len": len(section_texts["summary"]),
            "experience_sample": section_texts["experience"][:300],
        },
    )

    section_is_empty = {
        sec: not bool(re.findall(r"[a-zA-Z0-9+#.\-/]+", section_texts[sec]))
        for sec in section_texts
    }

    # Keyword importance indicators for UI
    kw_importances: dict[str, str] = {}
    for kw in jd_keywords:
        kw_lower = kw.lower()
        jd_lower = jd_text.lower()
        freq = len(re.findall(r"\b" + re.escape(kw_lower) + r"\b", jd_lower))
        in_title = kw_lower in jd_lower[:150]
        if freq >= 3 or in_title:
            kw_importances[kw] = "High"
        elif freq == 2:
            kw_importances[kw] = "Medium"
        else:
            kw_importances[kw] = "Low"

    weights = {"skills": 1.5, "experience": 1.0, "projects": 1.0, "education": 0.5}

    section_matched: dict[str, list[str]] = {sec: [] for sec in MATCH_SECTIONS}
    globally_matched: set[str] = set()
    match_layer: dict[str, dict[str, str]] = {sec: {} for sec in MATCH_SECTIONS}

    for kw in jd_keywords:
        matched_anywhere = False
        for sec in MATCH_SECTIONS:
            # Match via surface-level fuzzy comparison
            is_match, _ = fuzzy_match_keyword(kw, section_texts[sec], threshold=85)
            layer = "fuzzy"

            # Match via synonym concept expansion
            if not is_match:
                is_match = synonym_match(kw, section_texts[sec])
                layer = "synonym"

            # Match via semantic embedding similarity
            if not is_match:
                is_match = semantic_keyword_match(
                    kw,
                    section_texts[sec],
                    threshold=SEMANTIC_THRESHOLDS[sec],
                )
                layer = "semantic"

            if is_match:
                if kw not in section_matched[sec]:
                    section_matched[sec].append(kw)
                    match_layer[sec][kw] = layer
                matched_anywhere = True

        if matched_anywhere:
            globally_matched.add(kw)

    missing_keywords = [kw for kw in jd_keywords if kw not in globally_matched]
    total_kw = len(jd_keywords) or 1

    section_coverage = {
        sec: round(len(section_matched[sec]) / total_kw * 100.0, 2)
        for sec in UI_SECTIONS
    }

    keyword_coverage_score = round(len(globally_matched) / total_kw * 100.0, 2)

    # Calculate semantic similarity per section
    emb_jd = _embedder.encode([jd_text], normalize_embeddings=True)[0]
    semantic_weights = {"skills": 1.5, "experience": 1.0, "projects": 1.0, "education": 0.2}
    semantic_denom = 0.0
    semantic_weighted_sum = 0.0
    section_semantic: dict[str, float] = {}

    for sec_name in UI_SECTIONS:
        if section_is_empty[sec_name]:
            section_semantic[sec_name] = 0.0
        else:
            chunk_embeddings = encode_section_chunked(section_texts[sec_name])
            sims = sklearn_cosine_similarity(chunk_embeddings, emb_jd.reshape(1, -1))

            # Calculate semantic metrics for experience/projects
            if sec_name in ("projects", "experience"):
                n_top = 1
            else:
                n_top = max(1, min(2, len(sims)))

            top_sims = np.sort(sims.flatten())[::-1][:n_top]
            cos_sim = float(np.mean(top_sims))

            if cos_sim < 0.15:
                section_semantic[sec_name] = 0.0
            else:
                score_val = round(max(0.0, min(1.0, cos_sim)) * 100.0, 2)
                section_semantic[sec_name] = score_val
                semantic_denom += semantic_weights[sec_name]
                semantic_weighted_sum += score_val * semantic_weights[sec_name]

    semantic_score = round(semantic_weighted_sum / semantic_denom, 2) if semantic_denom > 0 else 0.0

    final_score = round((0.6 * keyword_coverage_score) + (0.4 * semantic_score), 2)

    matched_keywords = sorted(list(globally_matched))

    matched_keywords_detail = [
        {
            "keyword": kw,
            "match_reason": (
                f"Matched in the "
                f"{', '.join(SECTION_LABELS[sec] for sec in MATCH_SECTIONS if kw in section_matched[sec])} "
                f"section(s)."
            ),
        }
        for kw in matched_keywords
    ]

    missing_keywords_detail = [
        {"keyword": kw, "importance": kw_importances.get(kw, "Low")}
        for kw in missing_keywords
    ]

    missing_freqs = sorted(
        [(kw, len(re.findall(r"\b" + re.escape(kw.lower()) + r"\b", jd_text.lower()))) for kw in missing_keywords],
        key=lambda x: x[1],
        reverse=True,
    )
    top_missing_keywords = [item[0] for item in missing_freqs[:5]]

    formatting_flags = _calculate_formatting_flags(resume_json, resume_text)

    # Build achievements context for LLM recommendations

    
    certifications = _as_list(resume_json.get("certifications", []))

    # Extract achievements from raw resume text
    _ach_match = re.search(
        r"(?:KEY ACHIEVEMENTS?|ACHIEVEMENTS?|AWARDS?|HONORS?|RECOGNITIONS?)[:\s]*\n(.*?)(?=\n[A-Z][A-Z\s]{3,}[:\n]|\Z)",
        resume_text,
        re.DOTALL | re.IGNORECASE,
    )
    raw_achievements: list[str] = []
    if _ach_match:
        raw_achievements = [
            line.strip("•*-– \t").strip()
            for line in _ach_match.group(1).splitlines()
            if line.strip("•*-– \t").strip()
        ][:5]  # cap at 5 to avoid noise

    
    additional_strengths_context = ""
    if certifications:
        additional_strengths_context += (
            f"\nCertifications on resume: {'; '.join(str(c) for c in certifications)}"
        )
    if raw_achievements:
        additional_strengths_context += (
            f"\nKey achievements on resume: {'; '.join(raw_achievements)}"
        )

    try:
        rec_payload = _call_llm_with_retry(
            prompt=(
                f"Job Description:\n{jd_text}\n\n"
                f"Matched Keywords (already covered — do NOT list these as additional strengths):\n"
                f"{', '.join(matched_keywords)}\n\n"
                f"Missing Keywords:\n{', '.join(missing_keywords)}\n\n"
                f"Parsed Resume:\n{json.dumps(resume_json, indent=2)}\n\n"
                f"Resume Certifications and Achievements "
                f"(mine these for additional_resume_strengths — only include genuinely impressive items "
                f"that are NOT already in Matched Keywords):"
                f"{additional_strengths_context}"
            ),
            system_instruction=(
                "You are MirrorCue AI's expert recruiter.\n"
                "Analyze the candidate's keyword matches/gaps and parsed resume, and generate feedback fields.\n"
                "CRITICAL: additional_resume_strengths must ONLY contain items from the resume that are "
                "NOT present in the Matched Keywords list and NOT required by the JD. "
                "Never list a matched keyword as an additional strength.\n"
                "Always include standout certifications and notable hackathon/competition achievements "
                "from the Resume Certifications and Achievements section in additional_resume_strengths, "
                "as long as they are not already matched keywords.\n"
                "Return ONLY a JSON matching this exact schema:\n"
                "{\n"
                '  "recommendation": "single personalized actionable paragraph telling the candidate exactly '
                'how to close the gap between their current match score and a strong match, referencing their '
                'actual missing keywords and their strongest matched sections.",\n'
                '  "related_recommended_keywords": [{"keyword": "string", "reason": "string"}],\n'
                '  "additional_resume_strengths": [{"item": "string", "category": "certification|tool|achievement|skill"}]\n'
                "}"
            ),
            schema_hint=RECOMMENDATION_SCHEMA_HINT,
            agent="A3_ATS",
            user_id=user_id,
            analysis_id=analysis_id,
        )
        recommendation = rec_payload.get("recommendation", "")
        related_recommended_keywords = _as_list(rec_payload.get("related_recommended_keywords", []))
        additional_resume_strengths = _as_list(rec_payload.get("additional_resume_strengths", []))[:3]
    except Exception as exc:
        log_event(
            level=40,
            agent="A3_ATS",
            user_id=user_id,
            analysis_id=analysis_id,
            event="llm_recommendations_error",
            details={"error": str(exc)},
            exc_info=True,
        )
        recommendation = "Surface matched strengths early, add missing JD keywords naturally, and include 4-5 role-relevant market keywords."
        related_recommended_keywords = []
        additional_resume_strengths = []

    # Ensure achievements are preserved in recommendations
    _existing_items = {s.get("item", "").lower() for s in additional_resume_strengths}
    _matched_lower = {kw.lower() for kw in matched_keywords}

    for cert in certifications:
        if len(additional_resume_strengths) >= 6:
            break
        cert_str = str(cert).strip()
        if cert_str and cert_str.lower() not in _existing_items:
            additional_resume_strengths.append({"item": cert_str, "category": "certification"})
            _existing_items.add(cert_str.lower())

    for ach in raw_achievements:
        if len(additional_resume_strengths) >= 6:
            break
        if (
            ach
            and ach.lower() not in _existing_items
            and not any(kw in ach.lower() for kw in _matched_lower)
        ):
            additional_resume_strengths.append({"item": ach, "category": "achievement"})
            _existing_items.add(ach.lower())

    section_breakdown = {
        sec: {
            "coverage_percent": section_coverage[sec],
            "semantic_similarity": section_semantic[sec],
            "semantic_score": section_semantic[sec],
            "weight": weights[sec],
            "matched_keywords": section_matched[sec],
            "missing_keywords": [kw for kw in jd_keywords if kw not in section_matched[sec]],
        }
        for sec in UI_SECTIONS
    }

    result_payload = {
        "final_score": final_score,
        "score": final_score,
        "semantic_score": semantic_score,
        "keyword_coverage_score": keyword_coverage_score,
        "keyword_score": keyword_coverage_score,
        "section_breakdown": section_breakdown,
        "matched_keywords": matched_keywords,
        "matched_keywords_detail": matched_keywords_detail,
        "missing_keywords": missing_keywords,
        "missing_keywords_detail": missing_keywords_detail,
        "top_missing_keywords": top_missing_keywords,
        "recommendation": recommendation,
        "related_recommended_keywords": related_recommended_keywords,
        "additional_resume_strengths": additional_resume_strengths,
        "formatting_flags": formatting_flags,
    }

    return result_payload
```

<div style="page-break-after: always;"></div>

<div id="backendagentsbias_mirrorpy"></div>

## backend/agents/bias_mirror.py

```python
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

```

<div style="page-break-after: always;"></div>

<div id="backendagentsqa_agentpy"></div>

## backend/agents/qa_agent.py

```python
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
```

<div style="page-break-after: always;"></div>

<div id="backendagentsrewrite_agentpy"></div>

## backend/agents/rewrite_agent.py

```python
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
```

<div style="page-break-after: always;"></div>

<div id="backendagentsparserpy"></div>

## backend/agents/parser.py

```python
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

```

<div style="page-break-after: always;"></div>

<div id="backendmlbias_classifierpy"></div>

## backend/ml/bias_classifier.py

```python
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from backend.core.config import settings
from backend.ml.features import BIAS_LABEL_KEYS, FEATURE_COLUMNS, extract_features

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATH = Path(settings.bias_ml_model_path)
if not DEFAULT_MODEL_PATH.is_absolute():
    DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / DEFAULT_MODEL_PATH
METADATA_PATH = MODEL_DIR / "bias_classifier_meta.json"

_SEVERITY_BY_LABEL: dict[str, str] = {
    "prestige_gap": "medium",
    "degree_branch_bias": "high",
    "cgpa_penalty": "low",
    "career_gap": "high",
    "tier2_location": "medium",
    "name_origin": "medium",
    "project_credibility": "medium",
    "gender_coded_language": "medium",
}

_EVIDENCE_TEMPLATES: dict[str, str] = {
    "prestige_gap": "Recruiter may deprioritize candidates from Tier-2/3 colleges despite relevant skills.",
    "degree_branch_bias": "Non-CSE/IT branch can trigger degree filters even when skill alignment is strong.",
    "cgpa_penalty": "CGPA below common screening thresholds may cause automatic filtering.",
    "career_gap": "Unexplained employment gaps are often treated as elevated risk signals.",
    "tier2_location": "Non-metro location may reduce perceived network strength and immediate availability.",
    "name_origin": "Name-origin cues can trigger unconscious regional or community assumptions.",
    "project_credibility": "Projects without metrics or company context may be dismissed as lightweight.",
    "gender_coded_language": "Soft-skill-heavy or gender-coded phrasing can reduce perceived technical fit.",
}


class BiasClassifier:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self._model = None
        self._thresholds: dict[str, float] = {key: 0.45 for key in BIAS_LABEL_KEYS}
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            return
        payload = joblib.load(self.model_path)
        if isinstance(payload, dict):
            self._model = payload.get("model")
            self._thresholds.update(payload.get("thresholds", {}))
        else:
            self._model = payload
        if METADATA_PATH.exists():
            try:
                meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
                self._thresholds.update(meta.get("thresholds", {}))
            except json.JSONDecodeError:
                pass

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def predict_proba(self, profile: dict[str, Any], *, resume_text: str = "") -> dict[str, float]:
        if not self.is_ready:
            return {key: 0.0 for key in BIAS_LABEL_KEYS}
        features = extract_features(profile, resume_text=resume_text)
        vector = np.asarray([[features[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
        estimators = getattr(self._model, "estimators_", None)
        if estimators is None:
            return {key: 0.0 for key in BIAS_LABEL_KEYS}

        scores: dict[str, float] = {}
        for idx, key in enumerate(BIAS_LABEL_KEYS):
            estimator = estimators[idx]
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(vector)[0]
                scores[key] = float(proba[-1]) if len(proba) > 1 else float(proba[0])
            else:
                scores[key] = float(estimator.predict(vector)[0])
        return scores

    def predict_flags(
        self,
        profile: dict[str, Any],
        *,
        resume_text: str = "",
        patterns: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        scores = self.predict_proba(profile, resume_text=resume_text)
        flags: list[dict[str, Any]] = []
        for key in BIAS_LABEL_KEYS:
            threshold = self._thresholds.get(key, 0.45)
            if scores.get(key, 0.0) < threshold:
                continue
            label = key
            if patterns and key in patterns:
                display = patterns[key].get("label", key.replace("_", " ").title())
            else:
                display = key.replace("_", " ").title()
            flags.append({
                "bias_type": key,
                "label": display,
                "severity": _SEVERITY_BY_LABEL.get(key, "medium"),
                "evidence": _EVIDENCE_TEMPLATES.get(key, "Potential unconscious bias signal detected."),
                "recruiter_decoded": _EVIDENCE_TEMPLATES.get(key, "Potential unconscious bias signal detected."),
                "confidence": round(scores[key], 3),
                "model": "xgboost",
            })
        return flags


@lru_cache(maxsize=1)
def get_bias_classifier() -> BiasClassifier:
    return BiasClassifier()


def save_model(model: Any, thresholds: dict[str, float], metrics: dict[str, Any]) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "thresholds": thresholds}, DEFAULT_MODEL_PATH)
    METADATA_PATH.write_text(
        json.dumps({"thresholds": thresholds, "metrics": metrics, "labels": BIAS_LABEL_KEYS}, indent=2),
        encoding="utf-8",
    )
    get_bias_classifier.cache_clear()
    return DEFAULT_MODEL_PATH


def get_ml_health_status() -> dict[str, Any]:
    """Return ML pipeline status for /health/ml."""
    classifier = get_bias_classifier()
    meta: dict[str, Any] = {}
    if METADATA_PATH.exists():
        try:
            meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    metrics = meta.get("metrics", {})
    return {
        "status": "ok" if classifier.is_ready else "degraded",
        "bias_classifier": {
            "ready": classifier.is_ready,
            "model_path": str(classifier.model_path),
            "macro_f1": metrics.get("macro_f1"),
            "train_size": metrics.get("train_size"),
            "labels": meta.get("labels", BIAS_LABEL_KEYS),
            "per_label_f1": metrics.get("per_label_f1"),
        },
        "rewrite_mapper": {
            "ready": True,
            "engine": "rule_based",
        },
        "config": {
            "use_ml_bias_classifier": settings.use_ml_bias_classifier,
        },
    }

```

<div style="page-break-after: always;"></div>

<div id="backendmlfeaturespy"></div>

## backend/ml/features.py

```python
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

```

<div style="page-break-after: always;"></div>

<div id="backendmainpy"></div>

## backend/main.py

```python
import sys
import os
from pathlib import Path

# Suppress TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set project root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from backend.core.config import settings
from backend.core.database import check_db_connection, init_db
from backend.core.logging_config import configure_logging, log_event
from backend.routers import analysis_router, auth_router, resume_router
from backend.utils.llm_client import call_llm


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="MirrorCue AI backend API",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth_router)
    app.include_router(resume_router)
    app.include_router(analysis_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ml")
    async def health_ml() -> dict:
        from backend.ml.bias_classifier import get_ml_health_status

        return get_ml_health_status()

    @app.on_event("startup")
    async def startup_checks() -> None:
        Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)

        for issue in settings.runtime_issues():
            log_event(agent="SYSTEM", event="config_warning", details={"message": issue})

        try:
            await init_db()
            await check_db_connection()
            log_event(agent="SYSTEM", event="startup_check", details={"database": "ok"})
        except Exception:
            log_event(level=40, agent="SYSTEM", event="error", details={"component": "database"}, exc_info=True)

        try:
            from backend.ml.bias_classifier import get_bias_classifier

            classifier = get_bias_classifier()
            log_event(
                agent="SYSTEM",
                event="startup_check",
                details={
                    "bias_ml_model": "ok" if classifier.is_ready else "missing_run_train_bias_model",
                },
            )
        except Exception:
            log_event(
                level=40,
                agent="SYSTEM",
                event="startup_check",
                details={"bias_ml_model": "error"},
                exc_info=True,
            )

        if settings.gemini_api_key:
            try:
                call_llm(
                    prompt="Hello",
                    system_instruction="Reply with one word.",
                    temperature=0.0,
                    agent="SYSTEM",
                )
                log_event(agent="SYSTEM", event="startup_check", details={"gemini": "ok"})
            except Exception:
                log_event(level=40, agent="SYSTEM", event="error", details={"component": "gemini"}, exc_info=True)

        if settings.groq_api_key:
            try:
                call_llm(
                    prompt="Hello",
                    system_instruction="Reply with one word.",
                    temperature=0.0,
                    use_groq=True,
                    agent="SYSTEM",
                )
                log_event(agent="SYSTEM", event="startup_check", details={"groq": "ok"})
            except Exception:
                log_event(level=40, agent="SYSTEM", event="error", details={"component": "groq"}, exc_info=True)

    try:
        FastAPIInstrumentor.instrument_app(app)
    except Exception:
        log_event(
            level=40,
            agent="SYSTEM",
            event="warning",
            details={"message": "OpenTelemetry instrumentation failed"},
            exc_info=True,
        )
    return app


app = create_app()

```

<div style="page-break-after: always;"></div>

<div id="backendcoreconfigpy"></div>

## backend/core/config.py

```python
from functools import lru_cache
from typing import List
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("backend/.env", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_name: str = "MirrorCue AI"
    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    tuned_bias_model: str = Field(default="", alias="TUNED_BIAS_MODEL")
    tuned_rewrite_model: str = Field(default="", alias="TUNED_REWRITE_MODEL")
    bias_ml_model_path: str = Field(default="backend/models/bias_classifier.joblib", alias="BIAS_ML_MODEL_PATH")
    use_ml_bias_classifier: bool = Field(default=True, alias="USE_ML_BIAS_CLASSIFIER")

    database_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/mirrorcue",
        alias="DATABASE_URL",
    )
    secret_key: str = Field(
        default="change_me_to_a_real_secret_key_32_chars",
        alias="SECRET_KEY",
    )
    algorithm: str = Field(default="HS256", alias="ALGORITHM")
    access_token_expire_days: int = Field(default=7, alias="ACCESS_TOKEN_EXPIRE_DAYS")

    upload_dir: str = Field(default="./uploads", alias="UPLOAD_DIR")
    max_file_size_mb: int = Field(default=5, alias="MAX_FILE_SIZE_MB")

    cors_origins: str = Field(default="http://localhost:5173", alias="CORS_ORIGINS")

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, value: str) -> str:
        if not value.strip():
            return "http://localhost:5173"
        return value

    @property
    def cors_origin_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def secret_key_is_valid(self) -> bool:
        return len(self.secret_key) >= 32

    @property
    def normalized_database_url(self) -> str:
        raw_url = self.database_url.strip()
        if raw_url.startswith("postgres://"):
            raw_url = raw_url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif raw_url.startswith("postgresql://"):
            raw_url = raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return raw_url

    @property
    def is_supabase_url(self) -> bool:
        lowered = self.normalized_database_url.lower()
        return "supabase.co" in lowered or "pooler.supabase.com" in lowered

    @property
    def is_supabase_transaction_pooler(self) -> bool:
        lowered = self.normalized_database_url.lower()
        return "pooler.supabase.com:6543" in lowered

    @property
    def database_url_with_ssl_defaults(self) -> str:
        url = self.normalized_database_url
        if not self.is_supabase_url:
            return url

        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query.setdefault("ssl", "require")
        if self.is_supabase_transaction_pooler:
            query.setdefault("statement_cache_size", "0")
        new_query = urlencode(query)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))

    def runtime_issues(self) -> list[str]:
        issues: list[str] = []
        if not self.secret_key_is_valid:
            issues.append("SECRET_KEY must be at least 32 characters long.")
        if not self.normalized_database_url.startswith("postgresql+asyncpg://"):
            issues.append("DATABASE_URL should use the postgresql+asyncpg scheme.")
        if not self.gemini_api_key:
            issues.append("GEMINI_API_KEY is not configured.")
        if not self.groq_api_key:
            issues.append("GROQ_API_KEY is not configured.")
        return issues


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

```

<div style="page-break-after: always;"></div>

<div id="backendmodelsresumepy"></div>

## backend/models/resume.py

```python
import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base


class Resume(Base):
    __tablename__ = "resumes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user = relationship("User", back_populates="resumes")
    analyses = relationship("Analysis", back_populates="resume", cascade="all, delete-orphan")

```

<div style="page-break-after: always;"></div>

<div id="backendmodelsanalysispy"></div>

## backend/models/analysis.py

```python
import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    resume_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False, index=True)
    jd_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)

    parsed_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    ats_result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    bias_result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    qa_questions: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    qa_answers: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    rewrite_result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    user = relationship("User", back_populates="analyses")
    resume = relationship("Resume", back_populates="analyses")

```

<div style="page-break-after: always;"></div>

<div id="frontendsrccomponentsanalysisatstabtsx"></div>

## frontend/src/components/analysis/ATSTab.tsx

```typescript
import type { AnalysisResult } from "../../types";
import { ResumeAnalysisLoader } from "./ResumeAnalysisLoader";

function Gauge({ score }: { score: number }) {
  const color = score >= 70 ? "#22c55e" : score >= 49 ? "#f59e0b" : "#ef4444";
  const matchLabel = score >= 70 ? "Good match" : score >= 49 ? "Moderate match" : "Needs work";
  return (
    <div
      className="mx-auto flex h-44 w-44 items-center justify-center rounded-full"
      style={{
        background: `conic-gradient(${color} ${score * 3.6}deg, #f1f5f9 0deg)`,
        boxShadow: "0 0 30px rgba(0,212,170,0.1)",
      }}
    >
      <div className="flex h-32 w-32 flex-col items-center justify-center rounded-full bg-white text-slate-900">
        <span className="text-4xl font-bold">{Math.round(score)}%</span>
        <span className="text-xs uppercase tracking-widest text-slate-500">ATS Match</span>
        <span className="mt-1 text-[10px] font-semibold uppercase tracking-wide text-slate-400">{matchLabel}</span>
      </div>
    </div>
  );
}

export function ATSTab({ analysis }: { analysis: AnalysisResult }) {
  const result = analysis.ats_result;
  if (!result) {
    if (analysis.status === "failed") {
      return (
        <div className="rounded-2xl border border-dashed border-rose-300 bg-rose-50 p-8 text-center text-rose-600">
          <p className="font-semibold text-lg">Analysis Failed</p>
          <p className="mt-2 text-sm text-rose-600">We ran into an error while running the ATS matcher pipeline.</p>
        </div>
      );
    }
    return (
      <ResumeAnalysisLoader
        title="Analyzing ATS Alignment..."
        phrases={[
          "Parsing job description requirements...",
          "Comparing resume skill set mapping...",
          "Calculating semantic similarity metrics...",
          "Running term-frequency analysis...",
        ]}
      />
    );
  }

  const semanticScore = result.semantic_score || 0;
  const keywordScore = result.keyword_score || 0;

  const matchedDetail =
    result.matched_keywords_detail?.length
      ? result.matched_keywords_detail
      : result.matched_keywords.map((keyword) => ({
          keyword,
          match_reason: "Aligned with job description requirements",
        }));

  const missingDetail =
    result.missing_keywords_detail?.length
      ? result.missing_keywords_detail
      : result.missing_keywords.map((keyword) => ({
          keyword,
          importance: "Present in job description but missing from resume",
        }));

  const relatedKeywords = result.related_recommended_keywords ?? [];
  const resumeStrengths = result.additional_resume_strengths ?? [];

  return (
    <div className="space-y-6">
      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <h3 className="mb-6 text-lg font-semibold uppercase tracking-wider text-slate-600">
          ATS Alignment Index
        </h3>
        <div className="flex flex-col items-center gap-6">
          <Gauge score={result.score} />
          <div className="w-full grid grid-cols-2 gap-4">
            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 text-center">
              <p className="text-sm uppercase tracking-wide text-slate-600 font-semibold">Semantic Match</p>
              <p className="mt-2 text-2xl font-bold text-slate-900">{Math.round(semanticScore)}%</p>
            </div>
            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 text-center">
              <p className="text-sm uppercase tracking-wide text-slate-600 font-semibold">Keyword Match</p>
              <p className="mt-2 text-2xl font-bold text-slate-900">{Math.round(keywordScore)}%</p>
            </div>
          </div>

          {result.section_breakdown ? (
            <div className="w-full mt-2 border-t border-slate-100 pt-6">
              <h4 className="mb-4 text-xs font-bold uppercase tracking-wider text-slate-500 text-center">
                Detailed Section Breakdown
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 w-full">
                {(["skills", "experience", "projects", "education"] as const).map((sec) => {
                  const info = result.section_breakdown?.[sec];
                  if (!info) return null;
                  return (
                    <div key={sec} className="rounded-2xl border border-slate-200 bg-slate-50/50 p-4 transition-all hover:bg-slate-50 w-full text-left">
                      <div className="flex items-center justify-between">
                        <p className="text-sm font-semibold capitalize text-slate-800">{sec}</p>
                        <span className="rounded-lg bg-slate-200 px-2 py-0.5 text-xs font-bold text-slate-600">
                          {info.weight}x wt
                        </span>
                      </div>
                      <div className="mt-3 space-y-2">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-500">Keyword Match:</span>
                          <span className="font-bold text-slate-700">{Math.round(info.coverage_percent)}%</span>
                        </div>
                        <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden">
                          <div className="bg-amber-500 h-full rounded-full" style={{ width: `${info.coverage_percent}%` }} />
                        </div>

                        <div className="flex justify-between text-xs pt-1">
                          <span className="text-slate-500">Semantic Match:</span>
                          <span className="font-bold text-slate-700">{Math.round(info.semantic_similarity)}%</span>
                        </div>
                        <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden">
                          <div className="bg-blue-500 h-full rounded-full" style={{ width: `${info.semantic_similarity}%` }} />
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : null}
        </div>
      </section>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <section className="rounded-3xl border border-emerald-200 bg-emerald-50 p-6">
          <h3 className="mb-4 text-lg font-semibold text-slate-900 uppercase tracking-wider">
            Matched Keywords ({matchedDetail.length})
          </h3>
          <div className="flex flex-wrap gap-2">
            {matchedDetail.length ? (
              matchedDetail.map((item) => (
                <span
                  key={item.keyword}
                  className="inline-flex items-center rounded-xl border border-emerald-200 bg-white px-3 py-1.5 text-sm font-medium text-emerald-800 shadow-sm"
                >
                  <span className="text-emerald-500 mr-1.5">✓</span>
                  {item.keyword}
                </span>
              ))
            ) : (
              <p className="text-sm text-slate-600">No matched keywords detected yet.</p>
            )}
          </div>
        </section>

        <section className="rounded-3xl border border-rose-200 bg-rose-50 p-6">
          <h3 className="mb-4 text-lg font-semibold text-slate-900 uppercase tracking-wider">
            Missing Keywords ({missingDetail.length})
          </h3>
          <div className="flex flex-wrap gap-2">
            {missingDetail.length ? (
              missingDetail.map((item) => (
                <span
                  key={item.keyword}
                  className="inline-flex items-center rounded-xl border border-rose-200 bg-white px-3 py-1.5 text-sm font-medium text-rose-800 shadow-sm"
                >
                  <span className="text-rose-500 mr-1.5">✕</span>
                  {item.keyword}
                </span>
              ))
            ) : (
              <p className="text-sm text-slate-400">No critical gaps detected.</p>
            )}
          </div>
        </section>
      </div>

      <section className="rounded-3xl border border-blue-200 bg-blue-50 p-6">
        <h3 className="mb-4 text-lg font-semibold text-slate-900 uppercase tracking-wider">
          Related Recommended Keywords ({relatedKeywords.length})
        </h3>
        <p className="mb-4 text-sm text-slate-600">
          Industry-relevant terms recruiters and ATS systems commonly expect for this role.
        </p>
        <div className="space-y-3">
          {relatedKeywords.length ? (
            relatedKeywords.map((item) => (
              <div
                key={item.keyword}
                className="rounded-2xl border border-blue-200 bg-white px-4 py-3"
              >
                <p className="font-medium text-blue-800">{item.keyword}</p>
                {item.reason ? (
                  <p className="mt-1 text-sm text-slate-600">{item.reason}</p>
                ) : null}
              </div>
            ))
          ) : (
            <p className="text-sm text-slate-500">No market recommendations generated.</p>
          )}
        </div>
      </section>

      <section className="rounded-3xl border border-violet-200 bg-violet-50 p-6">
        <h3 className="mb-4 text-lg font-semibold text-slate-900 uppercase tracking-wider">
          Additional Resume Strengths ({resumeStrengths.length})
        </h3>
        <p className="mb-4 text-sm text-slate-600">
          Skills, certifications, and achievements in your resume not explicitly required by the JD.
        </p>
        <div className="flex flex-wrap gap-3">
          {resumeStrengths.length ? (
            resumeStrengths.map((item) => (
              <span
                key={`${item.item}-${item.category}`}
                className="inline-flex flex-col rounded-2xl border border-violet-200 bg-white px-4 py-2 text-sm"
              >
                <span className="font-medium text-violet-900">{item.item}</span>
                {item.category ? (
                  <span className="text-xs uppercase tracking-wide text-violet-600 mt-1">
                    {item.category}
                  </span>
                ) : null}
              </span>
            ))
          ) : (
            <p className="text-sm text-slate-500">No extra strengths identified.</p>
          )}
        </div>
      </section>

      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <h3 className="mb-4 text-lg font-semibold text-slate-900 uppercase tracking-wider">
          Formatting Flags
        </h3>
        <div className="space-y-3">
          {result.formatting_flags.length ? (
            result.formatting_flags.map((flag) => (
              <div
                key={flag}
                className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800"
              >
                {flag}
              </div>
            ))
          ) : (
            <p className="text-sm text-slate-500">No formatting warnings.</p>
          )}
        </div>
        <div className="mt-6 rounded-2xl bg-slate-50 border border-slate-200 p-4">
          <p className="text-sm font-semibold text-slate-600 uppercase tracking-wide">Recommendation</p>
          <p className="mt-2 text-slate-900">{result.recommendation}</p>
        </div>
      </section>
    </div>
  );
}
```

<div style="page-break-after: always;"></div>

<div id="frontendsrccomponentsanalysisbiasmirrortabtsx"></div>

## frontend/src/components/analysis/BiasMirrorTab.tsx

```typescript
import { useState } from "react";
import { ChevronDown, GraduationCap, Languages } from "lucide-react";
import clsx from "clsx";
import { BiasCard } from "./BiasCard";
import { ResumeAnalysisLoader } from "./ResumeAnalysisLoader";
import type { AnalysisResult } from "../../types";

export function BiasMirrorTab({ analysis }: { analysis: AnalysisResult }) {
  const bias = analysis.bias_result;
  const [branchExpanded, setBranchExpanded] = useState(true);
  const [masculineExpanded, setMasculineExpanded] = useState(true);

  if (!bias) {
    if (analysis.status === "failed") {
      return (
        <div className="rounded-2xl border border-dashed border-rose-300 bg-rose-50 p-8 text-center text-rose-600">
          <p className="font-semibold text-lg">Analysis Failed</p>
          <p className="mt-2 text-sm text-rose-600">We ran into an error while running the bias mirror pipeline.</p>
        </div>
      );
    }
    return (
      <ResumeAnalysisLoader
        title="Running Bias Audit..."
        phrases={[
          "Loading local XGBoost Multi-Label Classifier...",
          "Scanning demographic identifiers (name and gender cues)...",
          "Auditing college prestige and geographic location biases...",
          "Evaluating degree/branch priority vectors...",
          "Reviewing resume description credibility scores...",
        ]}
      />
    );
  }

  const getGradientColor = (score: number) => {
    if (score > 70) return "from-orange-400 to-rose-500";
    if (score > 40) return "from-amber-400 to-orange-500";
    return "from-emerald-400 to-teal-500";
  };

  const criticalCount = bias.flags.filter(f => f.severity === "high").length;
  const indiaSpecificCount = bias.flags.filter(
    f => f.bias_type && (f.bias_type.includes("india") || f.bias_type === "degree_branch_bias" || f.bias_type === "vernacular_english")
  ).length;

  const branch_bias = bias.branch_bias || null;
  const masculine_bias = bias.masculine_bias || null;

  const branchBiasRisk = branch_bias?.risk_level || "Low";
  const skillScore = branch_bias?.skill_alignment_score || 0;

  const masculineBiasRisk = masculine_bias?.risk_level || "Low";

  return (
    <div className="space-y-6">
      {/* Unconscious Bias Audit Summary */}
      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-slate-900 mb-2">UNCONSCIOUS BIAS AUDIT REPORT</h3>
            <p className="text-sm text-slate-600">{bias.summary}</p>
          </div>
          <div className="text-right">
            <p className="text-sm uppercase tracking-wide text-slate-600 font-semibold">BIAS INDEX RATING</p>
            <p className="mt-2 text-4xl font-bold text-rose-600">
              {Math.round(bias.bias_score)}
              <span className="text-2xl text-slate-400">/100</span>
            </p>
          </div>
        </div>
        <div className="mt-4 h-3 overflow-hidden rounded-full bg-slate-200">
          <div
            className={`h-full bg-gradient-to-r ${getGradientColor(bias.bias_score)}`}
            style={{ width: `${Math.min(100, bias.bias_score)}%` }}
          />
        </div>
        <div className="mt-4 flex flex-wrap gap-6 text-sm">
          <div>
            <p className="uppercase tracking-wide text-slate-600 font-semibold">LOCALIZED BIAS TRIGGERS</p>
            <p className="mt-1 text-lg font-bold text-slate-900">{indiaSpecificCount} India-Specific / Localized</p>
          </div>
          <div>
            <p className="uppercase tracking-wide text-slate-600 font-semibold">CRITICAL FAILURES</p>
            <p className="mt-1 text-lg font-bold text-rose-600">{criticalCount} High Severity</p>
          </div>
        </div>
      </section>

      {/* Enhanced Bias Categories Grid */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Branch Bias Card */}
        <article className="animate-fade-in rounded-3xl border border-slate-200 bg-white p-5 shadow-sm space-y-4 flex flex-col justify-between">
          <div className="space-y-4">
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-center gap-3">
                <div className="rounded-2xl bg-indigo-50 p-3 text-indigo-600">
                  <GraduationCap className="h-6 w-6" />
                </div>
                <div>
                  <h4 className="font-semibold text-slate-900">Branch Bias Audit</h4>
                  <p className="text-xs text-slate-500">Evaluating degree-independent skill alignment</p>
                </div>
              </div>
              <button
                onClick={() => setBranchExpanded(prev => !prev)}
                className="rounded-xl p-1.5 hover:bg-slate-100 transition text-slate-400"
              >
                <ChevronDown className={clsx("h-5 w-5 transform transition", branchExpanded && "rotate-180")} />
              </button>
            </div>

            <div className="flex flex-wrap gap-2">
              <span
                className={clsx(
                  "rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide border",
                  branchBiasRisk === "High" && "bg-rose-50 border-rose-200 text-rose-700",
                  branchBiasRisk === "Medium" && "bg-amber-50 border-amber-200 text-amber-700",
                  branchBiasRisk === "Low" && "bg-emerald-50 border-emerald-200 text-emerald-700"
                )}
              >
                Risk: {branchBiasRisk}
              </span>
              {branch_bias?.rankings_influenced && (
                <span className="rounded-full bg-rose-100 border border-rose-200 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-rose-700 animate-pulse">
                  ⚠️ Rankings Influenced
                </span>
              )}
            </div>

            <div className="flex items-center gap-4 rounded-2xl bg-slate-50 border border-slate-100 p-4">
              <div className="relative flex-shrink-0">
                <svg className="w-16 h-16 transform -rotate-90">
                  <circle cx="32" cy="32" r="28" className="text-slate-200" strokeWidth="5" stroke="currentColor" fill="transparent" />
                  <circle
                    cx="32"
                    cy="32"
                    r="28"
                    className="text-indigo-600 transition-all duration-500"
                    strokeWidth="5"
                    strokeDasharray={176}
                    strokeDashoffset={176 - (176 * skillScore) / 100}
                    strokeLinecap="round"
                    stroke="currentColor"
                    fill="transparent"
                  />
                </svg>
                <span className="absolute inset-0 flex items-center justify-center text-sm font-bold text-slate-800">
                  {Math.round(skillScore)}%
                </span>
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-800">Skill Alignment Score</p>
                <p className="text-xs text-slate-500 leading-normal">
                  Purely candidate-centered technical competency, projects, certifications, assessments & GitHub activity (independent of college/branch).
                </p>
              </div>
            </div>

            {branchExpanded && (
              <div className="space-y-4 pt-2 border-t border-slate-100 text-sm">
                <div className="space-y-1">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">📊 Evidence & Analysis</p>
                  <p className="text-slate-700 leading-relaxed italic bg-slate-50 p-3 rounded-2xl border border-slate-100">
                    {branch_bias?.evidence || "No evidence recorded."}
                  </p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">💡 Confidence Level</p>
                  <p className="text-slate-700 font-medium">{branch_bias?.confidence || "Medium"}</p>
                </div>
                {branch_bias?.recommendations && branch_bias.recommendations.length > 0 && (
                  <div className="space-y-1">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">🌱 Bias Breaking Actions</p>
                    <ul className="list-disc pl-4 space-y-1.5 text-slate-700">
                      {branch_bias.recommendations.map((rec, i) => (
                        <li key={i}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        </article>

        {/* Masculine Language Bias Card */}
        <article className="animate-fade-in rounded-3xl border border-slate-200 bg-white p-5 shadow-sm space-y-4 flex flex-col justify-between">
          <div className="space-y-4">
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-center gap-3">
                <div className="rounded-2xl bg-pink-50 p-3 text-pink-600">
                  <Languages className="h-6 w-6" />
                </div>
                <div>
                  <h4 className="font-semibold text-slate-900">Masculine Language Audit</h4>
                  <p className="text-xs text-slate-500">Evaluating job description linguistic inclusivity</p>
                </div>
              </div>
              <button
                onClick={() => setMasculineExpanded(prev => !prev)}
                className="rounded-xl p-1.5 hover:bg-slate-100 transition text-slate-400"
              >
                <ChevronDown className={clsx("h-5 w-5 transform transition", masculineExpanded && "rotate-180")} />
              </button>
            </div>

            <div className="flex flex-wrap gap-2">
              <span
                className={clsx(
                  "rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide border",
                  masculineBiasRisk === "High" && "bg-rose-50 border-rose-200 text-rose-700",
                  masculineBiasRisk === "Medium" && "bg-amber-50 border-amber-200 text-amber-700",
                  masculineBiasRisk === "Low" && "bg-emerald-50 border-emerald-200 text-emerald-700"
                )}
              >
                Risk: {masculineBiasRisk}
              </span>
              <span className="rounded-full bg-slate-100 border border-slate-200 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-slate-700">
                {masculine_bias?.density_score || 0}% Density
              </span>
            </div>

            <div className="rounded-2xl bg-slate-50 border border-slate-100 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-600 mb-2">📊 Language Balance Index</p>
              <div className="h-2 w-full bg-slate-200 rounded-full overflow-hidden">
                <div
                  className={clsx(
                    "h-full rounded-full transition-all duration-500",
                    masculineBiasRisk === "High" ? "bg-rose-500" : masculineBiasRisk === "Medium" ? "bg-amber-500" : "bg-emerald-500"
                  )}
                  style={{ width: `${Math.min(100, (masculine_bias?.density_score || 0) * 50)}%` }}
                />
              </div>
              <p className="text-[11px] text-slate-500 mt-1.5 leading-normal">
                Based on the ratio of gender-coded terms detected in the job description to the total word count.
              </p>
            </div>

            {masculineExpanded && (
              <div className="space-y-4 pt-2 border-t border-slate-100 text-sm">
                <div className="space-y-1">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">📝 Evidence & Scan Results</p>
                  <p className="text-slate-700 leading-relaxed italic bg-slate-50 p-3 rounded-2xl border border-slate-100">
                    {masculine_bias?.evidence || "No evidence recorded."}
                  </p>
                </div>

                {masculine_bias?.matched_terms && masculine_bias.matched_terms.length > 0 && (
                  <div className="space-y-2">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">🔍 Configurable Dictionary Matches</p>
                    <div className="overflow-x-auto rounded-2xl border border-slate-200 bg-white">
                      <table className="w-full text-xs text-left border-collapse">
                        <thead className="bg-slate-50 text-slate-700 border-b border-slate-200 font-semibold">
                          <tr>
                            <th className="p-2.5 font-medium">Term Found</th>
                            <th className="p-2.5 font-medium">Inclusive Alternative</th>
                            <th className="p-2.5 font-medium text-center">Count</th>
                          </tr>
                        </thead>
                        <tbody>
                          {masculine_bias.matched_terms.map(item => (
                            <tr key={item.term} className="hover:bg-slate-50 border-b border-slate-100 last:border-0">
                              <td className="p-2.5 text-rose-600 font-mono font-semibold">{item.term}</td>
                              <td className="p-2.5 text-emerald-600 font-semibold">✓ {item.replacement}</td>
                              <td className="p-2.5 text-slate-500 text-center font-medium">{item.count}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {masculine_bias?.recommendation && (
                  <div className="space-y-1">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">🌱 Inclusive Alternatives & Fixes</p>
                    <ul className="list-disc pl-4 space-y-1.5 text-slate-700">
                      {masculine_bias.recommendation.split("; ").map((rec, i) => (
                        <li key={i}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        </article>
      </div>

      {/* Individual Resume Bias Flag Cards */}
      <div className="space-y-4">
        {bias.flags.filter(flag => flag.bias_type !== "degree_branch_bias" && flag.bias_type !== "masculine_language_bias").length ? (
          bias.flags
            .filter(flag => flag.bias_type !== "degree_branch_bias" && flag.bias_type !== "masculine_language_bias")
            .map(flag => <BiasCard key={`${flag.bias_type}-${flag.candidate_wrote}`} flag={flag} />)
        ) : (
          <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-6 text-emerald-700">
            Great news — the current resume shows low visible bias signals.
          </div>
        )}
      </div>
    </div>
  );
}

```

<div style="page-break-after: always;"></div>

<div id="frontendsrccomponentsanalysisrewritetabtsx"></div>

## frontend/src/components/analysis/RewriteTab.tsx

```typescript
import type { AnalysisResult } from "../../types";
import { QAForm } from "./QAForm";
import { ResumeAnalysisLoader } from "./ResumeAnalysisLoader";

function downloadText(filename: string, content: string) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const href = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = href;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(href);
}

export function RewriteTab({ analysis }: { analysis: AnalysisResult }) {
  const questions = analysis.qa_questions?.questions ?? [];
  const rewrite = analysis.rewrite_result;

  // Extract bias flags once to avoid double-filtering and fix the TS conditional error
  const biasFlags = analysis.bias_result?.flags?.filter((f: any) =>
    f.candidate_wrote &&
    f.candidate_wrote !== "resume content" &&
    f.candidate_wrote !== "project details" &&
    f.candidate_wrote !== "experience details"
  ) ?? [];

  if (!rewrite) {
    if (analysis.status === "failed") {
      return (
        <div className="rounded-2xl border border-dashed border-rose-300 bg-rose-50 p-8 text-center text-rose-600">
          <p className="font-semibold text-lg">Analysis Failed</p>
          <p className="mt-2 text-sm text-rose-600">We ran into an error while running the resume rewrite pipeline.</p>
        </div>
      );
    }

    // QA form takes priority — if questions exist, always show the form first
    if (questions.length) {
      if (
        analysis.status === "rewriting" ||
        analysis.status === "rewrite" ||
        analysis.status === "qa_validate" ||
        analysis.status === "qa_validated"
      ) {
        return (
          <ResumeAnalysisLoader
            title="Generating Verified Resume Rewrite..."
            phrases={[
              "Mapping Q&A answers to resume items...",
              "Resolving grammatical structures...",
              "Injecting missing ATS keywords...",
              "Formulating action-oriented bullets with metrics...",
              "Removing unconscious bias phrases...",
            ]}
          />
        );
      }
      return (
        <QAForm analysisId={analysis.id} questions={questions} />
      );
    }

    // Only show error if status is complete AND there are no questions to answer
    if (analysis.status === "complete") {
      return (
        <div className="rounded-2xl border border-dashed border-rose-300 bg-rose-50 p-8 text-center text-rose-600">
          <p className="font-semibold">Rewrite Not Available</p>
          <p className="mt-2 text-sm text-rose-600">The analysis completed but the rewrite data wasn't generated. Please try submitting the QA answers again.</p>
        </div>
      );
    }

    // Default: pipeline still running, questions not yet generated
    return (
      <ResumeAnalysisLoader
        title="Formulating Clarification Questions..."
        phrases={[
          "Scanning resume sections for metric and technology gaps...",
          "Comparing keyword gaps for specialized questions...",
          "Preparing targeted clarification questions...",
        ]}
      />
    );
  }

  const exportText = [
    "MirrorCue Rewrite Summary",
    rewrite.rewritten_summary || "Summary not available",
    "",
    "Experience",
    ...rewrite.rewritten_experience?.flatMap((item) => [
      `${item.title} | ${item.company} | ${item.duration}`,
      ...item.bullets.map((bullet) => `- ${bullet}`),
      "",
    ]) || ["No experience items"],
    "Projects",
    ...rewrite.rewritten_projects?.flatMap((item) => [
      `${item.name} | ${item.tech_stack.join(", ")}`,
      ...item.bullets.map((bullet) => `- ${bullet}`),
      "",
    ]) || ["No projects"],
  ].join("\n");

  return (
    <div className="space-y-6">
      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap gap-8 items-end">
            <div>
              <p className="text-sm text-slate-500 uppercase tracking-wide font-semibold mb-1">ATS Score</p>
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-bold text-slate-400">{Math.round(analysis.ats_result?.score ?? 0)}%</span>
                <span className="text-slate-400 text-xl">→</span>
                <span className="text-4xl font-bold text-emerald-600">{Math.round(rewrite.ats_score_after)}%</span>
                {rewrite.ats_score_delta != null && (
                  <span className={`text-lg font-semibold ${rewrite.ats_score_delta >= 0 ? "text-emerald-500" : "text-rose-500"}`}>
                    ({rewrite.ats_score_delta >= 0 ? "+" : ""}{Math.round(rewrite.ats_score_delta)}%)
                  </span>
                )}
              </div>
            </div>
            <div>
              <p className="text-sm text-slate-500 uppercase tracking-wide font-semibold mb-1">Keywords added</p>
              <p className="text-3xl font-bold text-blue-600">{rewrite.total_keywords_added ?? 0}</p>
            </div>
            <div>
              <p className="text-sm text-slate-500 uppercase tracking-wide font-semibold mb-1">Bias phrases removed</p>
              <p className="text-3xl font-bold text-purple-600">{rewrite.total_bias_phrases_removed ?? 0}</p>
            </div>
          </div>
          <button
            onClick={() => downloadText("mirrorcue-rewrite.txt", exportText)}
            className="rounded-2xl border border-blue-300 bg-blue-50 px-4 py-2 text-sm text-blue-600 hover:bg-blue-100 transition self-start"
          >
            Download Rewrite
          </button>
        </div>
      </section>

      {/* Bias Removal Evidence Panel */}
      {biasFlags.length > 0 && (
        <section className="rounded-3xl border border-purple-200 bg-purple-50 p-5">
          <h3 className="text-sm font-semibold uppercase tracking-wide text-purple-700 mb-3">
            Bias phrases addressed in this rewrite
          </h3>
          <div className="space-y-2">
            {biasFlags.map((flag: any, i: number) => (
              <div key={i} className="flex flex-wrap items-start gap-2 text-sm">
                <span className="rounded-lg bg-rose-100 text-rose-700 px-2 py-0.5 line-through decoration-rose-400">
                  {flag.candidate_wrote}
                </span>
                <span className="text-slate-400 mt-0.5">→</span>
                <span className="rounded-lg bg-emerald-100 text-emerald-700 px-2 py-0.5">
                  {flag.fix || "rephrased with action verbs"}
                </span>
                <span className="text-slate-400 text-xs mt-1">
                  ({flag.label || flag.bias_type})
                </span>
              </div>
            ))}
          </div>
        </section>
      )}

      <div className="grid gap-6 xl:grid-cols-2">
        <section className="rounded-3xl border border-rose-200 bg-rose-50 p-6">
          <h3 className="mb-4 text-lg font-semibold text-slate-900 uppercase tracking-wider">Original</h3>
          <div className="space-y-4">
            {rewrite.original_experience?.length ? (
              rewrite.original_experience.map((item, index) => (
                <div key={`${item.title}-${index}`} className="rounded-2xl bg-white border border-rose-200 p-4">
                  <p className="font-medium text-slate-900">{item.title}</p>
                  <p className="text-sm text-slate-600">
                    {item.company}
                    {item.duration && item.duration !== "Not specified" && item.duration !== "N/A"
                      ? ` · ${item.duration}`
                      : ""}
                  </p>
                  <ul className="mt-3 space-y-2 text-sm text-slate-700">
                    {item.bullets.map((bullet) => <li key={bullet}>• {bullet}</li>)}
                  </ul>
                </div>
              ))
            ) : (
              <p className="text-sm text-slate-500">No experience items</p>
            )}
            {rewrite.original_projects?.length ? (
              rewrite.original_projects.map((item, index) => (
                <div key={`${item.name}-${index}`} className="rounded-2xl bg-white border border-rose-200 p-4">
                  <p className="font-medium text-slate-900">{item.name}</p>
                  <ul className="mt-3 space-y-2 text-sm text-slate-700">
                    {item.bullets.map((bullet) => <li key={bullet}>• {bullet}</li>)}
                  </ul>
                </div>
              ))
            ) : (
              <p className="text-sm text-slate-500">No projects</p>
            )}
          </div>
        </section>

        <section className="rounded-3xl border border-emerald-200 bg-emerald-50 p-6">
          <h3 className="mb-4 text-lg font-semibold text-slate-900 uppercase tracking-wider">MirrorCue Rewrite</h3>
          <div className="space-y-4">
            {rewrite.rewritten_experience?.length ? (
              rewrite.rewritten_experience.map((item, index) => (
                <div key={`${item.title}-${index}`} className="rounded-2xl bg-white border border-emerald-200 p-4">
                  <p className="font-medium text-slate-900">{item.title}</p>
                  {(item.company || (item.duration && item.duration !== "Not specified")) && (
                    <p className="text-sm text-slate-600">
                      {item.company}
                      {item.duration && item.duration !== "Not specified" && item.duration !== "N/A"
                        ? ` · ${item.duration}`
                        : ""}
                    </p>
                  )}
                  <ul className="mt-3 space-y-2 text-sm text-slate-700">
                    {item.bullets?.length
                      ? item.bullets.filter(Boolean).map((bullet, bi) => (
                          <li key={bi} className="flex gap-2">
                            <span className="text-emerald-500 mt-0.5">•</span>
                            <span>{bullet}</span>
                          </li>
                        ))
                      : <li className="text-slate-400 italic">No bullets generated for this item.</li>
                    }
                  </ul>
                </div>
              ))
            ) : (
              <p className="text-sm text-slate-500">No rewritten experience</p>
            )}
            {rewrite.rewritten_projects?.length ? (
              rewrite.rewritten_projects.map((item, index) => (
                <div key={`${item.name}-${index}`} className="rounded-2xl bg-white border border-emerald-200 p-4">
                  <p className="font-medium text-slate-900">{item.name}</p>
                  {item.tech_stack?.length > 0 && (
                    <p className="text-sm text-slate-600">{item.tech_stack.filter(Boolean).join(", ")}</p>
                  )}
                  <ul className="mt-3 space-y-2 text-sm text-slate-700">
                    {item.bullets?.length
                      ? item.bullets.filter(Boolean).map((bullet, bi) => (
                          <li key={bi} className="flex gap-2">
                            <span className="text-emerald-500 mt-0.5">•</span>
                            <span>{bullet}</span>
                          </li>
                        ))
                      : <li className="text-slate-400 italic">No bullets generated for this item.</li>
                    }
                  </ul>
                </div>
              ))
            ) : (
              <p className="text-sm text-slate-500">No rewritten projects</p>
            )}
          </div>
        </section>
      </div>

      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <h3 className="text-lg font-semibold text-slate-900 uppercase tracking-wider">New Summary</h3>
        <p className="mt-3 text-slate-700">{rewrite.rewritten_summary || "Summary not available"}</p>
      </section>

      <details className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm group">
        <summary className="cursor-pointer text-lg font-semibold text-slate-900 group-open:text-blue-600 uppercase tracking-wider">Changes Made</summary>
        <p className="mt-4 text-slate-700">{rewrite.changes_summary || "Changes summary not available"}</p>
      </details>
    </div>
  );
}
```

<div style="page-break-after: always;"></div>

<div id="backendtestsconftestpy"></div>

## backend/tests/conftest.py

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock
from backend.main import app
from backend.core.database import get_db

@pytest.fixture
def mock_db():
    mock = AsyncMock()
    
    # Setup standard mocked results for db execution
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_result.scalars.return_value.all.return_value = []
    
    mock.execute.return_value = mock_result
    return mock

@pytest.fixture
def client(mock_db):
    async def override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

```

<div style="page-break-after: always;"></div>

<div id="backendteststest_ats_matcherpy"></div>

## backend/tests/test_ats_matcher.py

```python
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from backend.agents.ats_matcher import match_ats

def mock_call_llm_json_side_effect(prompt, system_instruction, schema_hint, agent, user_id, analysis_id):
    if "JD Keyword Extraction Agent" in system_instruction:
        return {"keywords": ["Python", "FastAPI", "Docker", "AWS"]}
    elif "expert recruiter" in system_instruction:
        return {
            "recommendation": "Mocked recommendation paragraph.",
            "related_recommended_keywords": [{"keyword": "Kubernetes", "reason": "Cloud container orchestration tool."}],
            "additional_resume_strengths": [{"item": "React", "category": "skill"}]
        }
    return {}


@patch("backend.agents.ats_matcher.sklearn_cosine_similarity")
@patch("backend.agents.ats_matcher._embedder")
@patch("backend.agents.ats_matcher.call_llm_json")
def test_match_ats_irrelevant(mock_call_llm_json, mock_embedder, mock_cosine_similarity):
    mock_call_llm_json.side_effect = mock_call_llm_json_side_effect
    mock_embedder.encode.return_value = [np.zeros(384, dtype=np.float32)]
    mock_cosine_similarity.return_value = np.array([[0.15]])

    resume_json = {
        "skills": [],
        "experience": [],
        "projects": [],
        "education": []
    }
    resume_text = "I do sales and accounting."
    jd_text = "Looking for a Python Backend SDE with FastAPI."

    result = match_ats(resume_json, resume_text, jd_text)
    
    assert result["score"] < 20.0
    assert "section_breakdown" in result
    assert result["recommendation"] == "Mocked recommendation paragraph."
    assert len(result["formatting_flags"]) > 0


@patch("backend.agents.ats_matcher.sklearn_cosine_similarity")
@patch("backend.agents.ats_matcher._embedder")
@patch("backend.agents.ats_matcher.call_llm_json")
def test_match_ats_moderate(mock_call_llm_json, mock_embedder, mock_cosine_similarity):
    mock_call_llm_json.side_effect = mock_call_llm_json_side_effect
    mock_embedder.encode.return_value = [np.zeros(384, dtype=np.float32)]
    mock_cosine_similarity.return_value = np.array([[0.55]])

    resume_json = {
        "skills": ["python"],
        "experience": [],
        "projects": [],
        "education": []
    }
    resume_text = "I write Python scripts."
    jd_text = "Looking for a Python Backend SDE with FastAPI."

    result = match_ats(resume_json, resume_text, jd_text)
    
    assert 20.0 <= result["score"] <= 65.0
    assert "section_breakdown" in result
    assert result["recommendation"] == "Mocked recommendation paragraph."


@patch("backend.agents.ats_matcher.sklearn_cosine_similarity")
@patch("backend.agents.ats_matcher._embedder")
@patch("backend.agents.ats_matcher.call_llm_json")
def test_match_ats_strong(mock_call_llm_json, mock_embedder, mock_cosine_similarity):
    mock_call_llm_json.side_effect = mock_call_llm_json_side_effect
    mock_embedder.encode.return_value = [np.zeros(384, dtype=np.float32)]
    mock_cosine_similarity.return_value = np.array([[0.80]])

    resume_json = {
        "skills": ["python", "fastapi", "docker", "aws"],
        "experience": [{"title": "Senior Python Backend Developer", "company": "Tech Corp", "description": "Experienced backend developer designing and building scalable Python APIs with FastAPI. Deploying to AWS and containerizing with Docker."}],
        "projects": [],
        "education": []
    }
    resume_text = "Senior Python Developer with backend experience, FastAPI, AWS, Docker. Looking for a software engineering role."
    jd_text = "Looking for a Senior Python Developer with backend experience, FastAPI, AWS, Docker."

    result = match_ats(resume_json, resume_text, jd_text)
    
    assert result["score"] > 65.0
    assert "section_breakdown" in result
    assert result["recommendation"] == "Mocked recommendation paragraph."

@patch("backend.agents.ats_matcher.sklearn_cosine_similarity")
@patch("backend.agents.ats_matcher._embedder")
@patch("backend.agents.ats_matcher.call_llm_json")
def test_match_ats_experience_bullets_only(mock_call_llm_json, mock_embedder, mock_cosine_similarity):
    def custom_mock_llm_side_effect(prompt, system_instruction, schema_hint, agent, user_id, analysis_id):
        if "JD Keyword Extraction Agent" in system_instruction:
            return {"keywords": ["Python", "Pandas", "Scikit-learn", "NumPy", "NLP"]}
        elif "expert recruiter" in system_instruction:
            return {
                "recommendation": "Mocked recommendation paragraph.",
                "related_recommended_keywords": [],
                "additional_resume_strengths": [],
            }
        return {}

    mock_call_llm_json.side_effect = custom_mock_llm_side_effect
    mock_embedder.encode.return_value = [np.zeros(384, dtype=np.float32)]
    mock_cosine_similarity.return_value = np.array([[0.65]])

    resume_json = {
        "skills": ["Python", "Machine Learning"],
        "experience": [
            {
                "title": "Intern",
                "company": "Elewayte",
                "bullets": ["Python, Scikit-learn, Pandas"],
            },
            {
                "title": "Intern",
                "company": "Unified Mentor",
                "bullets": ["Python, Pandas, NumPy, Scikit-learn"],
            },
            {
                "title": "Intern",
                "company": "Ziffity",
                "bullets": ["NLP-based intent recognition"],
            },
        ],
        "projects": [],
        "graduation_year": "2025",
    }
    resume_text = "Intern with Python, Pandas, Scikit-learn, NumPy, NLP experience."
    jd_text = "Junior Python developer with Pandas, Scikit-learn, NumPy, and NLP skills."

    result = match_ats(resume_json, resume_text, jd_text)

    exp_coverage = result["section_breakdown"]["experience"]["coverage_percent"]
    exp_matched = result["section_breakdown"]["experience"]["matched_keywords"]
    assert exp_coverage >= 35.0, f"Experience coverage too low: {exp_coverage}%"
    assert "Python" in exp_matched
    assert "Pandas" in exp_matched or "Scikit-learn" in exp_matched


@patch("backend.agents.ats_matcher.sklearn_cosine_similarity")
@patch("backend.agents.ats_matcher._embedder")
@patch("backend.agents.ats_matcher.call_llm_json")
def test_match_ats_new_rules(mock_call_llm_json, mock_embedder, mock_cosine_similarity):
    def custom_mock_llm_side_effect(prompt, system_instruction, schema_hint, agent, user_id, analysis_id):
        if "JD Keyword Extraction Agent" in system_instruction:
            return {"keywords": ["software engineering", "Python", "FastAPI", "Git"]}
        elif "expert recruiter" in system_instruction:
            return {
                "recommendation": "Mocked recommendation paragraph.",
                "related_recommended_keywords": [{"keyword": "Kubernetes", "reason": "Cloud container orchestration tool."}],
                "additional_resume_strengths": [{"item": "React", "category": "skill"}]
            }
        return {}

    mock_call_llm_json.side_effect = custom_mock_llm_side_effect
    mock_embedder.encode.return_value = [np.zeros(384, dtype=np.float32)]
    mock_cosine_similarity.return_value = np.array([[0.75]])

    resume_json = {
        "skills": ["software engineering principles", "python dev"],
        "experience": [],
        "projects": [],
        "education": ["computer science degree"]
    }
    resume_text = "Experienced in software engineering principles and python dev. Degree in computer science."
    jd_text = "Python Developer. software engineering FastAPI FastAPI Git"

    result = match_ats(resume_json, resume_text, jd_text)
    
    assert "software engineering" in result["section_breakdown"]["skills"]["matched_keywords"]
    
    for sec in ["skills", "experience", "projects", "education"]:
        assert "coverage_percent" in result["section_breakdown"][sec]
        assert "semantic_similarity" in result["section_breakdown"][sec]
        assert "weight" in result["section_breakdown"][sec]
        
    assert "pass_threshold" in result
    assert result["pass_threshold"] == 50



```

<div style="page-break-after: always;"></div>

<div id="backendteststest_bias_mirrorpy"></div>

## backend/tests/test_bias_mirror.py

```python
import pytest
from backend.agents.bias_mirror import _estimate_bias_score, _load_bias_patterns

def test_bias_patterns_weights():
    # Load actual patterns from file
    patterns = _load_bias_patterns()
    
    assert "name_origin" in patterns
    assert "gender_coded_language" in patterns
    assert "project_credibility" in patterns
    assert "masculine_language_bias" in patterns

    # Verify custom severity weights are exactly what we configured
    assert patterns["name_origin"]["severity_weight"] == 0.30
    assert patterns["gender_coded_language"]["severity_weight"] == 0.30
    assert patterns["masculine_language_bias"]["severity_weight"] == 0.30
    assert patterns["project_credibility"]["severity_weight"] == 0.95

def test_estimate_bias_score():
    pattern_weights = {
        "name_origin": 0.30,
        "gender_coded_language": 0.30,
        "project_credibility": 0.95
    }

    # case 1: medium severity name bias flag
    # Expected: 0.30 * 10 = 3.0
    flags_1 = [{"bias_type": "name_origin", "severity": "medium"}]
    assert _estimate_bias_score(flags_1, pattern_weights) == 3.0

    # case 2: high severity project credibility flag
    # Expected: 0.95 * 20 = 19.0
    flags_2 = [{"bias_type": "project_credibility", "severity": "high"}]
    assert _estimate_bias_score(flags_2, pattern_weights) == 19.0

    # case 3: combined flags
    # Expected: 19.0 * 1.0 + 3.0 * 0.85 = 19.0 + 2.55 = 21.55
    flags_3 = [
        {"bias_type": "name_origin", "severity": "medium"},
        {"bias_type": "project_credibility", "severity": "high"}
    ]
    assert _estimate_bias_score(flags_3, pattern_weights) == 21.55

```

<div style="page-break-after: always;"></div>

<div id="backendteststest_resume_utilspy"></div>

## backend/tests/test_resume_utils.py

```python
from backend.agents.parser import _ensure_resume_defaults
from backend.utils.resume_utils import (
    build_experience_section_text,
    extract_experience_description,
    flatten_resume_text_field,
)


def test_flatten_resume_text_field_from_bullet_list():
    assert flatten_resume_text_field(
        ["Python, Scikit-learn, Pandas", "Built ML models"]
    ) == "Python, Scikit-learn, Pandas Built ML models"


def test_flatten_resume_text_field_from_nested_dicts():
    assert flatten_resume_text_field(
        [{"text": "NLP-based intent recognition"}, {"bullet": "Used Python and Pandas"}]
    ) == "NLP-based intent recognition Used Python and Pandas"


def test_extract_experience_description_prefers_description_then_bullets():
    exp = {
        "description": "",
        "bullets": [
            "Python, Scikit-learn, Pandas",
            "Python, Pandas, NumPy, Scikit-learn",
            "NLP-based intent recognition",
        ],
    }
    assert "Python" in extract_experience_description(exp)
    assert "NLP-based intent recognition" in extract_experience_description(exp)


def test_ensure_resume_defaults_coalesces_bullets_into_description():
    payload = {
        "experience": [
            {
                "title": "Intern",
                "company": "Elewayte",
                "duration": "2024",
                "bullets": ["Python, Scikit-learn, Pandas"],
            }
        ]
    }
    normalized = _ensure_resume_defaults(payload)
    assert normalized["experience"][0]["description"] == "Python, Scikit-learn, Pandas"


def test_build_experience_section_text_includes_internship_bullets():
    experience = [
        {
            "title": "Intern",
            "company": "Elewayte",
            "bullets": ["Python, Scikit-learn, Pandas"],
        },
        {
            "title": "Intern",
            "company": "Unified Mentor",
            "bullets": ["Python, Pandas, NumPy, Scikit-learn"],
        },
        {
            "title": "Intern",
            "company": "Ziffity",
            "bullets": ["NLP-based intent recognition"],
        },
    ]
    text = build_experience_section_text(experience)
    assert "Python" in text
    assert "Scikit-learn" in text
    assert "NLP-based intent recognition" in text

```

<div style="page-break-after: always;"></div>

