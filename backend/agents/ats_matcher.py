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