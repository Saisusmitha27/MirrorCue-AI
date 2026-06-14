from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from backend.ml.features import BIAS_LABEL_KEYS, BIAS_TYPE_TO_KEY, extract_features, features_to_vector, labels_to_vector

ROOT = Path(__file__).resolve().parent.parent.parent
BIAS_JSONL = ROOT / "backend" / "data" / "bias_tuning_data.jsonl"
RECRUITMENT_CSV = ROOT / "Recruitment_Bias_And_Fairness_Dataset.csv"
ENGINEERING_CSV = ROOT / "Engineering_graduate_salary.csv"
AI_HIRING_CSV = ROOT / "AI_hiring_audit_Dataset.csv"


def _load_jsonl_training() -> tuple[list[list[float]], list[list[int]]]:
    x_rows: list[list[float]] = []
    y_rows: list[list[int]] = []
    if not BIAS_JSONL.exists():
        return x_rows, y_rows

    with BIAS_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                payload = json.loads(row["input"])
                output = json.loads(row["output"])
                profile = payload.get("candidate_profile", payload)
                features = extract_features(profile)
                x_rows.append(features_to_vector(features))
                y_rows.append(labels_to_vector(output.get("flags", [])))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return x_rows, y_rows


def _load_recruitment_csv() -> tuple[list[list[float]], list[list[int]]]:
    x_rows: list[list[float]] = []
    y_rows: list[list[int]] = []
    if not RECRUITMENT_CSV.exists():
        return x_rows, y_rows

    with RECRUITMENT_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                gender = row.get("gender", "")
                score = float(row.get("screening_score", 0) or 0)
                shortlisted = int(float(row.get("shortlisted", 0) or 0))
                experience = float(row.get("experience_years", 0) or 0)
                education = str(row.get("education_level", "")).lower()
            except (TypeError, ValueError):
                continue

            profile = {
                "gender": gender,
                "years_experience": experience,
                "education_level": education,
                "skills": [],
                "projects": [{"description": "internship project"}],
                "certifications": [],
                "experience": [{"title": "Role"}] if experience > 0 else [],
                "college_tier": "Tier-2",
                "cgpa": "7.0",
                "location": "Chennai",
                "branch": "Computer Science",
                "job_role": "Software Engineer",
            }
            features = extract_features(profile, screening_score=score)
            x_rows.append(features_to_vector(features))

            labels = {key: 0 for key in BIAS_LABEL_KEYS}
            if gender.lower() == "female" and shortlisted == 0 and score >= 55:
                labels["gender_coded_language"] = 1
            if gender.lower() == "female" and shortlisted == 0 and score >= 70:
                labels["name_origin"] = 1
            y_rows.append([labels[key] for key in BIAS_LABEL_KEYS])

    return x_rows, y_rows


def _engineering_tier(value: str) -> str:
    try:
        tier = int(float(value))
    except (TypeError, ValueError):
        return "Tier-3"
    if tier == 1:
        return "Tier-1"
    if tier == 2:
        return "Tier-2"
    return "Tier-3"


def _load_engineering_csv() -> tuple[list[list[float]], list[list[int]]]:
    x_rows: list[list[float]] = []
    y_rows: list[list[int]] = []
    if not ENGINEERING_CSV.exists():
        return x_rows, y_rows

    with ENGINEERING_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                college_tier = _engineering_tier(row.get("CollegeTier", "0"))
                cgpa = float(row.get("collegeGPA", 0) or 0)
                city_tier = int(float(row.get("CollegeCityTier", 0) or 0))
                gender = "Female" if str(row.get("Gender", "")).lower() == "f" else "Male"
                specialization = str(row.get("Specialization", ""))
                grad_year = int(float(row.get("GraduationYear", 2020) or 2020))
            except (TypeError, ValueError):
                continue

            profile = {
                "college_tier": college_tier,
                "cgpa": str(cgpa),
                "gender": gender,
                "branch": specialization,
                "location": str(row.get("CollegeState", "")),
                "graduation_year": str(grad_year),
                "years_experience": max(0, 2026 - grad_year - 1),
                "skills": ["Python", "Java"],
                "projects": [{"description": "academic capstone"}],
                "certifications": [],
                "experience": [],
                "job_role": "Software Engineer",
            }
            features = extract_features(profile)
            x_rows.append(features_to_vector(features))

            labels = {key: 0 for key in BIAS_LABEL_KEYS}
            if college_tier == "Tier-3":
                labels["prestige_gap"] = 1
            if cgpa < 65:
                labels["cgpa_penalty"] = 1
            if city_tier == 0:
                labels["tier2_location"] = 1
            if _is_non_cse_spec(specialization):
                labels["degree_branch_bias"] = 1
            y_rows.append([labels[key] for key in BIAS_LABEL_KEYS])

    return x_rows, y_rows


def _is_non_cse_spec(spec: str) -> bool:
    lowered = spec.lower()
    return any(token in lowered for token in ("mechanical", "civil", "instrumentation", "electronics", "biotech"))


def _load_ai_hiring_csv() -> tuple[list[list[float]], list[list[int]]]:
    x_rows: list[list[float]] = []
    y_rows: list[list[int]] = []
    if not AI_HIRING_CSV.exists():
        return x_rows, y_rows

    with AI_HIRING_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                years = float(row.get("Years_Experience", 0) or 0)
                skill_fit = float(row.get("Skill_Fit_Score", 0) or 0)
                education = str(row.get("Education_Level", ""))
                job = str(row.get("Job_Category", ""))
                final_decision = int(float(row.get("Final_Decision", 0) or 0))
                divergence = abs(float(row.get("Score_Divergence", 0) or 0))
            except (TypeError, ValueError):
                continue

            cgpa = "8.0" if education.lower() in {"phd", "masters"} else "7.2"
            profile = {
                "candidate_id": row.get("Candidate_ID"),
                "years_experience": years,
                "college_tier": "Tier-2" if education.lower() == "bachelors" else "Tier-1",
                "cgpa": cgpa,
                "branch": "Computer Science" if "software" in job.lower() or "data" in job.lower() else "Business Administration",
                "job_role": job,
                "gender": "Male",
                "location": "Chennai",
                "skills": ["Python", "SQL"],
                "projects": [{"description": "analytics project"}],
                "certifications": [],
                "experience": [{"title": job}] if years > 0 else [],
            }
            features = extract_features(profile, skill_fit_score=skill_fit)
            x_rows.append(features_to_vector(features))

            labels = {key: 0 for key in BIAS_LABEL_KEYS}
            if skill_fit < 35 and final_decision == 0:
                labels["project_credibility"] = 1
            if skill_fit >= 55 and final_decision == 0 and divergence > 5:
                labels["degree_branch_bias"] = 1
            if years >= 10 and final_decision == 0:
                labels["career_gap"] = 1
            y_rows.append([labels[key] for key in BIAS_LABEL_KEYS])

    return x_rows, y_rows


def build_training_dataset() -> tuple[list[list[float]], list[list[int]], dict[str, int]]:
    """Merge JSONL (primary) + 3 real-world CSV datasets."""
    sources = {
        "jsonl": _load_jsonl_training(),
        "recruitment": _load_recruitment_csv(),
        "engineering": _load_engineering_csv(),
        "ai_hiring": _load_ai_hiring_csv(),
    }

    x_all: list[list[float]] = []
    y_all: list[list[int]] = []
    counts: dict[str, int] = {}

    for name, (x_rows, y_rows) in sources.items():
        counts[name] = len(x_rows)
        x_all.extend(x_rows)
        y_all.extend(y_rows)

    return x_all, y_all, counts
