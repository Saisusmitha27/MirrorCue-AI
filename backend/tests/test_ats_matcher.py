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


