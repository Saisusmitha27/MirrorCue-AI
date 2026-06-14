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
