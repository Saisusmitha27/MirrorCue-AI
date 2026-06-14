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
