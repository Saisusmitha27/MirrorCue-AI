from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class BiasFlag(BaseModel):
    bias_type: str
    label: str
    candidate_wrote: str
    recruiter_decoded: str
    severity: str = Field(pattern="^(low|medium|high)$")
    fix: str
    line_context: str


class ATSResult(BaseModel):
    score: float = Field(ge=0, le=100)
    semantic_score: float = Field(ge=0, le=100)
    keyword_score: float = Field(ge=0, le=100)
    matched_keywords: list[str] = []
    missing_keywords: list[str] = []
    formatting_flags: list[str] = []
    jd_seniority_level: str
    recommendation: str


class QAQuestion(BaseModel):
    id: str
    section: str
    item_name: str
    question: str
    why_needed: str
    example_answer: str
    answer_type: str


class AnalysisResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    resume_id: UUID
    jd_text: str
    status: str
    parsed_json: dict | None = None
    ats_result: dict | None = None
    bias_result: dict | None = None
    qa_questions: dict | None = None
    qa_answers: dict | None = None
    rewrite_result: dict | None = None
    created_at: datetime
    updated_at: datetime


class AnalysisRunRequest(BaseModel):
    analysis_id: UUID


class RewriteRequest(BaseModel):
    qa_answers: dict[str, str]


class AnalysisListItem(BaseModel):
    id: UUID
    filename: str
    ats_score: float | None = None
    bias_score: float | None = None
    created_at: datetime
    status: str


class StatusResponse(BaseModel):
    analysis_id: UUID | None = None
    status: str
