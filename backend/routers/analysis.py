from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.agents.orchestrator import run_pipeline
from backend.core.database import AsyncSessionLocal, get_db
from backend.core.logging_config import log_event
from backend.models.analysis import Analysis
from backend.models.user import User
from backend.routers.auth import get_current_user
from backend.schemas.analysis import (
    AnalysisListItem,
    AnalysisResult,
    AnalysisRunRequest,
    RewriteRequest,
    StatusResponse,
)

router = APIRouter(prefix="/analysis", tags=["analysis"])


async def _get_user_analysis(db: AsyncSession, analysis_id: UUID, user_id: UUID) -> Analysis:
    result = await db.execute(
        select(Analysis)
        .options(selectinload(Analysis.resume))
        .where(Analysis.id == analysis_id, Analysis.user_id == user_id)
    )
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return analysis


async def _run_analysis_background(analysis_id: UUID, user_id: UUID, qa_answers: dict[str, str] | None = None) -> None:
    async with AsyncSessionLocal() as session:
        analysis = await session.get(Analysis, analysis_id)
        if not analysis:
            return
        try:
            await run_pipeline(
                session=session,
                analysis_id=str(analysis_id),
                user_id=str(user_id),
                jd_text=analysis.jd_text,
                qa_answers=qa_answers,
            )
        except Exception as exc:
            log_event(
                level=40,
                agent="A1_ORCHESTRATOR",
                user_id=str(user_id),
                analysis_id=str(analysis_id),
                event="background_task_error",
                details={"error": str(exc)},
                exc_info=True,
            )
            await session.execute(
                update(Analysis).where(Analysis.id == analysis_id).values(status="failed")
            )
            await session.commit()


@router.post("/run", response_model=StatusResponse)
async def run_analysis(
    payload: AnalysisRunRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StatusResponse:
    analysis = await _get_user_analysis(db, payload.analysis_id, current_user.id)
    analysis.status = "running"
    await db.commit()
    background_tasks.add_task(_run_analysis_background, analysis.id, current_user.id, None)
    return StatusResponse(analysis_id=analysis.id, status="running")


@router.get("/list", response_model=list[AnalysisListItem])
async def list_analyses(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[AnalysisListItem]:
    result = await db.execute(
        select(Analysis)
        .options(selectinload(Analysis.resume))
        .where(Analysis.user_id == current_user.id)
        .order_by(Analysis.created_at.desc())
    )
    analyses = result.scalars().all()
    return [
        AnalysisListItem(
            id=analysis.id,
            filename=analysis.resume.filename if analysis.resume else "Resume",
            ats_score=(analysis.ats_result or {}).get("score") if analysis.ats_result else None,
            bias_score=(analysis.bias_result or {}).get("bias_score") if analysis.bias_result else None,
            created_at=analysis.created_at,
            status=analysis.status,
        )
        for analysis in analyses
    ]


@router.get("/{analysis_id}", response_model=AnalysisResult)
async def get_analysis(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AnalysisResult:
    analysis = await _get_user_analysis(db, analysis_id, current_user.id)
    return AnalysisResult.model_validate(analysis)


@router.post("/{analysis_id}/rewrite", response_model=StatusResponse)
async def rewrite_analysis(
    analysis_id: UUID,
    payload: RewriteRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StatusResponse:
    analysis = await _get_user_analysis(db, analysis_id, current_user.id)
    analysis.qa_answers = payload.qa_answers
    analysis.status = "rewriting"
    await db.commit()

    background_tasks.add_task(_run_analysis_background, analysis.id, current_user.id, payload.qa_answers)
    return StatusResponse(analysis_id=analysis.id, status="rewriting")
