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