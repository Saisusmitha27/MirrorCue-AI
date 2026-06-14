from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.core.database import get_db
from backend.models.analysis import Analysis
from backend.models.resume import Resume
from backend.models.user import User
from backend.routers.auth import get_current_user
from backend.schemas.resume import ResumeResponse, ResumeUploadResponse

router = APIRouter(prefix="/resume", tags=["resume"])


async def _validate_pdf(file: UploadFile) -> None:
    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF uploads are allowed")

    signature = await file.read(5)
    await file.seek(0)
    if signature != b"%PDF-":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is not a valid PDF")


async def _save_upload(file: UploadFile, user_id: str) -> tuple[str, str]:
    await _validate_pdf(file)

    user_dir = Path(settings.upload_dir) / user_id
    user_dir.mkdir(parents=True, exist_ok=True)

    safe_name = file.filename or "resume.pdf"
    stored_name = f"{uuid4()}_{safe_name}"
    destination = user_dir / stored_name

    total_bytes = 0
    async with aiofiles.open(destination, "wb") as handle:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > settings.max_file_size_bytes:
                await handle.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds {settings.max_file_size_mb}MB limit",
                )
            await handle.write(chunk)

    await file.seek(0)
    return safe_name, str(destination)


@router.post("/upload", response_model=ResumeUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_resume(
    file: UploadFile = File(...),
    jd_text: str = Form(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ResumeUploadResponse:
    jd_text = jd_text.strip()
    if not jd_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job description is required")

    filename, file_path = await _save_upload(file, str(current_user.id))

    resume = Resume(
        user_id=current_user.id,
        filename=filename,
        file_path=file_path,
    )
    db.add(resume)
    await db.flush()

    analysis = Analysis(
        user_id=current_user.id,
        resume_id=resume.id,
        jd_text=jd_text,
        status="pending",
    )
    db.add(analysis)
    await db.commit()
    await db.refresh(resume)
    await db.refresh(analysis)

    return ResumeUploadResponse(resume_id=resume.id, analysis_id=analysis.id)


@router.get("/list", response_model=list[ResumeResponse])
async def list_resumes(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ResumeResponse]:
    result = await db.execute(
        select(Resume).where(Resume.user_id == current_user.id).order_by(Resume.uploaded_at.desc())
    )
    rows = result.scalars().all()
    return [ResumeResponse.model_validate(row) for row in rows]
