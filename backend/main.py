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
