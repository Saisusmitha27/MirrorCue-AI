from functools import lru_cache
from typing import List
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("backend/.env", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_name: str = "MirrorCue AI"
    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    tuned_bias_model: str = Field(default="", alias="TUNED_BIAS_MODEL")
    tuned_rewrite_model: str = Field(default="", alias="TUNED_REWRITE_MODEL")
    bias_ml_model_path: str = Field(default="backend/models/bias_classifier.joblib", alias="BIAS_ML_MODEL_PATH")
    use_ml_bias_classifier: bool = Field(default=True, alias="USE_ML_BIAS_CLASSIFIER")

    database_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/mirrorcue",
        alias="DATABASE_URL",
    )
    secret_key: str = Field(
        default="change_me_to_a_real_secret_key_32_chars",
        alias="SECRET_KEY",
    )
    algorithm: str = Field(default="HS256", alias="ALGORITHM")
    access_token_expire_days: int = Field(default=7, alias="ACCESS_TOKEN_EXPIRE_DAYS")

    upload_dir: str = Field(default="./uploads", alias="UPLOAD_DIR")
    max_file_size_mb: int = Field(default=5, alias="MAX_FILE_SIZE_MB")

    cors_origins: str = Field(default="http://localhost:5173", alias="CORS_ORIGINS")

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, value: str) -> str:
        if not value.strip():
            return "http://localhost:5173"
        return value

    @property
    def cors_origin_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def secret_key_is_valid(self) -> bool:
        return len(self.secret_key) >= 32

    @property
    def normalized_database_url(self) -> str:
        raw_url = self.database_url.strip()
        if raw_url.startswith("postgres://"):
            raw_url = raw_url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif raw_url.startswith("postgresql://"):
            raw_url = raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return raw_url

    @property
    def is_supabase_url(self) -> bool:
        lowered = self.normalized_database_url.lower()
        return "supabase.co" in lowered or "pooler.supabase.com" in lowered

    @property
    def is_supabase_transaction_pooler(self) -> bool:
        lowered = self.normalized_database_url.lower()
        return "pooler.supabase.com:6543" in lowered

    @property
    def database_url_with_ssl_defaults(self) -> str:
        url = self.normalized_database_url
        if not self.is_supabase_url:
            return url

        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query.setdefault("ssl", "require")
        if self.is_supabase_transaction_pooler:
            query.setdefault("statement_cache_size", "0")
        new_query = urlencode(query)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))

    def runtime_issues(self) -> list[str]:
        issues: list[str] = []
        if not self.secret_key_is_valid:
            issues.append("SECRET_KEY must be at least 32 characters long.")
        if not self.normalized_database_url.startswith("postgresql+asyncpg://"):
            issues.append("DATABASE_URL should use the postgresql+asyncpg scheme.")
        if not self.gemini_api_key:
            issues.append("GEMINI_API_KEY is not configured.")
        if not self.groq_api_key:
            issues.append("GROQ_API_KEY is not configured.")
        return issues


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
