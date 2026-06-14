from sqlalchemy import text
from collections.abc import AsyncGenerator
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.core.config import settings

engine_kwargs = {
    "echo": False,
    "pool_pre_ping": True,
}

if settings.is_supabase_transaction_pooler:
    engine_kwargs["poolclass"] = NullPool

engine = create_async_engine(settings.database_url_with_ssl_defaults, **engine_kwargs)
AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    from backend.models.base import Base
    from backend.models.analysis import Analysis  # noqa: F401
    from backend.models.resume import Resume  # noqa: F401
    from backend.models.user import User  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def check_db_connection() -> None:
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
