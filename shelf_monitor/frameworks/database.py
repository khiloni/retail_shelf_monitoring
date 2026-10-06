"""SQLAlchemy async database setup.

Default: SQLite (no external service needed).
Optional: PostgreSQL via config.
"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from ..frameworks.logging_config import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


class DatabaseManager:
    """Manages the async SQLAlchemy engine and session factory."""

    def __init__(self, database_url: str) -> None:
        self._url = database_url
        connect_args = {}
        if "sqlite" in database_url:
            connect_args = {"check_same_thread": False}
        self._engine = create_async_engine(
            database_url,
            connect_args=connect_args,
            echo=False,
        )
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False, class_=AsyncSession
        )

    def get_session(self) -> async_sessionmaker[AsyncSession]:
        return self._session_factory

    async def create_tables(self) -> None:
        from ..adaptors.repositories.sqlite_planogram_repo import PlanogramModel  # noqa: F401
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")

    async def close(self) -> None:
        await self._engine.dispose()
        logger.info("Database connection closed")

    async def health_check(self) -> bool:
        try:
            async with self._session_factory() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.error(f"Database health check failed: {exc}")
            return False
