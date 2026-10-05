"""SQLite/SQLAlchemy implementation of PlanogramRepository."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import Column, DateTime, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ...entities.planogram import Planogram
from ...frameworks.database import Base
from ...frameworks.logging_config import get_logger
from ...usecases.interfaces.repositories import PlanogramRepository

logger = get_logger(__name__)


class PlanogramModel(Base):
    __tablename__ = "planograms"

    shelf_id = Column(String(128), primary_key=True)
    reference_image_path = Column(String(512), default="")
    data_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class SqlitePlanogramRepository(PlanogramRepository):
    """Stores planograms in SQLite (or any SQLAlchemy-supported DB) via async sessions."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def create(self, planogram: Planogram) -> Planogram:
        async with self._session_factory() as session:
            async with session.begin():
                existing = await session.get(PlanogramModel, planogram.shelf_id)
                data = planogram.model_dump_json()
                if existing:
                    existing.reference_image_path = planogram.reference_image_path
                    existing.data_json = data
                    existing.updated_at = datetime.now(timezone.utc)
                else:
                    rec = PlanogramModel(
                        shelf_id=planogram.shelf_id,
                        reference_image_path=planogram.reference_image_path,
                        data_json=data,
                    )
                    session.add(rec)
            await session.commit()
        return planogram

    async def get_by_shelf_id(self, shelf_id: str) -> Optional[Planogram]:
        async with self._session_factory() as session:
            rec = await session.get(PlanogramModel, shelf_id)
            if rec:
                return Planogram.model_validate_json(rec.data_json)
        return None

    async def update(self, planogram: Planogram) -> Planogram:
        return await self.create(planogram)

    async def delete(self, shelf_id: str) -> bool:
        async with self._session_factory() as session:
            async with session.begin():
                rec = await session.get(PlanogramModel, shelf_id)
                if rec:
                    await session.delete(rec)
                    await session.commit()
                    return True
        return False

    async def list_all(self) -> List[Planogram]:
        async with self._session_factory() as session:
            stmt = select(PlanogramModel)
            res = await session.execute(stmt)
            records = res.scalars().all()
            return [Planogram.model_validate_json(r.data_json) for r in records]
