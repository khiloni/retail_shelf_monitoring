"""Abstract repositories for planograms and alerts."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ...entities.alert import Alert
from ...entities.planogram import Planogram


class PlanogramRepository(ABC):
    @abstractmethod
    async def create(self, planogram: Planogram) -> Planogram: ...

    @abstractmethod
    async def get_by_shelf_id(self, shelf_id: str) -> Optional[Planogram]: ...

    @abstractmethod
    async def update(self, planogram: Planogram) -> Planogram: ...

    @abstractmethod
    async def delete(self, shelf_id: str) -> bool: ...

    @abstractmethod
    async def list_all(self) -> List[Planogram]: ...


class AlertRepository(ABC):
    @abstractmethod
    async def create(self, alert: Alert) -> Alert: ...

    @abstractmethod
    async def get_by_id(self, alert_id: str) -> Optional[Alert]: ...

    @abstractmethod
    async def get_by_cell(
        self, shelf_id: str, row_idx: int, item_idx: int
    ) -> Optional[Alert]: ...

    @abstractmethod
    async def update(self, alert: Alert) -> Alert: ...

    @abstractmethod
    async def get_active_alerts(self, shelf_id: Optional[str] = None) -> List[Alert]: ...

    @abstractmethod
    async def confirm_alert(self, alert_id: str, confirmed_by: str) -> Alert: ...

    @abstractmethod
    async def dismiss_alert(self, alert_id: str) -> Alert: ...
