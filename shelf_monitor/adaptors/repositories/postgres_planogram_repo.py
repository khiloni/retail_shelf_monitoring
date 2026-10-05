"""PostgreSQL planogram repository (aliases SqlitePlanogramRepository as SQLAlchemy handles both)."""
from .sqlite_planogram_repo import SqlitePlanogramRepository

PostgresPlanogramRepository = SqlitePlanogramRepository
