from __future__ import annotations

from dataclasses import dataclass

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection

from app.core.config import PROJECT_ROOT


ALEMBIC_INI_PATH = PROJECT_ROOT / "alembic.ini"
ALEMBIC_SCRIPT_PATH = PROJECT_ROOT / "alembic"


class MigrationValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class MigrationValidationResult:
    current_revisions: tuple[str, ...]
    expected_heads: tuple[str, ...]
    has_version_table: bool

    @property
    def is_current(self) -> bool:
        return self.has_version_table and set(self.current_revisions) == set(self.expected_heads)

    def describe(self) -> str:
        current = ", ".join(self.current_revisions) if self.current_revisions else "(none)"
        expected = ", ".join(self.expected_heads) if self.expected_heads else "(none)"
        if not self.has_version_table:
            return f"alembic_version table is missing; current={current}; expected_head={expected}"
        return f"current_revision={current}; expected_head={expected}"


def _build_alembic_config() -> Config:
    config = Config(str(ALEMBIC_INI_PATH))
    config.set_main_option("script_location", str(ALEMBIC_SCRIPT_PATH))
    return config


def _get_expected_heads() -> tuple[str, ...]:
    script_directory = ScriptDirectory.from_config(_build_alembic_config())
    return tuple(sorted(script_directory.get_heads()))


def _get_current_revisions(connection: Connection) -> tuple[bool, tuple[str, ...]]:
    inspector = inspect(connection)
    if not inspector.has_table("alembic_version"):
        return False, ()

    rows = connection.execute(text("SELECT version_num FROM alembic_version")).scalars().all()
    revisions = tuple(sorted(str(row) for row in rows if row))
    return True, revisions


def get_migration_validation_result(connection: Connection) -> MigrationValidationResult:
    has_version_table, current_revisions = _get_current_revisions(connection)
    expected_heads = _get_expected_heads()
    return MigrationValidationResult(
        current_revisions=current_revisions,
        expected_heads=expected_heads,
        has_version_table=has_version_table,
    )


def validate_database_migrations(connection: Connection, *, mode: str = "strict") -> MigrationValidationResult:
    result = get_migration_validation_result(connection)
    if result.is_current:
        return result

    message = f"Database schema is not at Alembic head: {result.describe()}"
    if mode == "strict":
        raise MigrationValidationError(message)
    return result