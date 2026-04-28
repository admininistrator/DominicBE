from logging.config import fileConfig
from pathlib import Path
import sys

from alembic import context
from sqlalchemy import Column, MetaData, String, Table, engine_from_config, inspect, pool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.core.config import settings  # noqa: E402
from app.core.database import Base  # noqa: E402
import app.models.chat_models  # noqa: F401,E402
import app.models.knowledge_models  # noqa: F401,E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", settings.sqlalchemy_database_url)

target_metadata = Base.metadata


def ensure_version_table_shape(connection) -> None:
    metadata = MetaData()
    version_table = Table(
        "alembic_version",
        metadata,
        Column("version_num", String(255), primary_key=True),
    )
    metadata.create_all(connection, tables=[version_table], checkfirst=True)

    inspector = inspect(connection)
    columns = inspector.get_columns("alembic_version")
    version_column = next((column for column in columns if column["name"] == "version_num"), None)
    if version_column is None:
        return

    current_length = getattr(version_column.get("type"), "length", None)
    if current_length is not None and current_length >= 255:
        return

    dialect_name = connection.dialect.name
    if dialect_name == "postgresql":
        connection.exec_driver_sql(
            "ALTER TABLE alembic_version ALTER COLUMN version_num TYPE VARCHAR(255)"
        )
    elif dialect_name.startswith("mysql"):
        connection.exec_driver_sql(
            "ALTER TABLE alembic_version MODIFY version_num VARCHAR(255) NOT NULL"
        )


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        with connection.begin():
            ensure_version_table_shape(connection)
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
