from __future__ import annotations

import argparse
import json
import math
import os
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import MetaData, create_engine, inspect, select, text
from sqlalchemy.engine import Engine

from app.core.config import settings

TABLE_ORDER = [
    "users",
    "chat_sessions",
    "chat_summaries",
    "messages",
    "knowledge_documents",
    "knowledge_chunks",
    "ingestion_jobs",
    "retrieval_events",
    "answer_citations",
    "audit_logs",
]

JSON_COLUMNS = {
    "knowledge_documents": {"metadata_json"},
    "knowledge_chunks": {"metadata_json"},
    "retrieval_events": {"metadata_json"},
    "audit_logs": {"detail_json"},
}


def mask_url(url: str) -> str:
    parsed = urlsplit(url)
    if "@" not in parsed.netloc:
        return url

    credentials, host = parsed.netloc.rsplit("@", 1)
    if ":" not in credentials:
        return urlunsplit((parsed.scheme, f"***@{host}", parsed.path, parsed.query, parsed.fragment))

    username, _password = credentials.split(":", 1)
    return urlunsplit((parsed.scheme, f"{username}:***@{host}", parsed.path, parsed.query, parsed.fragment))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy application data from a legacy MySQL database into PostgreSQL.",
    )
    parser.add_argument(
        "--source-url",
        default=os.getenv("SOURCE_DATABASE_URL") or os.getenv("LEGACY_MYSQL_DATABASE_URL"),
        help="Legacy MySQL SQLAlchemy URL. Can also come from SOURCE_DATABASE_URL or LEGACY_MYSQL_DATABASE_URL.",
    )
    parser.add_argument(
        "--target-url",
        default=os.getenv("TARGET_DATABASE_URL") or settings.sqlalchemy_database_url,
        help="Target PostgreSQL SQLAlchemy URL. Defaults to current app settings or TARGET_DATABASE_URL.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Row batch size for copy operations.",
    )
    parser.add_argument(
        "--truncate-target",
        action="store_true",
        help="Delete existing rows in target tables before copying.",
    )
    parser.add_argument(
        "--tables",
        default=",".join(TABLE_ORDER),
        help="Comma-separated ordered list of tables to copy.",
    )
    return parser


def validate_urls(source_url: str, target_url: str) -> None:
    if not source_url:
        raise SystemExit("Missing source URL. Set --source-url or SOURCE_DATABASE_URL.")
    if not target_url:
        raise SystemExit("Missing target URL. Set --target-url or TARGET_DATABASE_URL.")
    if not source_url.startswith("mysql"):
        raise SystemExit("Source URL must be a MySQL SQLAlchemy URL.")
    if not target_url.startswith("postgresql"):
        raise SystemExit("Target URL must be a PostgreSQL SQLAlchemy URL.")


def reflect_tables(engine: Engine, table_names: list[str]) -> MetaData:
    metadata = MetaData()
    metadata.reflect(bind=engine, only=table_names)
    return metadata


def available_tables(engine: Engine) -> set[str]:
    return set(inspect(engine).get_table_names())


def chunked(rows: Iterable[dict], batch_size: int) -> Iterable[list[dict]]:
    batch: list[dict] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def decode_json_string(value):
    current = value
    for _ in range(2):
        if not isinstance(current, str):
            break
        text = current.strip()
        if not text:
            break
        try:
            current = json.loads(text)
        except Exception:
            break
    return current


def normalize_row(table_name: str, row: dict, target_columns: set[str], source_columns: set[str]) -> dict:
    normalized: dict = {}
    json_columns = JSON_COLUMNS.get(table_name, set())
    for column in target_columns.intersection(row.keys()):
        value = row[column]
        if column in json_columns:
            value = decode_json_string(value)
        normalized[column] = value

    if table_name == "users" and "password_hash" in target_columns and "password_hash" not in source_columns:
        normalized.setdefault("password_hash", None)

    return normalized


def delete_existing_rows(target_engine: Engine, target_metadata: MetaData, table_names: list[str]) -> None:
    with target_engine.begin() as connection:
        for table_name in reversed(table_names):
            if table_name not in target_metadata.tables:
                continue
            connection.execute(target_metadata.tables[table_name].delete())


def ensure_target_empty_or_truncated(target_engine: Engine, target_metadata: MetaData, table_names: list[str], truncate_target: bool) -> None:
    if truncate_target:
        delete_existing_rows(target_engine, target_metadata, table_names)
        return

    with target_engine.connect() as connection:
        non_empty_tables: list[str] = []
        for table_name in table_names:
            if table_name not in target_metadata.tables:
                continue
            table = target_metadata.tables[table_name]
            count = connection.execute(select(text("count(*)")).select_from(table)).scalar_one()
            if count:
                non_empty_tables.append(f"{table_name}={count}")

    if non_empty_tables:
        raise SystemExit(
            "Target database is not empty for selected tables. "
            "Rerun with --truncate-target if you want to replace existing data. "
            f"Found: {', '.join(non_empty_tables)}"
        )


def copy_table(source_engine: Engine, target_engine: Engine, source_metadata: MetaData, target_metadata: MetaData, table_name: str, batch_size: int) -> int:
    source_table = source_metadata.tables.get(table_name)
    target_table = target_metadata.tables.get(table_name)
    if source_table is None or target_table is None:
        print(f"[skip] {table_name}: missing in source or target")
        return 0

    source_columns = set(source_table.columns.keys())
    target_columns = set(target_table.columns.keys())
    if "password_hash" in target_columns and "password_hash" not in source_columns and table_name == "users":
        print("[warn] users: source has no password_hash column; migrated rows will keep password_hash=NULL")

    inserted = 0
    with source_engine.connect() as source_connection, target_engine.begin() as target_connection:
        result = source_connection.execution_options(stream_results=True).execute(
            select(source_table).order_by(*(source_table.primary_key.columns or source_table.columns[:1]))
        )
        row_iter = (
            normalize_row(table_name, dict(row._mapping), target_columns, source_columns)
            for row in result
        )
        for batch in chunked(row_iter, batch_size):
            target_connection.execute(target_table.insert(), batch)
            inserted += len(batch)

    print(f"[ok] {table_name}: copied {inserted} rows")
    return inserted


def reset_postgres_sequences(target_engine: Engine, target_metadata: MetaData, table_names: list[str]) -> None:
    with target_engine.begin() as connection:
        for table_name in table_names:
            table = target_metadata.tables.get(table_name)
            if table is None or "id" not in table.columns:
                continue

            qualified_table = f'"{table.schema}"."{table.name}"' if table.schema else f'"{table.name}"'
            sequence_name = connection.execute(
                text("SELECT pg_get_serial_sequence(:table_name, 'id')"),
                {"table_name": qualified_table.replace('"', '')},
            ).scalar_one_or_none()
            if not sequence_name:
                continue

            max_id = connection.execute(select(text("coalesce(max(id), 0)")).select_from(table)).scalar_one()
            is_called = max_id > 0
            next_value = max_id if is_called else 1
            connection.execute(
                text("SELECT setval(:sequence_name, :next_value, :is_called)"),
                {
                    "sequence_name": sequence_name,
                    "next_value": next_value,
                    "is_called": is_called,
                },
            )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    table_names = [table.strip() for table in args.tables.split(",") if table.strip()]
    validate_urls(args.source_url, args.target_url)

    print(f"[info] source={mask_url(args.source_url)}")
    print(f"[info] target={mask_url(args.target_url)}")
    print(f"[info] tables={', '.join(table_names)}")
    print(f"[info] batch_size={args.batch_size}")
    print(f"[info] truncate_target={args.truncate_target}")

    source_engine = create_engine(args.source_url, pool_pre_ping=True)
    target_engine = create_engine(args.target_url, pool_pre_ping=True)

    source_table_names = available_tables(source_engine)
    target_table_names = available_tables(target_engine)

    missing_source = [table for table in table_names if table not in source_table_names]
    if missing_source:
        print(f"[warn] source missing tables: {', '.join(missing_source)}")

    missing_target = [table for table in table_names if table not in target_table_names]
    if missing_target:
        raise SystemExit(f"Target is missing tables: {', '.join(missing_target)}. Run alembic upgrade head first.")

    common_tables = [table for table in table_names if table in source_table_names and table in target_table_names]
    source_metadata = reflect_tables(source_engine, common_tables)
    target_metadata = reflect_tables(target_engine, common_tables)

    ensure_target_empty_or_truncated(target_engine, target_metadata, common_tables, args.truncate_target)

    total_inserted = 0
    copied_counts: dict[str, int] = {}
    for table_name in common_tables:
        copied = copy_table(source_engine, target_engine, source_metadata, target_metadata, table_name, args.batch_size)
        copied_counts[table_name] = copied
        total_inserted += copied

    reset_postgres_sequences(target_engine, target_metadata, common_tables)

    print("[done] migration complete")
    for table_name in common_tables:
        print(f"[summary] {table_name}: {copied_counts[table_name]}")
    print(f"[summary] total_rows={total_inserted}")


if __name__ == "__main__":
    main()