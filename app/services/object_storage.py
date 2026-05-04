from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from time import perf_counter

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _sanitize_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", (value or "").strip())
    normalized = normalized.strip(".-")
    return normalized or "artifact"


def _local_storage_root() -> Path:
    return Path(settings.object_storage_local_path).expanduser().resolve()


def _normalized_endpoint_url() -> str | None:
    endpoint = (settings.object_storage_endpoint or "").strip()
    if not endpoint:
        return None
    if endpoint.startswith(("http://", "https://")):
        return endpoint
    scheme = "https" if settings.object_storage_secure else "http"
    return f"{scheme}://{endpoint}"


def build_document_artifact_key(
    owner_username: str,
    document_id: int,
    artifact_kind: str,
    artifact_name: str,
) -> str:
    owner_part = _sanitize_component(owner_username)
    kind_part = _sanitize_component(artifact_kind)
    name_part = _sanitize_component(artifact_name)
    return f"knowledge/{owner_part}/documents/{document_id}/{kind_part}/{name_part}"


def build_document_artifact_prefix(owner_username: str, document_id: int) -> str:
    owner_part = _sanitize_component(owner_username)
    return f"knowledge/{owner_part}/documents/{document_id}/"


def is_object_storage_enabled() -> bool:
    return (settings.object_storage_provider or "").strip().lower() not in {"", "none", "disabled"}


def _get_s3_client():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for s3/minio object storage support.") from exc

    return boto3.client(
        "s3",
        endpoint_url=_normalized_endpoint_url(),
        aws_access_key_id=settings.object_storage_access_key or None,
        aws_secret_access_key=settings.object_storage_secret_key or None,
        region_name=settings.object_storage_region or None,
        use_ssl=settings.object_storage_secure,
    )


def check_object_storage_health() -> dict:
    started_at = perf_counter()
    provider = (settings.object_storage_provider or "local").strip().lower()
    dependency = "minio" if provider == "minio" else "object_storage"
    base = {
        "ok": False,
        "dependency": dependency,
        "provider": provider,
        "bucket": settings.object_storage_bucket,
        "endpoint": _normalized_endpoint_url(),
    }

    if not is_object_storage_enabled():
        return {
            **base,
            "latency_ms": round((perf_counter() - started_at) * 1000, 2),
            "detail": "Object storage is disabled.",
        }

    if provider == "local":
        root = _local_storage_root()
        return {
            **base,
            "ok": True,
            "root_path": str(root),
            "latency_ms": round((perf_counter() - started_at) * 1000, 2),
        }

    if provider in {"s3", "minio"}:
        try:
            from botocore.exceptions import ClientError
        except ImportError as exc:
            return {
                **base,
                "latency_ms": round((perf_counter() - started_at) * 1000, 2),
                "detail": str(exc),
            }

        try:
            client = _get_s3_client()
            client.list_buckets()

            bucket_exists = None
            try:
                client.head_bucket(Bucket=settings.object_storage_bucket)
                bucket_exists = True
            except ClientError:
                bucket_exists = False

            return {
                **base,
                "ok": True,
                "bucket_exists": bucket_exists,
                "latency_ms": round((perf_counter() - started_at) * 1000, 2),
            }
        except Exception as exc:
            return {
                **base,
                "latency_ms": round((perf_counter() - started_at) * 1000, 2),
                "detail": str(exc),
            }

    return {
        **base,
        "latency_ms": round((perf_counter() - started_at) * 1000, 2),
        "detail": f"Unsupported object storage provider: {settings.object_storage_provider}",
    }


def store_document_artifact(
    owner_username: str,
    document_id: int,
    artifact_kind: str,
    artifact_name: str,
    content: bytes,
    *,
    content_type: str | None = None,
) -> dict:
    provider = (settings.object_storage_provider or "local").strip().lower()
    key = build_document_artifact_key(owner_username, document_id, artifact_kind, artifact_name)

    if provider == "local":
        destination = _local_storage_root() / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return {
            "provider": provider,
            "bucket": settings.object_storage_bucket,
            "key": key,
            "uri": f"local://{settings.object_storage_bucket}/{key}",
            "content_type": content_type,
            "size_bytes": len(content),
        }

    if provider in {"s3", "minio"}:
        try:
            from botocore.exceptions import ClientError
        except ImportError as exc:
            raise RuntimeError("boto3 is required for s3/minio object storage support.") from exc

        client = _get_s3_client()
        bucket = settings.object_storage_bucket

        try:
            client.head_bucket(Bucket=bucket)
        except ClientError:
            create_kwargs: dict = {"Bucket": bucket}
            if settings.object_storage_region and settings.object_storage_region != "us-east-1":
                create_kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": settings.object_storage_region,
                }
            client.create_bucket(**create_kwargs)

        put_kwargs: dict = {
            "Bucket": bucket,
            "Key": key,
            "Body": content,
        }
        if content_type:
            put_kwargs["ContentType"] = content_type
        client.put_object(**put_kwargs)
        return {
            "provider": provider,
            "bucket": bucket,
            "key": key,
            "uri": f"s3://{bucket}/{key}",
            "content_type": content_type,
            "size_bytes": len(content),
        }

    raise ValueError(f"Unsupported object storage provider: {settings.object_storage_provider}")


def delete_document_artifacts(owner_username: str, document_id: int) -> dict:
    provider = (settings.object_storage_provider or "local").strip().lower()
    prefix = build_document_artifact_prefix(owner_username, document_id)

    if provider == "local":
        destination = _local_storage_root() / prefix
        deleted_keys = 0
        if destination.exists():
            deleted_keys = sum(1 for path in destination.rglob("*") if path.is_file())
            shutil.rmtree(destination, ignore_errors=True)
        return {
            "provider": provider,
            "bucket": settings.object_storage_bucket,
            "prefix": prefix,
            "deleted_keys": deleted_keys,
        }

    if provider in {"s3", "minio"}:
        try:
            from botocore.exceptions import ClientError
        except ImportError as exc:
            raise RuntimeError("boto3 is required for s3/minio object storage support.") from exc

        client = _get_s3_client()
        deleted_keys = 0
        continuation_token: str | None = None

        while True:
            list_kwargs = {
                "Bucket": settings.object_storage_bucket,
                "Prefix": prefix,
            }
            if continuation_token:
                list_kwargs["ContinuationToken"] = continuation_token

            try:
                response = client.list_objects_v2(**list_kwargs)
            except ClientError as exc:
                error_code = str(exc.response.get("Error", {}).get("Code") or "")
                if error_code in {"NoSuchBucket", "404"}:
                    break
                raise

            objects = [{"Key": item["Key"]} for item in (response.get("Contents") or []) if item.get("Key")]
            if objects:
                client.delete_objects(
                    Bucket=settings.object_storage_bucket,
                    Delete={"Objects": objects, "Quiet": True},
                )
                deleted_keys += len(objects)

            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")

        return {
            "provider": provider,
            "bucket": settings.object_storage_bucket,
            "prefix": prefix,
            "deleted_keys": deleted_keys,
        }

    raise ValueError(f"Unsupported object storage provider: {settings.object_storage_provider}")