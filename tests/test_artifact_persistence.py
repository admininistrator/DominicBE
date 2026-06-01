from __future__ import annotations

import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.chat_models import ChatArtifact, ChatSession, Message
from app.services.artifacts.persistence import (
    ArtifactRecord,
    SqlArtifactPersistenceService,
    artifact_response_to_record,
    list_artifacts_by_message_ids,
    persist_artifact_responses,
)


def _sqlite_session(tmp_path):
    db_path = tmp_path / "artifacts.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(
        bind=engine,
        tables=[ChatSession.__table__, Message.__table__, ChatArtifact.__table__],
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    db.add(ChatSession(id=1, username="tester", title="Artifacts"))
    db.add(
        Message(
            id=10,
            session_id=1,
            request_id="req-artifacts",
            sender_username="tester",
            role="assistant",
            content="Here is the diagram",
            status="success",
        )
    )
    db.commit()
    return engine, db


def test_artifact_response_to_record_extracts_content_elements_and_metadata():
    scene = {
        "type": "excalidraw",
        "version": 2,
        "elements": [{"type": "rectangle", "id": "box", "x": 0, "y": 0, "width": 100, "height": 50}],
    }
    artifact = {
        "id": "art_1",
        "type": "excalidraw",
        "title": "Architecture",
        "content": json.dumps(scene),
        "preview_url": "https://example.test/preview.png",
        "metadata": {"tool_server": "excalidraw", "checkpoint_id": "cp_1"},
    }

    record = artifact_response_to_record(artifact, session_id=1, message_id=10)

    assert record is not None
    assert record.id == "art_1"
    assert record.kind == "excalidraw"
    assert record.title == "Architecture"
    assert json.loads(record.content_json)["type"] == "excalidraw"
    assert json.loads(record.elements_json)[0]["id"] == "box"
    assert record.metadata["tool_server"] == "excalidraw"
    assert record.metadata["checkpoint_id"] == "cp_1"
    assert record.metadata["preview_url"] == "https://example.test/preview.png"


def test_sql_artifact_persistence_round_trips_public_artifact_response(tmp_path):
    engine, db = _sqlite_session(tmp_path)
    try:
        scene = {
            "type": "excalidraw",
            "version": 2,
            "elements": [{"type": "text", "id": "label", "x": 5, "y": 6, "text": "API"}],
        }
        artifact = {
            "id": "art_round_trip",
            "type": "excalidraw",
            "title": "Round trip",
            "content": json.dumps(scene),
            "url": "https://excalidraw.com/#json=abc",
            "metadata": {"tool_server": "excalidraw"},
        }

        saved = persist_artifact_responses(db, session_id=1, message_id=10, artifacts=[artifact])
        loaded = list_artifacts_by_message_ids(db, [10])

        assert [record.id for record in saved] == ["art_round_trip"]
        assert loaded[10][0]["id"] == "art_round_trip"
        assert loaded[10][0]["type"] == "excalidraw"
        assert loaded[10][0]["title"] == "Round trip"
        assert loaded[10][0]["url"] == "https://excalidraw.com/#json=abc"
        assert loaded[10][0]["metadata"] == {"tool_server": "excalidraw"}
        assert json.loads(loaded[10][0]["content"])["elements"][0]["id"] == "label"
    finally:
        db.close()
        engine.dispose()


def test_sql_artifact_persistence_synthesizes_excalidraw_scene_from_elements_json(tmp_path):
    engine, db = _sqlite_session(tmp_path)
    try:
        service = SqlArtifactPersistenceService()
        service.save_many(
            db,
            [
                ArtifactRecord(
                    id="art_elements_only",
                    session_id=1,
                    message_id=10,
                    kind="excalidraw",
                    title="Elements only",
                    elements_json=json.dumps([
                        {"type": "rectangle", "id": "synth", "x": 1, "y": 2, "width": 3, "height": 4}
                    ]),
                    metadata={"type": "excalidraw"},
                )
            ],
        )

        loaded = service.list_by_message_ids(db, [10])
        scene = json.loads(loaded[10][0]["content"])

        assert loaded[10][0]["id"] == "art_elements_only"
        assert scene["type"] == "excalidraw"
        assert scene["elements"][0]["id"] == "synth"
        assert scene["appState"]["viewBackgroundColor"] == "#ffffff"
    finally:
        db.close()
        engine.dispose()
