from __future__ import annotations

import os

os.environ["DEBUG"] = "false"
os.environ["ENVIRONMENT"] = "local"
os.environ["AUTH_SECRET_KEY"] = "test-secret-key-for-unit-tests-only"

from app.services.chat_service import _build_evidence_context, _build_sources


def test_sources_and_prompt_include_optional_section_page_metadata():
    sources = _build_sources([
        {
            "document_id": 1,
            "chunk_id": 2,
            "title": "Doc",
            "score": 0.9,
            "snippet": "Snippet",
            "source_uri": "doc.txt",
            "page_number": 3,
            "section_key": "bai-thuc-hanh-so-4",
            "section_title": "Bài thực hành số 4",
            "section_level": 2,
            "char_start": 10,
            "char_end": 20,
        }
    ])
    assert sources[0]["page_number"] == 3
    assert sources[0]["section_key"] == "bai-thuc-hanh-so-4"
    evidence = _build_evidence_context([{**sources[0], "content": "Section content"}])
    assert "section=Bài thực hành số 4" in evidence
    assert "page=3" in evidence

