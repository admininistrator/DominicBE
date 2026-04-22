"""
Diagnostic test: upload test_assignments.docx, then search 'Phân tích bài tập 7.1'
and verify the full RAG pipeline works end-to-end.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from app.crud import crud_knowledge
from app.services.knowledge_service import (
    extract_text_from_file,
    chunk_text,
    normalize_text_for_ingestion,
    compute_text_embedding,
    ingest_uploaded_file,
    _execute_indexing,
)
from app.services.retrieval_service import search_knowledge, _normalize_for_search, _lexical_overlap_score

DOCX_PATH = os.path.join(os.path.dirname(__file__), "test_assignments.docx")
TEST_USER = "test_user"
QUERY = "Phân tích bài tập 7.1"


def sep(title=""):
    print(f"\n{'='*60}")
    if title:
        print(f"  {title}")
        print('='*60)


def run():
    sep("STEP 1: Extract text from DOCX")
    with open(DOCX_PATH, "rb") as f:
        content = f.read()
    print(f"File size: {len(content)} bytes")

    try:
        raw_text = extract_text_from_file(content, "test_assignments.docx")
    except Exception as e:
        print(f"[ERROR] extract_text_from_file failed: {e}")
        print("  → Hãy cài: pip install python-docx")
        return

    print(f"Extracted text length: {len(raw_text)} chars")
    print(f"First 500 chars:\n{raw_text[:500]}")

    sep("STEP 2: Normalize + Chunk")
    normalized = normalize_text_for_ingestion(raw_text)
    print(f"Normalized length: {len(normalized)} chars")
    chunks = chunk_text(normalized)
    print(f"Total chunks: {len(chunks)}")
    for i, c in enumerate(chunks[:5]):
        print(f"  Chunk {i}: {len(c['content'])} chars, ~{c['token_count']} tokens | {c['content'][:80]!r}...")

    sep("STEP 3: Check query tokenization")
    norm_q = _normalize_for_search(QUERY)
    print(f"Query normalized: {norm_q!r}")

    # Check which chunks contain relevant tokens
    relevant = []
    for c in chunks:
        score = _lexical_overlap_score(QUERY, c["content"])
        if score > 0:
            relevant.append((score, c["chunk_index"], c["content"][:120]))
    relevant.sort(reverse=True)
    print(f"\nChunks with lexical overlap > 0: {len(relevant)}")
    for score, idx, preview in relevant[:5]:
        print(f"  chunk_{idx}: score={score:.4f} | {preview!r}")

    if not relevant:
        print("\n[WARN] Không có chunk nào khớp với query!")
        print("  Thử tìm từ khóa 'bai tap' hoặc '7' trong chunks:")
        for c in chunks:
            if "7" in c["content"] or "bài tập" in c["content"].lower() or "bai tap" in c["content"].lower():
                print(f"  chunk_{c['chunk_index']}: {c['content'][:150]!r}")

    sep("STEP 4: Ingest into DB")
    db = SessionLocal()
    try:
        result = ingest_uploaded_file(
            db=db,
            owner_username=TEST_USER,
            filename="test_assignments.docx",
            content=content,
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        print(f"Ingest result: {result}")
        doc_id = result["document_id"]
        job_id = result["job_id"]

        sep("STEP 5: Verify chunks in DB")
        db_chunks = crud_knowledge.get_chunks_by_document(db, doc_id)
        print(f"Chunks in DB: {len(db_chunks)}")
        for ch in db_chunks[:3]:
            print(f"  chunk_{ch.chunk_index}: {len(ch.content)} chars | {ch.content[:80]!r}")

        sep("STEP 6: Search 'Phân tích bài tập 7.1'")
        search_result = search_knowledge(
            db=db,
            owner_username=TEST_USER,
            query=QUERY,
            top_k=5,
            document_id=doc_id,
        )
        print(f"Returned: {search_result['returned']} results")
        print(f"Evidence strength: {search_result['evidence_strength']}")
        print(f"Fallback used: {search_result['fallback_used']} ({search_result.get('fallback_reason')})")
        print(f"Rewritten query: {search_result.get('rewritten_query')!r}")
        print(f"Candidate count: {search_result.get('candidate_count')}")

        if search_result["results"]:
            print("\nTop results:")
            for r in search_result["results"]:
                print(f"  score={r['score']:.4f} sem={r['semantic_score']:.4f} lex={r['lexical_score']:.4f}")
                print(f"  snippet: {r['snippet']!r}")
        else:
            print("\n[WARN] Không có kết quả search!")
            sep("STEP 6b: Debug – global search (no doc_id filter)")
            global_result = search_knowledge(
                db=db,
                owner_username=TEST_USER,
                query=QUERY,
                top_k=5,
            )
            print(f"Global search returned: {global_result['returned']}")

            sep("STEP 6c: Embedding similarity debug")
            from app.services.retrieval_service import _cosine_similarity
            query_emb = compute_text_embedding(QUERY)
            print(f"Query embedding dims: {len(query_emb)}, non-zero: {sum(1 for x in query_emb if x != 0)}")
            for ch in db_chunks[:5]:
                meta = ch.metadata_json or {}
                emb = meta.get("embedding", [])
                if emb:
                    sim = _cosine_similarity(query_emb, [float(x) for x in emb])
                    lex = _lexical_overlap_score(QUERY, ch.content)
                    print(f"  chunk_{ch.chunk_index}: cosine={sim:.4f} lexical={lex:.4f} | {ch.content[:80]!r}")

            sep("STEP 6d: Score thresholds")
            from app.core.config import settings
            print(f"retrieval_min_score: {settings.retrieval_min_score}")
            print(f"retrieval_min_lexical_score: {settings.retrieval_min_lexical_score}")
            print(f"retrieval_low_confidence_score: {settings.retrieval_low_confidence_score}")

    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()
    finally:
        db.close()

    sep("DONE")


if __name__ == "__main__":
    run()

