"""
Re-upload test_assignments.docx with the fixed DOCX extractor.
Deletes any existing document with the same filename, then re-ingests.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from app.crud import crud_knowledge
from app.services.knowledge_service import ingest_uploaded_file, extract_text_from_file
from app.services.retrieval_service import search_knowledge

DOCX_PATH = os.path.join(os.path.dirname(__file__), "test_assignments.docx")
TEST_USER = "test_user"
QUERIES = [
    "Phân tích bài tập 7.1",
    "bài tập 7.1 yêu cầu gì",
    "SELECT CONCAT employees offices",
    "bài tập bổ sung 7",
]


def sep(title=""):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


db = SessionLocal()
try:
    # 1. Delete old documents with same filename
    sep("Xóa tài liệu cũ")
    existing_docs = crud_knowledge.list_documents(db, TEST_USER, skip=0, limit=200)
    for doc in existing_docs:
        if "test_assignments" in (doc.title or "").lower():
            # Check raw_text quality
            raw_len = len(doc.raw_text or "")
            print(f"  Doc #{doc.id}: '{doc.title}' raw_text={raw_len} chars → {'xóa (quá ngắn)' if raw_len < 100 else 'xóa (re-upload mới)'}")
            crud_knowledge.hard_delete_document(db, doc.id)

    # 2. Re-upload with new extractor
    sep("Upload lại với extractor mới")
    with open(DOCX_PATH, "rb") as f:
        content = f.read()

    raw_text = extract_text_from_file(content, "test_assignments.docx")
    print(f"Extracted: {len(raw_text)} chars")
    print(f"Preview: {raw_text[:300]!r}")

    result = ingest_uploaded_file(
        db=db,
        owner_username=TEST_USER,
        filename="test_assignments.docx",
        content=content,
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    print(f"\nIngest result: {result}")
    doc_id = result["document_id"]

    # 3. Test search queries
    sep("Search test")
    for query in QUERIES:
        r = search_knowledge(db=db, owner_username=TEST_USER, query=query, top_k=3, document_id=doc_id)
        print(f"\n  Query: {query!r}")
        print(f"  → returned={r['returned']} evidence={r['evidence_strength']} fallback={r['fallback_used']}")
        if r["results"]:
            top = r["results"][0]
            print(f"  → top score={top['score']:.4f} lex={top['lexical_score']:.4f} sem={top['semantic_score']:.4f}")
            print(f"  → snippet: {top['snippet']!r}")
        else:
            print("  → Không có kết quả!")

    sep("XONG – Tài liệu đã sẵn sàng")
    print(f"Document ID: {doc_id}")
    print(f"Chunks: {result['chunks_count']}")
    print("\nHãy chat với câu hỏi 'Phân tích bài tập 7.1' và chọn document này.")

finally:
    db.close()

