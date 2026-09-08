import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from . import db, pipeline

app = FastAPI(title="Superjoin Fact Knowledge Layer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init_db()

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            500,
            "ANTHROPIC_API_KEY is not set on the server. Export it before starting the app.",
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = pipeline.process_pdf(tmp_path, file.filename)
    finally:
        os.unlink(tmp_path)

    return result


@app.get("/api/documents")
def list_documents():
    return db.list_documents()


@app.get("/api/facts")
def list_facts(document_id: int | None = Query(default=None)):
    if document_id is not None:
        return db.facts_for_document(document_id)
    return db.all_facts()


@app.get("/api/facts/{fact_id}")
def get_fact(fact_id: int):
    fact = db.get_fact(fact_id)
    if not fact:
        raise HTTPException(404, "Fact not found")
    return fact


@app.get("/api/relationships")
def list_relationships(relationship: str | None = Query(default=None)):
    rels = db.list_relationships(relationship)
    # enrich with the actual facts so the frontend doesn't need N+1 calls
    enriched = []
    for r in rels:
        fa = db.get_fact(r["fact_id_a"])
        fb = db.get_fact(r["fact_id_b"])
        enriched.append({**r, "fact_a": fa, "fact_b": fb})
    return enriched


@app.get("/api/issues")
def list_issues():
    return db.list_issues()


@app.get("/api/summary")
def summary():
    docs = db.list_documents()
    facts_all = db.all_facts()
    rels = db.list_relationships()
    by_rel = {}
    for r in rels:
        by_rel[r["relationship"]] = by_rel.get(r["relationship"], 0) + 1
    return {
        "documents": len(docs),
        "facts": len(facts_all),
        "relationships": by_rel,
        "issues": len(db.list_issues()),
    }
