import logging
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
import inngest
import inngest.fast_api
from dotenv import load_dotenv
from data_loader import load_and_chunk_pdf, embed_texts, client
from vector_db import QdrantStorage
from custom_types import RAQQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc


load_dotenv()

LLM_MODEL = "gemini-3.6-flash"


def _answer(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(f"- {c}" for c in contexts)
    prompt = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely using the context above. "
        "If the context doesn't contain the answer, say so."
    )
    resp = client.models.generate_content(model=LLM_MODEL, contents=prompt)
    return resp.text


inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    retries=0,
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, name=f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
    retries=0,
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k),
        output_type=RAGSearchResult,
    )
    answer = await ctx.step.run("llm-answer", lambda: _answer(question, found.contexts))

    return RAQQueryResult(
        answer=answer,
        sources=found.sources,
        num_contexts=len(found.contexts),
    ).model_dump()


app = FastAPI()


@app.get("/query")
def query(question: str, top_k: int = 5):
    """Search the filings and have Gemini write the answer."""
    query_vec = embed_texts([question])[0]
    found = QdrantStorage().search(query_vec, top_k)
    answer = _answer(question, found["contexts"])
    return {
        "answer": answer,
        "sources": found["sources"],
        "num_contexts": len(found["contexts"]),
    }


@app.get("/search")
def search(question: str, top_k: int = 5):
    """Retrieval only: matching chunks and sources, no LLM call.
    The stock agent calls this and writes the final answer itself."""
    query_vec = embed_texts([question])[0]
    return QdrantStorage().search(query_vec, top_k)


@app.post("/ingest")
def ingest(pdf_path: str, source_id: str | None = None):
    """Start ingesting a PDF that already exists on this PC."""
    inngest_client.send_sync(
        inngest.Event(
            name="rag/ingest_pdf",
            data={"pdf_path": pdf_path, "source_id": source_id or pdf_path},
        )
    )
    return {"status": "ingest started", "source_id": source_id or pdf_path}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Pick a PDF with the file picker, save it to ./docs, and start ingesting it."""
    Path("docs").mkdir(exist_ok=True)
    dest = Path("docs") / file.filename
    dest.write_bytes(await file.read())
    await inngest_client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={"pdf_path": str(dest.resolve()), "source_id": file.filename},
        )
    )
    return {"status": "ingest started", "source_id": file.filename}


inngest.fast_api.serve(app, inngest_client, functions=[rag_ingest_pdf, rag_query_pdf_ai])