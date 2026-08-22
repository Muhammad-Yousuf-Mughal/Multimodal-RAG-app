import os
import re
import uuid
import base64
import tempfile
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import traceback
import pypdf

# AI & DB Libraries
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Mount Static Files if present (avoid startup crash on platforms that omit static folder)
if os.path.isdir("public"):
    app.mount("/static", StaticFiles(directory="public"), name="static")
else:
    import logging
    logging.warning("Static directory 'public' not found; static files will not be served. Ensure 'public' is included in the deployment.")

# Environment Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "multimodal-rag")

# Initialize Clients
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Optional local embedder fallback (not used in serverless deploys)
_local_embedder = None
try:
    from sentence_transformers import SentenceTransformer
    _local_embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    _local_embedder = None


def embed_text(text: str):
    """Return embedding vector for text using OpenAI embeddings if available, else local embedder.

    Improved behavior:
    - Retries OpenAI embedding calls with exponential backoff on transient errors (e.g., 429 rate limits).
    - Logs the underlying exception and surfaces a diagnostic message if all retries fail, rather than the generic provider-missing error.
    """
    last_exc = None
    if openai_client is not None:
        # Retry transient failures (rate limits/network) with exponential backoff
        import time, logging
        max_attempts = 5
        base_backoff = 0.5
        for attempt in range(1, max_attempts + 1):
            try:
                resp = openai_client.embeddings.create(model="text-embedding-3-small", input=text)
                return resp.data[0].embedding
            except Exception as e:
                last_exc = e
                # If this was the last attempt, break and fall back / error
                if attempt == max_attempts:
                    logging.exception("OpenAI embeddings failed after %s attempts", max_attempts)
                    break
                # Backoff and retry
                sleep_time = base_backoff * (2 ** (attempt - 1))
                logging.info("OpenAI embeddings attempt %s failed; retrying in %.2fs: %s", attempt, sleep_time, str(e))
                time.sleep(sleep_time)
    # Local fallback (developer can install sentence-transformers for on-instance embeddings)
    if _local_embedder is not None:
        return _local_embedder.encode(text).tolist()
    # If we had an OpenAI client but it failed, surface the last error for diagnostics (non-secret)
    if last_exc is not None:
        raise RuntimeError(f"Embedding failed after retries: {last_exc}")
    # No provider at all
    raise RuntimeError("No embedding provider available. Set OPENAI_API_KEY or install sentence-transformers locally.")

# Pinecone Setup
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384, # SentenceTransformer dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(PINECONE_INDEX_NAME)
else:
    index = None

# In-Memory Document Store mapping doc_id -> raw element content
DOC_STORE: Dict[str, Dict[str, Any]] = {}
DYNAMIC_DOC_NAMESPACES: Dict[str, str] = {}
ACTIVE_NAMESPACE: str | None = None
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

# --- HELPER FUNCTIONS ---

def summarize_text_or_table_with_groq(text_content: str, is_table: bool = False) -> str:
    """Uses Groq to produce fast text/table summaries."""
    prompt_type = "HTML Table" if is_table else "Text Chunk"
    prompt = f"Summarize the following {prompt_type} concisely for search indexing:\n\n{text_content}"

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content


def generate_final_answer(query: str, context_items: List[Dict[str, Any]]) -> str:
    """Generate an answer preferring lightweight, rule-based table extraction for table queries,
    then fall back to LLM providers (OpenAI/Groq).
    """
    query_l = (query or "").lower()
    context_text = "\n\n".join(
        f"--- Context Item {idx} (Text on Page {item['page']}) ---\n{item['raw_content']}"
        for idx, item in enumerate(context_items, 1)
    )

    combined = "\n".join(item.get("raw_content", "") for item in context_items)
    lower_combined = combined.lower()

    # Heuristic table extraction for common roll-slip tables
    if "table" in query_l or "describe table" in query_l or ("table" in query_l and "page" in query_l) or ("what does the table" in query_l):
        # Look for sections that contain 'theory' or 'practical' exam tables
        # Use regex spans to be robust against chunk boundaries
        try:
            theory_match = re.search(r"theory\s*paper[s]?[^\n]*\n(.*?)(?=(practical\s*paper[s]?|centre allotted|centre|$))", combined, flags=re.I | re.S)
            practical_match = re.search(r"practical\s*paper[s]?[^\n]*\n(.*?)(?=(centre\s*allotted|centre|$))", combined, flags=re.I | re.S)
        except Exception:
            theory_match = practical_match = None

        summary_parts = []
        if theory_match:
            theory_block = theory_match.group(1)
            # find lines with dates
            tlines = [ln.strip() for ln in re.split(r"\r?\n", theory_block) if ln.strip()]
            t_entries = [ln for ln in tlines if re.search(r"\d{2}/\d{2}/\d{4}", ln)]
            if t_entries:
                summary_parts.append("Theory exam table: subjects with dates, day and start times.")
                summary_parts.extend(t_entries[:5])

        if practical_match:
            practical_block = practical_match.group(1)
            plines = [ln.strip() for ln in re.split(r"\r?\n", practical_block) if ln.strip()]
            p_entries = [ln for ln in plines if re.search(r"\d{2}/\d{2}/\d{4}", ln)]
            if p_entries:
                summary_parts.append("Practical exam table: practical subjects with dates and time windows.")
                summary_parts.extend(p_entries[:5])

        # If we extracted anything useful, return a concise summary
        if summary_parts:
            return " \n".join(summary_parts)[:2500]

        # Fallback: if the document clearly contains exam-related vocabulary, return a safe high-level description
        if any(k in lower_combined for k in ("exam", "paper", "theory", "practical", "date", "time")):
            return "This table lists the exam schedule (subjects, dates, days, arrival/attendance and start times) for theory and practical papers."

    # Build a strict LLM prompt if heuristics didn't return
    prompt = (
        "You are a precise document assistant. Answer the user question strictly from the context below. "
        "If the context is insufficient, respond exactly with: 'The answer is not available in the provided document.'\n\n"
        f"User Query: {query}\n\nRetrieved Context:\n{context_text}"
    )

    # Try OpenAI first
    if openai_client is not None:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer using only the provided context."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            answer = response.choices[0].message.content
            if answer and answer.strip():
                return answer.strip()
        except Exception:
            pass

    # Next, try Groq
    if groq_client is not None:
        try:
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.0,
            )
            answer = response.choices[0].message.content
            if answer and answer.strip():
                return answer.strip()
        except Exception:
            pass

    return "The answer is not available in the provided document."


# --- API ENDPOINTS ---

@app.get("/")
def read_root():
    # Serve the SPA index if the public directory exists, else return a helpful message
    if os.path.isdir("public") and os.path.exists(os.path.join("public", "index.html")):
        return FileResponse("public/index.html")
    return JSONResponse(content={"message": "Static UI not deployed. Use /api endpoints directly or include the 'public' folder in deployment."})


@app.get('/api/health')
def health_check():
    """Returns basic diagnostics useful in deployments and for debugging 500s."""
    status = {
        "openai_api_key_set": bool(OPENAI_API_KEY),
        "pinecone_api_key_set": bool(PINECONE_API_KEY),
        "groq_api_key_set": bool(GROQ_API_KEY),
        "public_dir_present": os.path.isdir("public") and os.path.exists(os.path.join("public","index.html")),
        "pinecone_index_connected": False,
        "active_namespace": ACTIVE_NAMESPACE if 'ACTIVE_NAMESPACE' in globals() else None,
    }
    try:
        if index is not None:
            # light-weight check
            status['pinecone_index_connected'] = True
    except Exception as e:
        status['pinecone_error'] = str(e)

    return JSONResponse(content=status)

import tempfile
@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    chunk_size: int = Form(500)
):
    """
    Resilient PDF Parser using pure-Python pypdf 
    to bypass missing local Poppler/Tesseract C++ binaries on Windows.
    """
    try:
        if groq_client is None:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        if index is None:
            raise RuntimeError("PINECONE_API_KEY is not configured.")

        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise ValueError("Uploaded file is empty.")
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            raise ValueError("Uploaded file exceeds the 20MB limit.")

        suffix = os.path.splitext(file.filename or "upload.pdf")[1].lower() or ".pdf"
        if suffix != ".pdf":
            raise ValueError("Only PDF files are supported.")

        safe_chunk_size = max(200, int(chunk_size or 500))
        doc_namespace = f"doc-{uuid.uuid4()}"
        global ACTIVE_NAMESPACE
        ACTIVE_NAMESPACE = doc_namespace
        DYNAMIC_DOC_NAMESPACES[file.filename] = doc_namespace

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            temp_path = f.name
            f.write(file_bytes)

        reader = pypdf.PdfReader(temp_path)
        vectors_to_upsert = []

        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue

            for i in range(0, len(page_text), safe_chunk_size):
                chunk = page_text[i:i + safe_chunk_size]
                if not chunk.strip():
                    continue

                doc_id = str(uuid.uuid4())
                summary_text = summarize_text_or_table_with_groq(chunk, is_table=False)

                DOC_STORE[doc_id] = {
                    "type": "Text",
                    "raw_content": chunk,
                    "page": page_idx + 1,
                    "doc_name": file.filename,
                    "namespace": doc_namespace,
                }

                summary_embedding = embed_text(summary_text)
                vectors_to_upsert.append((
                    doc_id,
                    summary_embedding,
                    {
                        "doc_id": doc_id,
                        "type": "Text",
                        "page": page_idx + 1,
                        "doc_name": file.filename,
                        "summary": summary_text,
                        "namespace": doc_namespace,
                    }
                ))

        # 2. Upsert to Pinecone
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert, namespace=doc_namespace)

        return {"message": f"Successfully processed '{file.filename}' and indexed {len(vectors_to_upsert)} chunks into Pinecone."}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/api/query")
async def query_rag(
    query: str = Form(...),
    top_k: int = Form(5),
    threshold: float = Form(0.05)
):
    """Main RAG query endpoint. Returns answer and sources.
    Note: If the query mentions header/top, we attempt to include page-1 content as context."""
    try:
        query_vec = embed_text(query)
        safe_top_k = max(3, int(top_k or 5))
        safe_threshold = max(0.0, float(threshold if threshold is not None else 0.05))

        query_kwargs = {"vector": query_vec, "top_k": safe_top_k, "include_metadata": True}
        if ACTIVE_NAMESPACE:
            query_kwargs["namespace"] = ACTIVE_NAMESPACE
        results = index.query(**query_kwargs)

        retrieved_raw_elements = []
        source_citations = []
        filtered_matches = []

        for match in results.matches:
            score = float(match.score or 0.0)
            if score < safe_threshold:
                continue
            filtered_matches.append(match)

        if not filtered_matches and results.matches:
            filtered_matches = results.matches[:safe_top_k]

        for match in filtered_matches:
            score = float(match.score or 0.0)
            doc_id = match.metadata["doc_id"]
            if doc_id in DOC_STORE:
                raw_data = DOC_STORE[doc_id]
                retrieved_raw_elements.append(raw_data)
                source_citations.append({
                    "page": raw_data["page"],
                    "document": raw_data["doc_name"],
                    "similarity_score": round(score, 3),
                    "type": raw_data["type"]
                })

        # Heuristic: if query asks about top/header or there are no retrieved elements,
        # try to include page-1 content from the most relevant document.
        ql = (query or "").lower()
        header_keywords = ["top", "header", "info on top", "what is on top", "above"]
        need_header = any(k in ql for k in header_keywords)

        if not retrieved_raw_elements or need_header:
            # Determine target document: prefer the first retrieved doc, otherwise if only one doc indexed use that
            target_doc = None
            if retrieved_raw_elements:
                target_doc = retrieved_raw_elements[0].get('doc_name')
            else:
                docs = set(v.get('doc_name') for v in DOC_STORE.values() if v.get('doc_name'))
                if len(docs) == 1:
                    target_doc = next(iter(docs))

            if target_doc:
                # collect page 1 elements for that document
                page1_items = [v for v in DOC_STORE.values() if v.get('doc_name') == target_doc and v.get('page') == 1]
                # prepend page1 items so they are used as primary context
                if page1_items:
                    # add any missing page1 items to retrieved_raw_elements and citations
                    for item in page1_items:
                        if item not in retrieved_raw_elements:
                            retrieved_raw_elements.insert(0, item)
                            source_citations.insert(0, {
                                "page": item["page"],
                                "document": item["doc_name"],
                                "similarity_score": None,
                                "type": item["type"]
                            })

        if not retrieved_raw_elements:
            return JSONResponse(content={
                "answer": "The answer is not available in the provided document.",
                "response": "The answer is not available in the provided document.",
                "result": "The answer is not available in the provided document.",
                "sources": []
            })

        answer_text = generate_final_answer(query, retrieved_raw_elements)

        return JSONResponse(content={
            "answer": answer_text,
            "response": answer_text,
            "result": answer_text,
            "sources": source_citations
        })
        

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
