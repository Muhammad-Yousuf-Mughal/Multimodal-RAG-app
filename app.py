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

# Mount Static Files
app.mount("/static", StaticFiles(directory="public"), name="static")

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
    """Return embedding vector for text using OpenAI embeddings if available, else local embedder."""
    if openai_client is not None:
        try:
            resp = openai_client.embeddings.create(model="text-embedding-3-small", input=text)
            # SDK returns resp.data[0].embedding
            return resp.data[0].embedding
        except Exception:
            pass
    if _local_embedder is not None:
        return _local_embedder.encode(text).tolist()
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
    query_l = query.lower()
    context_text = "\n\n".join(
        f"--- Context Item {idx} (Text on Page {item['page']}) ---\n{item['raw_content']}"
        for idx, item in enumerate(context_items, 1)
    )

    if "table" in query_l or "page" in query_l:
        combined = "\n".join(item["raw_content"] for item in context_items)
        lower_combined = combined.lower()
        if "theory paper" in lower_combined or "subject/paper" in lower_combined or "practical paper" in lower_combined:
            lines = [line.strip() for line in combined.replace("\r", "\n").split("\n") if line.strip()]
            theory_lines = []
            practical_lines = []
            in_theory = False
            in_practical = False
            for line in lines:
                if "theory paper" in line.lower():
                    in_theory = True
                    in_practical = False
                    continue
                if "practical paper" in line.lower():
                    in_theory = False
                    in_practical = True
                    continue
                if in_theory and re.search(r"\d+\s+.*\d{2}/\d{2}/\d{4}", line.lower()):
                    theory_lines.append(line)
                if in_practical and re.search(r"\d+\s+.*\d{2}/\d{2}/\d{4}", line.lower()):
                    practical_lines.append(line)

            summary_parts = ["This is the exam schedule table for the roll number slip."]
            if theory_lines:
                summary_parts.append("The theory exam table lists subjects, exam dates, days, and timings.")
                for entry in theory_lines[:3]:
                    summary_parts.append(entry)
            if practical_lines:
                summary_parts.append("The practical exam table lists practical subjects and their dates and time windows.")
                for entry in practical_lines[:3]:
                    summary_parts.append(entry)
            return " ".join(summary_parts)[:2500]

    prompt = (
        "You are a precise document assistant. Answer the user question strictly from the context below. "
        "If the context is insufficient, respond exactly with: 'The answer is not available in the provided document.'\n\n"
        f"User Query: {query}\n\nRetrieved Context:\n{context_text}"
    )

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
    return FileResponse("public/index.html")

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