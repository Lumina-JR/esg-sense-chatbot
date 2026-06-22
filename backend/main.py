import os
import re
import time
import fitz  # PyMuPDF
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="S.E.N.S.E ESG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load historical ESG reference data (kept in git, loaded once at startup) ──
esg_df = pd.read_excel("backend/data/historical_esg.xlsx")

# ── Groq client (OpenAI-compatible) ──
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# ── In-memory session store ──────────────────────────────────────────────
# Keyed by session_id. Holds the last uploaded doc(s) and last indicator/
# tick-box used, so follow-up questions don't require re-uploading or
# re-specifying context. This resets on backend restart — fine for now,
# swap for Redis if you need persistence across deploys.
#
# session_store[session_id] = {
#     "documents": { filename: extracted_text, ... },
#     "last_indicator": str,
#     "last_tickbox": str,
#     "updated_at": float,
# }
session_store: dict[str, dict] = {}

SESSION_TTL_SECONDS = 60 * 60 * 4  # 4 hours of inactivity → session is dropped


def _prune_expired_sessions():
    now = time.time()
    expired = [
        sid for sid, s in session_store.items()
        if now - s["updated_at"] > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        session_store.pop(sid, None)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "\n".join(page.get_text("text") for page in doc)
    except Exception as e:
        return f"Error extracting text: {str(e)}"


def get_historical_context(indicator_name: str, tick_box_name: str) -> str:
    """Filter historical ESG examples. Returns empty string if nothing matches."""
    if not indicator_name and not tick_box_name:
        return ""

    mask = pd.Series(False, index=esg_df.index)
    if indicator_name:
        mask |= esg_df["indicator_name"].str.contains(indicator_name, case=False, na=False)
    if tick_box_name:
        mask |= esg_df["tick_box_name"].str.contains(tick_box_name, case=False, na=False)

    relevant = esg_df[mask]
    if relevant.empty:
        return ""
    return relevant.to_string(index=False)


def extract_citations(answer_text: str) -> list[dict]:
    """
    Pulls quoted spans out of the model's answer and returns them as
    structured citation objects. Looks for:
      - "double-quoted text" (straight or curly quotes)
      - optional trailing page reference like (p. 12) or (page 12)
    """
    citations = []

    # Matches "..." or “...” possibly followed by a page reference
    pattern = re.compile(
        r'["“]([^"”]{8,500})["”]'          # the quoted span (min 8 chars to skip noise)
        r'(?:\s*\(?\s*(?:p\.?|page)\s*(\d+)\)?)?',  # optional page number
        re.IGNORECASE,
    )

    for match in pattern.finditer(answer_text):
        quote = match.group(1).strip()
        page = match.group(2)
        if not quote:
            continue
        citations.append({
            "text": quote,
            "source": "Uploaded document",
            "page": int(page) if page else None,
        })

    return citations


@app.post("/chat")
async def chat(
    query: str = Form(...),
    indicator_name: str = Form(""),
    tick_box_name: str = Form(""),
    session_id: str = Form(...),
    files: list[UploadFile] = File(default=[]),
):
    _prune_expired_sessions()

    session = session_store.setdefault(session_id, {
        "documents": {},
        "last_indicator": "",
        "last_tickbox": "",
        "updated_at": time.time(),
    })

    # ── Handle new uploads (if any) — these are ADDED to the session's
    # document set, not a replacement, so the analyst can bring in
    # multiple company documents over the course of a session. ──
    for f in files:
        content = await f.read()
        if f.filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(content)
        else:
            # best-effort plain decode for txt/csv etc.
            try:
                text = content.decode("utf-8", errors="ignore")
            except Exception:
                text = ""
        session["documents"][f.filename] = text

    if not session["documents"]:
        raise HTTPException(
            400,
            "No document available for this session. Please upload at least one document.",
        )

    # ── Resolve indicator/tick-box: explicit values win, otherwise fall
    # back to whatever was last used in this session (follow-up support).
    # Historical context only applies when we actually have a value —
    # by design, a bare follow-up with no indicator/tick-box at all (even
    # from history) simply skips historical context. ──
    indicator = indicator_name.strip() or session["last_indicator"]
    tickbox = tick_box_name.strip() or session["last_tickbox"]

    session["last_indicator"] = indicator
    session["last_tickbox"] = tickbox
    session["updated_at"] = time.time()

    historical_context = get_historical_context(indicator, tickbox)
    context_block = historical_context or "No historical examples available."

    # ── Combine all documents currently held in this session ──
    combined_docs = "\n\n".join(
        f"=== Document: {name} ===\n{text[:14000]}"
        for name, text in session["documents"].items()
    )

    prompt = f"""You are S.E.N.S.E — a precise, analyst-grade ESG research assistant.

Indicator: {indicator or 'Not specified'}
Tick-box: {tickbox or 'Not specified'}

Historical guidance & approved citation examples:
{context_block}

Company Document(s):
{combined_docs}

User Request: {query}

Instructions:
- Check if the tick-box is supported in the document.
- Always quote **exact text** from the company document, wrapped in quotation marks.
- Include page numbers in parentheses after each quote where available, e.g. "exact text" (p. 12).
- Reply with "Yes - Supported" or "No - Not found" + clear reasoning + citations.
- Be strict and professional.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1800,
    )

    answer = response.choices[0].message.content
    citations = extract_citations(answer)

    return {
        "answer": answer,
        "citations": citations,
    }


@app.delete("/session")
async def clear_session(session_id: str = Form(...)):
    session_store.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


@app.get("/")
async def root():
    return {"status": "S.E.N.S.E Backend (Groq Llama) is running"}
