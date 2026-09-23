import os
import re
import time
from typing import Annotated
import pymupdf  # PyMuPDF — use the "pymupdf" module directly; "fitz" is deprecated
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
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

# ── Model config ────────────────────────────────────────────────────────
# Centralized here so future model swaps are one line.
GROQ_MODEL = "qwen/qwen3.6-27b"

# ── Groq client — server-side only. ──
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://groq.com",
)

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError(
        "GROQ_API_KEY is not set. Add it under Render → Environment."
    )

# ── Load historical ESG reference data (kept in git, loaded once) ──
HISTORICAL_XLSX_PATH = "data/historical_esg.xlsx"
esg_df = pd.read_excel(HISTORICAL_XLSX_PATH)

REQUIRED_XLSX_COLS = {"indicator_code", "indicator_name", "tick_box_name", "citation_text"}
missing_cols = REQUIRED_XLSX_COLS - set(esg_df.columns)
if missing_cols:
    raise RuntimeError(f"historical_esg.xlsx is missing required columns: {missing_cols}")


# ── Load and parse methodology PDF (kept in git, parsed once at startup) ──
METHODOLOGY_PDF_PATH = "data/methodology.pdf"

INDICATOR_PATTERN = re.compile(
    r"=== INDICATOR:\s*(.+?)\s*===\s*\n"
    r"INDICATOR NAME:\s*(.+?)\s*\n"
    r"(.*?)(?=\n=== INDICATOR:|\Z)",
    re.DOTALL,
)
TICKBOX_PATTERN = re.compile(
    r"---\s*TICK-BOX:\s*(.+?)\s*---\s*\n"
    r"(.*?)(?=\n---\s*TICK-BOX:|\Z)",
    re.DOTALL,
)


def load_methodology(path: str) -> dict:
    doc = pymupdf.open(path)
    full_text = "\n".join(page.get_text("text") for page in doc)
    doc.close()

    methodology = {}
    for code, name, body in INDICATOR_PATTERN.findall(full_text):
        code = code.strip()
        tickboxes = {}
        for tb_name, tb_body in TICKBOX_PATTERN.findall(body):
            tickboxes[tb_name.strip()] = tb_body.strip()
        methodology[code] = {"name": name.strip(), "tickboxes": tickboxes}
    return methodology


methodology_store = load_methodology(METHODOLOGY_PDF_PATH)

if not methodology_store:
    raise RuntimeError(
        "methodology.pdf parsed to zero indicators. Check the delimiter "
        "format ('=== INDICATOR: CODE ===' / '--- TICK-BOX: NAME ---')."
    )


def _validate_sources_aligned():
    excel_pairs = set(
        zip(esg_df["indicator_code"].astype(str), esg_df["tick_box_name"].astype(str))
    )
    methodology_pairs = {
        (code, tb_name)
        for code, data in methodology_store.items()
        for tb_name in data["tickboxes"]
    }

    excel_only = excel_pairs - methodology_pairs
    methodology_only = methodology_pairs - excel_pairs

    if excel_only:
        print(f"[WARN] {len(excel_only)} indicator/tick-box pairs in Excel "
              f"have no methodology guidance: {sorted(excel_only)[:5]}...")
    if methodology_only:
        print(f"[WARN] {len(methodology_only)} indicator/tick-box pairs in "
              f"methodology.pdf have no historical citations: "
              f"{sorted(methodology_only)[:5]}...")


_validate_sources_aligned()


# ── In-memory session store ──────────────────────────────────────────────
session_store: dict[str, dict] = {}
SESSION_TTL_SECONDS = 60 * 60 * 4  # 4 hours of inactivity


def _prune_expired_sessions():
    now = time.time()
    expired = [sid for sid, s in session_store.items()
               if now - s["updated_at"] > SESSION_TTL_SECONDS]
    for sid in expired:
        session_store.pop(sid, None)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        return "\n".join(page.get_text("text") for page in doc)
    except Exception as e:
        return f"Error extracting text: {str(e)}"


def get_historical_context(indicator_code: str, tick_box_name: str) -> str:
    if not indicator_code and not tick_box_name:
        return ""

    mask = pd.Series(False, index=esg_df.index)
    if indicator_code:
        mask |= esg_df["indicator_code"].astype(str).str.contains(
            re.escape(indicator_code), case=False, na=False
        )
    if tick_box_name:
        mask |= esg_df["tick_box_name"].astype(str).str.contains(
            tick_box_name, case=False, na=False, regex=False
        )

    relevant = esg_df[mask]
    if relevant.empty:
        return ""
    return relevant.to_string(index=False)


def get_methodology_guidance(indicator_code: str, tick_box_name: str) -> str:
    if not indicator_code:
        return ""

    entry = methodology_store.get(indicator_code.strip())
    if not entry:
        return ""

    if tick_box_name:
        tb_lower = tick_box_name.strip().lower()
        for tb_name, guidance in entry["tickboxes"].items():
            if tb_lower in tb_name.lower() or tb_name.lower() in tb_lower:
                return f"[{entry['name']} — {tb_name}]\n{guidance}"
        return ""

    sections = [f"[{entry['name']} — {tb}]\n{g}" for tb, g in entry["tickboxes"].items()]
    return "\n\n".join(sections)


def extract_citations(answer_text: str) -> list[dict]:
    citations = []
    pattern = re.compile(
        r'["“]([^"”]{8,500})["”]'
        r'(?:\s*\(?\s*(?:p\.?|page)\s*(\d+)\)?)?',
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


# ── Updated Chat Endpoint ────────────────────────────────────────────────
@app.post("/chat")
async def chat(
    query: str = Form(...),
    indicator_name: str = Form(""),
    tick_box_name: str = Form(""),
    session_id: str = Form(...),
    file: Annotated[UploadFile | None, File()] = None,  # Fixed: Cleanly optional single file payload
):
    _prune_expired_sessions()

    # Retrieve or create session context
    session = session_store.setdefault(session_id, {
        "documents": {},
        "last_indicator": "",
        "last_tickbox": "",
        "updated_at": time.time(),
    })
    session["updated_at"] = time.time()

    # ── 1. Process Uploaded PDF File ──
    if file and file.filename:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file layout. Only PDF files (.pdf) are permitted.",
            )
        
        try:
            content = await file.read()
            extracted_text = extract_text_from_pdf(content)
            session["documents"][file.filename] = extracted_text
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process file upload: {str(e)}",
            )

    # ── 2. Build Context boundaries ──
    # Combine text from all documents associated with this active session
    uploaded_docs_context = "\n\n".join(
        f"--- DOCUMENT: {fname} ---\n{text}"
        for fname, text in session["documents"].items()
    )

    # Resolve indicator mappings
    indicator_code = ""
    if indicator_name:
        # Match "S.1.3" from strings like "S.1.3 - Diversity Programmes"
        match = re.match(r"^([A-Z]\.\d+(?:\.\d+)?)\b", indicator_name.strip())
        indicator_code = match.group(1) if match else indicator_name.strip()

    methodology_context = get_methodology_guidance(indicator_code, tick_box_name)
    historical_context = get_historical_context(indicator_code, tick_box_name)

    # ── 3. Execute LLM Call ──
    system_prompt = (
        "You are the S.E.N.S.E ESG Assistant, an expert tool assessing compliance. "
        "Analyze the provided uploaded documents alongside the methodology text rules "
        "and historical baseline logs. Answer the user query comprehensively. "
        "When referencing statements from the uploaded document, wrap exact text quotes "
        "in double quotes like \"exact quote snippet here\" and include page numbers if visible."
    )

    user_payload = f"""User Query: {query}

=== METHODOLOGY RULEBOOK MATCH ===
{methodology_context or 'No direct matching methodology criteria loaded for this selection.'}

=== HISTORICAL CITATION BASELINES ===
{historical_context or 'No historical precedent entries found.'}

=== ACTIVE UPLOADED DOCUMENTS ===
{uploaded_docs_context or 'No documents have been uploaded in this session yet.'}
"""

    try:
        # Requesting structure compatible with Reasoning models that have an explicit thinking phase
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload}
            ],
            extra_body={"reasoning_format": "parsed"}  # Exposes Qwen step-by-step thinking track
        )
        
        answer = response.choices[0].message.content
        citations = extract_citations(answer)

        return {
            "answer": answer,
            "citations": citations,
            "session_id": session_id,
            "tracked_files": list(session["documents"].keys())
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream inference engine failure: {str(e)}"
        )
