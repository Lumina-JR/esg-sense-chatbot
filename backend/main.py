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

# ── Model config ────────────────────────────────────────────────────────
# llama-3.3-70b-versatile was deprecated by Groq. Switched to Qwen 3.6 27B
# (qwen/qwen3.6-27b) — a reasoning model with an explicit thinking mode,
# which exposes the step-by-step analysis process separately from the
# final answer via reasoning_format="parsed". Centralized here so future
# model swaps are one line.
GROQ_MODEL = "qwen/qwen3.6-27b"

# ── Groq client — server-side only. API key lives in Render's env vars,
# never sent to or requested from the browser. ──
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

if not os.getenv("GROQ_API_KEY"):
    # Fail loudly at startup rather than on first request — easier to
    # catch a missing Render env var immediately.
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
# Expected structure (see methodology.pdf for the full template):
#
#   === INDICATOR: S.1.3 ===
#   INDICATOR NAME: Diversity Programmes
#
#   --- TICK-BOX: Initiatives supporting a diverse workforce ---
#   LOOK FOR:
#   - ...
#   DO NOT ACCEPT:
#   - ...
#   EDGE CASES:
#   - ...
#
# Parsed once into: methodology[indicator_code]["tickboxes"][tick_box_name] = guidance_text
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
    doc = fitz.open(path)
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


# ── Sanity check: flag indicator/tick-box pairs that exist in one source
# but not the other, so mismatches surface at startup instead of silently
# producing thin context at query time. ──
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
# session_store[session_id] = {
#     "documents": { filename: extracted_text, ... },
#     "last_indicator": str,
#     "last_tickbox": str,
#     "updated_at": float,
# }
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
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "\n".join(page.get_text("text") for page in doc)
    except Exception as e:
        return f"Error extracting text: {str(e)}"


def get_historical_context(indicator_code: str, tick_box_name: str) -> str:
    """Filter historical citation examples. Empty string if no match terms given."""
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
    """Look up LOOK FOR / DO NOT ACCEPT / EDGE CASES guidance for this pair."""
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

    # No specific tick-box given — return all guidance for the indicator
    sections = [f"[{entry['name']} — {tb}]\n{g}" for tb, g in entry["tickboxes"].items()]
    return "\n\n".join(sections)


def extract_citations(answer_text: str) -> list[dict]:
    """Pull quoted spans + optional page refs out of the model's answer."""
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

    # ── New uploads are ADDED to the session's doc set, not replaced ──
    for f in files:
        content = await f.read()
        if f.filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(content)
        else:
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

    # ── Indicator/tick-box: explicit values win, else fall back to the
    # session's last-used values (follow-up support). ──
    indicator = indicator_name.strip() or session["last_indicator"]
    tickbox = tick_box_name.strip() or session["last_tickbox"]

    session["last_indicator"] = indicator
    session["last_tickbox"] = tickbox
    session["updated_at"] = time.time()

    historical_context = get_historical_context(indicator, tickbox)
    methodology_guidance = get_methodology_guidance(indicator, tickbox)

    context_block = historical_context or "No historical examples available."
    methodology_block = methodology_guidance or "No specific methodology guidance found for this indicator/tick-box."

    combined_docs = "\n\n".join(
        f"=== Document: {name} ===\n{text[:14000]}"
        for name, text in session["documents"].items()
    )

    prompt = f"""You are S.E.N.S.E — a precise, analyst-grade ESG research assistant.

Indicator: {indicator or 'Not specified'}
Tick-box: {tickbox or 'Not specified'}

METHODOLOGY GUIDANCE (follow these rules strictly — what to look for, what
to reject, and how to handle edge cases):
{methodology_block}

HISTORICAL CITATION EXAMPLES (past approved/rejected examples for calibration):
{context_block}

COMPANY DOCUMENT(S):
{combined_docs}

User Request: {query}

Instructions:
- Work through this step by step: first identify what the methodology
  requires for this tick-box, then scan the document for matching evidence,
  then compare what you found against the "DO NOT ACCEPT" list before
  reaching a verdict.
- Apply the methodology guidance above before judging the document. If the
  document only partially satisfies a "LOOK FOR" criterion, say so explicitly
  rather than rounding up to a full match.
- If the company document contains language matching anything under
  "DO NOT ACCEPT" in the methodology, do not treat it as sufficient evidence.
- Always quote **exact text** from the company document, wrapped in
  quotation marks, with page numbers where available, e.g. "exact text" (p. 12).
- Reply with "Yes - Supported", "Partially Supported", or "No - Not found",
  followed by clear reasoning and citations.
- Be strict and professional.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1800,
        reasoning_effort="default",   # "default" = thinking mode on, "none" = off
        reasoning_format="parsed",    # returns thinking in its own field, separate from the answer
    )

    message = response.choices[0].message
    answer = message.content
    reasoning = getattr(message, "reasoning", None) or ""
    citations = extract_citations(answer)

    return {
        "answer": answer,
        "reasoning": reasoning,
        "citations": citations,
    }


@app.delete("/session")
async def clear_session(session_id: str = Form(...)):
    session_store.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


@app.get("/")
async def root():
    return {
        "status": "S.E.N.S.E Backend is running",
        "model": GROQ_MODEL,
        "indicators_loaded": len(methodology_store),
        "historical_rows_loaded": len(esg_df),
    }
