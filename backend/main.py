from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import fitz  # PyMuPDF
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="S.E.N.S.E ESG Assistant")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load historical ESG data
esg_df = pd.read_excel("backend/data/historical_esg.xlsx")

# Groq client (Llama models)
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),           # ← Change env var name
    base_url="https://api.groq.com/openai/v1"    # ← Groq endpoint
)

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "\n".join([page.get_text("text") for page in doc])
    except Exception as e:
        return f"Error extracting text: {str(e)}"

@app.post("/chat")
async def chat(
    query: str = Form(...),
    indicator_name: str = Form(""),
    tick_box_name: str = Form(""),
    files: list[UploadFile] = File(...),
    session_id: str = Form(None)
):
    if not files:
        raise HTTPException(400, "Please upload at least one document")

    # Extract text from uploaded document
    content = await files[0].read()
    doc_text = extract_text_from_pdf(content)

    # Retrieve relevant historical examples from Excel
    relevant = esg_df[
        (esg_df['indicator_name'].str.contains(indicator_name, case=False, na=False) |
         esg_df['tick_box_name'].str.contains(tick_box_name, case=False, na=False))
    ]
    context = relevant.to_string(index=False) if not relevant.empty else "No historical examples available."

    prompt = f"""You are S.E.N.S.E — a precise, analyst-grade ESG research assistant.

Indicator: {indicator_name or 'Not specified'}
Tick-box: {tick_box_name or 'Not specified'}

Historical guidance & approved citation examples:
{context}

Company Document:
{doc_text[:14000]}

User Request: {query}

Instructions:
- Check if the tick-box is supported in the document.
- Always quote **exact text** from the company document.
- Use page numbers if available.
- Reply with "Yes - Supported" or "No - Not found" + clear reasoning + citations.
- Be strict and professional.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # Fast & strong Llama model on Groq
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1800
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "citations": []   # Can be enhanced later
    }

@app.get("/")
async def root():
    return {"status": "S.E.N.S.E Backend (Groq Llama) is running"}
