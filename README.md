# MedScanAI

A simple AI medicine information assistant built with Streamlit, Gemini, and LangGraph.

Upload a photo of a medicine package or strip — MedScanAI extracts the text, identifies
the medicine, researches reliable medical sources, and presents the information in a clear,
easy-to-understand format.

---

## Features

- OCR text extraction with EasyOCR
- Medicine identification with Gemini 3.1 Flash-Lite
- Evidence-grounded research with LangGraph + Tavily
- Source validation and self-correction loop
- Simple, clean Streamlit UI
- Deployable directly to Streamlit Community Cloud

---

## Project structure

```
MedScanAI/
├── app.py                  # Streamlit entry point
├── ocr_service.py          # EasyOCR text extraction
├── medicine_identifier.py  # Gemini medicine identification
├── medicine_agent.py       # LangGraph research agent
├── schemas.py              # Shared Pydantic models
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup (local development)

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd MedScanAI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Copy `.env.example` to `.env` and fill in your keys:

```
GEMINI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key_here
LANGSMITH_PROJECT=MedScan-AI-Agent
```

- **GEMINI_API_KEY** – [Google AI Studio](https://aistudio.google.com/app/apikey)
- **TAVILY_API_KEY** – [Tavily](https://tavily.com)
- **LANGSMITH_API_KEY** – [LangSmith](https://smith.langchain.com) (optional, for tracing)

### 5. Run

```bash
streamlit run app.py
```

---

```toml
GEMINI_API_KEY = "your_key_here"
TAVILY_API_KEY = "your_key_here"
LANGSMITH_TRACING = "true"
LANGSMITH_API_KEY = "your_key_here"
LANGSMITH_PROJECT = "MedScan-AI-Agent"
```

No `packages.txt` is required. EasyOCR is pure Python and has no Linux system-package
dependencies.

---

## Notes

- **EasyOCR first-run download**: On first startup, EasyOCR downloads its English language
  model files (~50 MB). This is one-time and cached automatically.
- **Model**: Only `gemini-3.1-flash-lite` is used.
- **Safety**: MedScanAI is an educational tool. It does not diagnose, prescribe, or
  provide personalized medical advice.

---

## Disclaimer

MedScanAI provides general educational information gathered from publicly available medical
sources. It is not a substitute for advice from a qualified healthcare professional. Always
consult a doctor or pharmacist before starting, stopping, or changing any medication.
