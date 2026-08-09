"""Medicine identification module for MedScanAI.

Sends OCR-extracted text to Gemini and returns a structured MedicineIdentity.
The LLM is initialized once and cached for the lifetime of the Streamlit session.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from schemas import Ingredient, MedicineIdentity  # noqa: F401 (re-exported)

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3.1-flash-lite"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

MEDICINE_IDENTITY_PROMPT = """
You are the Medicine Identification module of MedScan AI.

Your job is ONLY to identify the medicine from OCR text extracted
from a photograph of medicine packaging.

You are NOT responsible for explaining the medicine, its uses,
dosage, side effects, interactions, precautions, or treatment.

The OCR text may contain errors and irrelevant information.

Your output must be based ONLY on information that can reasonably
be established from the supplied packaging text.

============================================================
1. BRAND NAME
============================================================

Identify the commercial/brand name of the medicine.

Examples:

Dolo-650
Amoxil-500
Brophyle-N
Crocin
Augmentin 625

Do not confuse manufacturer, marketer, distributor, pharmacy, retailer,
company name, or slogan with the medicine brand name.

If the brand cannot be established reliably, return null.

============================================================
2. ACTIVE INGREDIENTS
============================================================

Extract ALL active pharmaceutical ingredients.

Look for packaging phrases such as:

- Each tablet contains
- Each capsule contains
- Composition
- Active ingredient
- Each 5 mL contains
- Contains
- Each film-coated tablet contains

IMPORTANT:

Do not treat excipients, preservatives, colours, flavours, inactive
ingredients, or manufacturer names as active ingredients.

If multiple active ingredients are present, extract ALL of them.

============================================================
3. STRENGTH
============================================================

Extract the strength corresponding to each active ingredient.

Only associate a strength with an ingredient when the packaging
evidence supports the relationship.

============================================================
4. DOSAGE FORM
============================================================

Identify the dosage form when supported by the OCR text.

Examples: Tablet, Film-coated tablet, Capsule, Syrup, Suspension,
Injection, Cream, Ointment, Gel, Drops, Inhaler.

Do not infer the dosage form only from the brand name.

============================================================
5. MANUFACTURER
============================================================

Identify the actual manufacturer when it is clearly stated.

Look for: Manufactured by, Mfg. by, Manufactured at, A product of.

Do not confuse with marketer, distributor, retailer, or pharmacy.
If only a marketer is visible, manufacturer may remain null.

============================================================
6. PRESCRIPTION STATUS
============================================================

Only identify prescription status when it is explicitly visible
or clearly stated in the OCR text.

Examples: Schedule H, Schedule H1, Rx, Prescription only medicine.

If prescription status is not visible, return null.

============================================================
7. OCR ERROR HANDLING
============================================================

You may correct obvious OCR errors when the surrounding packaging
evidence strongly supports the correction. However, do NOT invent a
medicine identity merely because the name resembles a known medicine.

If the OCR contains conflicting medicine names or ingredients,
reduce confidence.

============================================================
8. CONFIDENCE
============================================================

HIGH: brand name, active ingredients, and strengths are all clearly
supported and the OCR evidence is consistent.

MEDIUM: medicine is probably identifiable but some packaging
information is incomplete or OCR contains moderate ambiguity.

LOW: medicine identity is uncertain, OCR is badly corrupted,
ingredients conflict, or identification requires significant guessing.

============================================================
9. CRITICAL RULE: DO NOT INVENT
============================================================

Do NOT fill missing packaging information using your general knowledge.

Example: if the manufacturer cannot be read from the OCR text,
return manufacturer = null even if you know who makes this medicine.

The identification module describes what can be established from
the scanned package only.

============================================================
10. COMBINATION MEDICINES
============================================================

If the package contains multiple active ingredients, extract every
active ingredient and preserve each ingredient's strength separately.
Do not merge them into one ingredient.

============================================================
FINAL RULE
============================================================

Return only the MedicineIdentity structure.

Do not provide uses, dosage instructions, side effects, precautions,
interactions, or medical advice.
"""


# ---------------------------------------------------------------------------
# Cached LLM initialization
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Read GEMINI_API_KEY from st.secrets (Cloud) or environment (local)."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")


@st.cache_resource(show_spinner=False)
def _get_structured_llm():
    """Initialize and cache the structured Gemini LLM."""
    api_key = _get_api_key()
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
        google_api_key=api_key,
    )
    return llm.with_structured_output(MedicineIdentity)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def identify_medicine(ocr_text: str, max_retries: int = 2) -> Optional[MedicineIdentity]:
    """Identify medicine information from OCR-extracted text.

    Parameters
    ----------
    ocr_text:
        Raw text extracted from medicine packaging.
    max_retries:
        Number of retries for Gemini rate-limit errors.

    Returns
    -------
    MedicineIdentity | None
        None when identification fails or confidence is too low.
    """
    if not ocr_text or len(ocr_text.strip()) < 3:
        logger.info("Medicine identification skipped: OCR text is empty or too short.")
        return None

    user_prompt = f"""
Identify the medicine from the following OCR text.

================ OCR TEXT ================

{ocr_text.strip()}

==========================================

Follow all medicine-identification rules.

Important:

- Identify the brand from the packaging evidence.
- Extract ALL active ingredients.
- Extract the strength for each ingredient.
- Identify dosage form only when supported.
- Identify manufacturer only when supported.
- Identify prescription status only when explicitly stated.
- Do not invent missing information.
- If this is a combination medicine, include every active ingredient.
- Assign confidence according to the evidence quality.
"""

    structured_llm = _get_structured_llm()

    for attempt in range(max_retries + 1):
        try:
            medicine = structured_llm.invoke([
                SystemMessage(content=MEDICINE_IDENTITY_PROMPT),
                HumanMessage(content=user_prompt),
            ])

            if medicine is None:
                logger.warning("Gemini returned no medicine identity.")
                return None

            has_brand = bool(medicine.brand_name)
            has_ingredients = bool(medicine.active_ingredients)

            if not has_brand and not has_ingredients:
                logger.warning("Could not reliably identify the medicine.")
                return None

            return medicine

        except Exception as exc:
            error_text = str(exc)

            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                if attempt >= max_retries:
                    logger.error("Gemini rate limit reached. Maximum retries exceeded.")
                    return None
                wait = 65
                logger.warning("Gemini rate limit. Waiting %d s before retry…", wait)
                time.sleep(wait)
            else:
                logger.error("Medicine identification error: %s", exc)
                return None

    return None
