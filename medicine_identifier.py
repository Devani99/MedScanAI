import os
import time
from typing import Optional, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()


# ============================================================
# DATA MODELS
# ============================================================

class Ingredient(BaseModel):
    name: str
    strength: Optional[str] = None


class MedicineIdentity(BaseModel):
    brand_name: Optional[str] = None

    active_ingredients: list[Ingredient] = Field(
        default_factory=list
    )

    manufacturer: Optional[str] = None

    dosage_form: Optional[str] = None

    prescription_status: Optional[str] = None

    confidence: Literal[
        "high",
        "medium",
        "low"
    ] = "low"


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_MODEL = os.getenv(
    "MEDSCAN_GEMINI_MODEL",
    "gemini-3.1-flash-lite"
)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set in the environment."
    )


llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0,
    google_api_key=GEMINI_API_KEY
)


structured_llm = llm.with_structured_output(
    MedicineIdentity
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

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

Do not confuse:

- manufacturer
- marketer
- distributor
- pharmacy
- retailer
- company name
- slogan

with the medicine brand name.

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

Examples:

Dolo-650:
Paracetamol 650 mg

Brophyle-N:
Acebrophylline 100 mg
Acetylcysteine 600 mg

Amoxil-500:
Amoxicillin 500 mg

IMPORTANT:

Do not treat these as active ingredients:

- excipients
- preservatives
- colours
- flavours
- inactive ingredients
- manufacturer names
- company names

If multiple active ingredients are present, extract ALL of them.

============================================================
3. STRENGTH
============================================================

Extract the strength corresponding to each active ingredient.

Examples:

650 mg
500 mg
10 mg
250 mg/5 mL
100 mg + 600 mg

Do not randomly associate numbers with ingredients.

Only associate a strength with an ingredient when the packaging
evidence supports the relationship.

============================================================
4. DOSAGE FORM
============================================================

Identify the dosage form when supported by the OCR text.

Examples:

Tablet
Film-coated tablet
Capsule
Syrup
Suspension
Injection
Cream
Ointment
Gel
Drops
Inhaler

Do not infer the dosage form only from the brand name.

============================================================
5. MANUFACTURER
============================================================

Identify the actual manufacturer when it is clearly stated.

Look for:

- Manufactured by
- Mfg. by
- Manufactured at
- A product of

Do not confuse the manufacturer with:

- marketer
- distributor
- retailer
- pharmacy

If only a marketer is visible, manufacturer may remain null.

============================================================
6. PRESCRIPTION STATUS
============================================================

Only identify prescription status when it is explicitly visible
or clearly stated in the OCR text.

Examples:

Schedule H
Schedule H1
Schedule H Prescription Drug
Rx
Prescription only medicine

Do NOT determine prescription status from general medical knowledge.

If prescription status is not visible, return null.

============================================================
7. OCR ERROR HANDLING
============================================================

OCR may produce errors such as:

"Dolo 650" instead of "Dolo-650"
"Paracetamo1" instead of "Paracetamol"
"Acetylcystein" instead of "Acetylcysteine"

You may correct obvious OCR errors when the surrounding packaging
evidence strongly supports the correction.

However, do NOT invent a medicine identity merely because the name
resembles a known medicine.

If the OCR contains conflicting medicine names or ingredients,
reduce confidence.

============================================================
8. CONFIDENCE
============================================================

HIGH:

Use high only when:

- brand name is clearly supported
- active ingredient(s) are clearly supported
- strength(s) are clearly supported
- OCR evidence is consistent

MEDIUM:

Use medium when:

- the medicine is probably identifiable
- but some packaging information is incomplete
- or OCR contains moderate ambiguity

LOW:

Use low when:

- medicine identity is uncertain
- OCR is badly corrupted
- ingredients conflict
- brand and ingredients do not agree
- identification requires significant guessing

Never choose high merely because you recognize a medicine.

============================================================
9. CRITICAL RULE: DO NOT INVENT
============================================================

This is extremely important.

Do NOT fill missing packaging information using your general
knowledge.

Example:

OCR says:

Dolo-650
Paracetamol 650 mg

but the manufacturer cannot be read.

Correct:

manufacturer = null

Incorrect:

manufacturer = "MICRO LABS LIMITED"

even if you know that Dolo-650 is commonly manufactured by
Micro Labs.

The identification module describes what can be established from
the scanned package.

The research agent is responsible for retrieving additional
medicine information later.

============================================================
10. COMBINATION MEDICINES
============================================================

If the package contains multiple active ingredients:

- extract every active ingredient
- preserve each ingredient's strength
- do not merge them into one ingredient
- do not omit a second ingredient

For example:

Brophyle-N

must become:

active_ingredients = [
    {
        "name": "Acebrophylline",
        "strength": "100 mg"
    },
    {
        "name": "Acetylcysteine",
        "strength": "600 mg"
    }
]

============================================================
FINAL RULE
============================================================

Return only the MedicineIdentity structure.

Do not provide:

- uses
- dosage instructions
- side effects
- precautions
- interactions
- medical advice
- explanations

Those will be generated later by the medicine research agent.
"""


# ============================================================
# IDENTIFY MEDICINE
# ============================================================

def identify_medicine(
    ocr_text: str,
    max_retries: int = 2
) -> Optional[MedicineIdentity]:

    """
    Identify medicine information from PaddleOCR text.

    Parameters
    ----------
    ocr_text:
        Raw text extracted from medicine packaging.

    max_retries:
        Number of retries for Gemini rate-limit errors.

    Returns
    -------
    MedicineIdentity | None
    """

    if not ocr_text:
        print(
            "Medicine identification skipped: "
            "OCR text is empty."
        )
        return None

    ocr_text = ocr_text.strip()

    if len(ocr_text) < 3:
        print(
            "Medicine identification skipped: "
            "OCR text is too short."
        )
        return None

    user_prompt = f"""
Identify the medicine from the following OCR text.

================ OCR TEXT ================

{ocr_text}

===========================================

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

    for attempt in range(max_retries + 1):

        try:

            medicine = structured_llm.invoke(
                [
                    SystemMessage(
                        content=MEDICINE_IDENTITY_PROMPT
                    ),
                    HumanMessage(
                        content=user_prompt
                    )
                ]
            )

            if medicine is None:
                print(
                    "Gemini returned no medicine identity."
                )
                return None

            # ----------------------------------------
            # BASIC VALIDATION
            # ----------------------------------------

            has_brand = bool(
                medicine.brand_name
            )

            has_ingredients = bool(
                medicine.active_ingredients
            )

            if not has_brand and not has_ingredients:

                print(
                    "Could not reliably identify "
                    "the medicine."
                )

                return None

            return medicine

        except Exception as e:

            error_text = str(e)

            # ----------------------------------------
            # RATE LIMIT
            # ----------------------------------------

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            ):

                if attempt >= max_retries:

                    print(
                        "Gemini rate limit reached. "
                        "Maximum retries exceeded."
                    )

                    return None

                wait_time = 65

                print(
                    f"Gemini rate limit reached. "
                    f"Waiting {wait_time} seconds "
                    f"before retry..."
                )

                time.sleep(wait_time)

            # ----------------------------------------
            # OTHER ERROR
            # ----------------------------------------

            else:

                print(
                    "Medicine identification error:"
                )

                print(e)

                return None

    return None


# ============================================================
# HELPER FUNCTION
# ============================================================

def medicine_identity_to_dict(
    medicine: MedicineIdentity
) -> dict:

    """
    Convert MedicineIdentity to a normal dictionary.
    Useful for Streamlit/API responses.
    """

    return medicine.model_dump(
        exclude_none=True
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_ocr_text = """
    DOLO-650
    Paracetamol Tablets IP 650 mg
    Each tablet contains:
    Paracetamol IP 650 mg
    Manufactured by MICRO LABS LIMITED
    """

    print(
        f"Using Gemini model: {GEMINI_MODEL}"
    )

    medicine = identify_medicine(
        test_ocr_text
    )

    if medicine:

        print("\nMedicine identified:")
        print(
            medicine.model_dump_json(
                indent=2
            )
        )

    else:

        print(
            "\nMedicine identification failed."
        )