"""MedScanAI medicine research agent.

Stack:
- Gemini 3.1 Flash-Lite via LangChain
- Tavily web retrieval
- LangGraph orchestration
- LangSmith tracing (optional, configured through environment variables)

Workflow:
  build_query → search → evaluate_sources → synthesize →
  self_check → (corrective loop) → validate → extract_sources →
  clean_output → END
"""

from __future__ import annotations

import logging
import os
from typing import Optional
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph

from schemas import (
    MedicineAgentState,
    MedicineIdentity,
    MedicineInformation,
    SelfCheckResult,
    SourceEvaluation,
    SourceInfo,
    ValidationResult,
)

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3.1-flash-lite"
MAX_CORRECTIVE_RETRIES = int(os.getenv("MEDSCAN_MAX_RETRIES", "1"))


# ---------------------------------------------------------------------------
# Secrets helpers
# ---------------------------------------------------------------------------

def _get_gemini_key() -> str:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")


def _get_tavily_key() -> str:
    try:
        return st.secrets["TAVILY_API_KEY"]
    except Exception:
        return os.getenv("TAVILY_API_KEY", "")


def _configure_langsmith() -> None:
    """Push LangSmith env vars so the SDK picks them up."""
    for key in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"):
        value = None
        try:
            value = st.secrets[key]
        except Exception:
            value = os.getenv(key)
        if value:
            os.environ[key] = str(value)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SOURCE_SELECTION_PROMPT = """
You are MedScan AI's medical source-quality evaluator.

Do NOT answer the medicine question. Select only search results that are suitable
as evidence for patient-facing medicine information.

Prioritize:
1. official drug regulators and official drug labels
2. government health agencies and national health services
3. official prescribing/patient information
4. major recognized medical institutions
5. professionally reviewed medical references

Official manufacturer prescribing information can be used when appropriate.
Commercial pharmacies/general health sites are lower priority and should only be
selected when stronger evidence is unavailable and the content is clearly
professional. Reject blogs, forums, social media, SEO/marketing pages, and
user-generated content.

Evaluate the organization, title, URL, and supplied content. Do not reject a
credible source merely because it is not on a predefined domain whitelist.

The active ingredient + strength + dosage form are more important than the brand
name when matching medical information.

Return selected result indices only through the structured schema.
"""

MEDICINE_SYSTEM_PROMPT = """
You are MedScan AI's evidence-synthesis component.

Use ONLY the supplied retrieved evidence for medical claims. Do not fill gaps
from memory. If a fact is unsupported, leave the text field null or the list
empty. Never invent information merely to make the response complete.

MEDICINE IDENTITY
Prioritize active ingredient + strength + dosage form. OCR-derived brand,
manufacturer, pharmacy, or distributor text may be imperfect.

USES
Describe evidence-supported common uses. Do not diagnose the user.

HOW TO TAKE
Primarily answer whether the medicine is generally taken before food, after food,
with food, without food, or with/without food. Add general administration
instructions only when supported (for example swallow with water, swallow whole,
do not crush, dissolve first, shake well).

Never generate personalized dosing. Do not decide the user's number of tablets,
frequency, treatment duration, age-based dose, or weight-based dose. If reliable
food-timing evidence is unavailable, state that reliable food-timing information
was not established and advise following the medicine label or pharmacist/
prescriber instructions.

IMPORTANT PRECAUTIONS
Combine major evidence-supported cautions here, including allergies, liver/kidney
conditions, alcohol, duplicate active ingredient warnings, relevant conditions,
pregnancy, and breastfeeding. Pregnancy and breastfeeding must NOT be separate
sections. Preserve nuance; do not reduce pregnancy information to simply "safe"
or "unsafe".

POSSIBLE SIDE EFFECTS
Include only effects the evidence identifies as possible adverse effects at
recommended/normal use. Do not label rare effects as common. Do not mix overdose
consequences into this field.

IF YOU TAKE MORE THAN THE RECOMMENDED AMOUNT
The overdose_risks field represents the user-facing section titled exactly:
"What Can Happen If You Take More Than Recommended".
Only include effects or risks explicitly connected by the evidence to overdose,
excess dose, or taking too much. Never move ordinary serious adverse reactions
into this field just because they sound dangerous.

WHEN TO SEEK MEDICAL HELP
Include evidence-supported urgent warning situations, including serious adverse
reactions or overdose situations when appropriate. Do not invent emergency criteria.

DRUG INTERACTIONS / STORAGE
Include only evidence-supported information.

CONFIDENCE
high = important claims have strong authoritative support.
medium = credible but incomplete evidence.
low = identity/evidence is uncertain.
Writing quality is not evidence quality.

Use concise, patient-friendly English.
"""

SELF_CHECK_PROMPT = """
You are MedScan AI's self-check and corrective-retrieval planner.

Compare the generated MedicineInformation with the supplied evidence. Do not add
new medical facts from memory.

Identify:
- unsupported fields/claims
- important missing fields that could reasonably be retrieved
- contradictions or weak evidence
- incorrect classification between possible_side_effects, overdose_risks, and
  when_to_seek_medical_help
- unsupported food timing
- unsupported pregnancy/breastfeeding claims

Important fields to try to recover when missing include: what_is_it, uses,
how_it_works, how_to_take/food timing, important_precautions,
possible_side_effects, drug_interactions, overdose_risks,
when_to_seek_medical_help, and storage.

Do not require every field to be populated if reliable information genuinely
cannot be established. If missing/unsupported information could benefit from
another search, create short targeted corrective_search_terms.

Set is_sufficient=false when important claims are unsupported or important
retrievable information is missing. Confidence reflects evidence quality.
"""

VALIDATOR_PROMPT = """
You are MedScan AI's final independent evidence validator.

Do NOT generate new medicine information. Check every important user-facing
claim against the supplied evidence.

Pay special attention to medicine identity, uses, food timing/how_to_take,
precautions, pregnancy/breastfeeding, possible side effects, interactions,
overdose risks, storage, and when to seek medical help.

Flag invented dose/frequency/duration, unsupported meal timing, unsupported
pregnancy claims, unsupported interactions, and any normal-use adverse effect
incorrectly presented as an overdose effect.

A medically plausible statement is not enough; it must be supported by the
retrieved evidence. Set is_sufficient=true only when displayed claims are
adequately grounded. Missing optional information alone does not require failure
if it was not reliably established and is omitted from the user-facing output.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _medicine_identity_text(medicine: MedicineIdentity) -> str:
    parts: list[str] = []
    for ingredient in medicine.active_ingredients:
        value = ingredient.name.strip()
        if ingredient.strength:
            value += f" {ingredient.strength.strip()}"
        parts.append(value)
    if medicine.dosage_form:
        parts.append(medicine.dosage_form)
    return " ".join(parts).strip()


def _format_evidence(results: list[dict]) -> str:
    blocks: list[str] = []
    for i, result in enumerate(results, 1):
        blocks.append(
            f"\n=== SOURCE {i} ===\n"
            f"TITLE: {result.get('title', '')}\n"
            f"URL: {result.get('url', '')}\n"
            f"CONTENT:\n{result.get('content', '')}\n"
        )
    return "\n".join(blocks)


def _merge_results(existing: list[dict], new: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for item in [*existing, *new]:
        url = (item.get("url") or "").strip()
        key = url or f"{item.get('title', '')}|{item.get('content', '')[:100]}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _public_output(info: Optional[MedicineInformation], sources: list[SourceInfo]) -> dict:
    """Strip null/empty values for user-facing display."""
    if info is None:
        return {"sources": [s.model_dump(exclude_none=True) for s in sources]}

    raw = info.model_dump()
    cleaned: dict = {}
    for key, value in raw.items():
        if value is None:
            continue
        if isinstance(value, (list, dict, str)) and len(value) == 0:
            continue
        cleaned[key] = value

    if sources:
        cleaned["sources"] = [s.model_dump(exclude_none=True) for s in sources]
    return cleaned


# ---------------------------------------------------------------------------
# Cached agent factory
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _build_agent():
    """Build and cache the LangGraph medicine research agent."""
    _configure_langsmith()

    gemini_key = _get_gemini_key()
    tavily_key = _get_tavily_key()

    # Set Tavily key in environment so the tool picks it up
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
        google_api_key=gemini_key,
    )

    source_evaluator_llm = llm.with_structured_output(SourceEvaluation)
    medicine_llm = llm.with_structured_output(MedicineInformation)
    self_check_llm = llm.with_structured_output(SelfCheckResult)
    validator_llm = llm.with_structured_output(ValidationResult)
    search_tool = TavilySearch(max_results=10, topic="general")

    # ----------------------------------------------------------------
    # Graph nodes (closures over cached LLM/tool objects)
    # ----------------------------------------------------------------

    def build_search_query(state: MedicineAgentState):
        medicine = state["medicine"]
        identity = _medicine_identity_text(medicine)
        query = (
            f"{identity} official patient medicine information uses mechanism "
            "with food without food before meal after meal precautions warnings "
            "pregnancy breastfeeding side effects interactions overdose taking too much storage"
        )
        return {"search_query": query}

    def search_web(state: MedicineAgentState):
        response = search_tool.invoke({"query": state["search_query"]})
        new_results = response.get("results", []) if isinstance(response, dict) else []
        all_results = _merge_results(state.get("search_results", []), new_results)
        logger.info("Search: %d new results, %d total.", len(new_results), len(all_results))
        return {"search_results": all_results}

    def evaluate_sources(state: MedicineAgentState):
        results = state["search_results"]
        if not results:
            return {"selected_results": []}

        candidates = []
        for i, result in enumerate(results):
            candidates.append(
                f"RESULT INDEX: {i}\n"
                f"TITLE: {result.get('title', '')}\n"
                f"URL: {result.get('url', '')}\n"
                f"CONTENT: {result.get('content', '')[:2500]}\n"
            )

        evaluation = source_evaluator_llm.invoke([
            SystemMessage(content=SOURCE_SELECTION_PROMPT),
            HumanMessage(content="\n\n".join(candidates)),
        ])

        selected = [
            results[i]
            for i in evaluation.selected_indices
            if isinstance(i, int) and 0 <= i < len(results)
        ]
        logger.info("Source evaluator selected %d results.", len(selected))
        return {"selected_results": selected}

    def synthesize_information(state: MedicineAgentState):
        medicine = state["medicine"]
        results = state["selected_results"]
        if not results:
            return {"medicine_information": None}

        ingredients = ", ".join(
            f"{x.name} {x.strength or ''}".strip()
            for x in medicine.active_ingredients
        )
        prompt = (
            f"MEDICINE FROM PACKAGE\n"
            f"Brand: {medicine.brand_name or 'Unknown'}\n"
            f"Active ingredient(s): {ingredients or 'Unknown'}\n"
            f"Dosage form: {medicine.dosage_form or 'Unknown'}\n"
            f"Manufacturer: {medicine.manufacturer or 'Unknown'}\n"
            f"Prescription information from package: {medicine.prescription_status or 'Unknown'}\n"
            f"OCR/identity confidence: {medicine.confidence or 'Unknown'}\n\n"
            f"RETRIEVED EVIDENCE\n{_format_evidence(results)}\n\n"
            "Create MedicineInformation using only this evidence."
        )
        info = medicine_llm.invoke([
            SystemMessage(content=MEDICINE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        return {"medicine_information": info}

    def self_check_information(state: MedicineAgentState):
        info = state["medicine_information"]
        if info is None:
            check = SelfCheckResult(
                is_sufficient=False,
                missing_important_fields=["medicine information"],
                issues=["No suitable evidence was available for synthesis."],
                corrective_search_terms=[
                    f"{_medicine_identity_text(state['medicine'])} official drug label patient information"
                ],
                confidence="low",
            )
            return {"self_check": check}

        prompt = (
            f"GENERATED INFORMATION\n{info.model_dump_json(indent=2)}\n\n"
            f"EVIDENCE\n{_format_evidence(state['selected_results'])}\n\n"
            "Identify unsupported or important missing information and propose targeted "
            "corrective searches only when useful."
        )
        check = self_check_llm.invoke([
            SystemMessage(content=SELF_CHECK_PROMPT),
            HumanMessage(content=prompt),
        ])
        logger.info(
            "Self-check sufficient=%s, confidence=%s, missing=%s",
            check.is_sufficient,
            check.confidence,
            check.missing_important_fields,
        )
        return {"self_check": check}

    def self_check_router(state: MedicineAgentState):
        check = state["self_check"]
        if check and check.is_sufficient:
            return "validate"
        if state["retry_count"] < MAX_CORRECTIVE_RETRIES:
            return "correct"
        return "validate"

    def build_corrective_query(state: MedicineAgentState):
        medicine = state["medicine"]
        check = state["self_check"]
        identity = _medicine_identity_text(medicine)

        terms: list[str] = []
        if check:
            terms.extend(check.corrective_search_terms)
            terms.extend(check.missing_important_fields)
            terms.extend(check.unsupported_fields)

        target = " ".join(dict.fromkeys(x.strip() for x in terms if x and x.strip()))
        if not target:
            target = "official drug label food administration precautions overdose storage"

        query = f"{identity} {target}"
        logger.info("Corrective search #%d: %s", state["retry_count"] + 1, query)
        return {
            "search_query": query,
            "retry_count": state["retry_count"] + 1,
        }

    def validate_information(state: MedicineAgentState):
        info = state["medicine_information"]
        if info is None:
            result = ValidationResult(
                is_sufficient=False,
                missing_fields=["medicine information"],
                issues=["No medicine information could be synthesized."],
                confidence="low",
            )
            return {"validation": result}

        prompt = (
            f"GENERATED INFORMATION\n{info.model_dump_json(indent=2)}\n\n"
            f"EVIDENCE\n{_format_evidence(state['selected_results'])}\n\n"
            "Perform the final evidence validation. Missing optional fields may remain omitted, "
            "but every displayed medical claim must be supported."
        )
        validation = validator_llm.invoke([
            SystemMessage(content=VALIDATOR_PROMPT),
            HumanMessage(content=prompt),
        ])
        info.confidence = validation.confidence
        logger.info(
            "Final validation sufficient=%s, confidence=%s",
            validation.is_sufficient,
            validation.confidence,
        )
        return {"validation": validation, "medicine_information": info}

    def extract_sources(state: MedicineAgentState):
        sources: list[SourceInfo] = []
        seen: set[str] = set()
        for result in state["selected_results"]:
            url = (result.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            domain = urlparse(url).hostname
            sources.append(SourceInfo(title=result.get("title"), url=url, domain=domain))
        return {"final_sources": sources}

    def clean_output(state: MedicineAgentState):
        return {
            "final_output": _public_output(
                state.get("medicine_information"),
                state.get("final_sources", []),
            )
        }

    # ----------------------------------------------------------------
    # Build graph
    # ----------------------------------------------------------------

    builder = StateGraph(MedicineAgentState)

    builder.add_node("build_query", build_search_query)
    builder.add_node("search", search_web)
    builder.add_node("evaluate_sources", evaluate_sources)
    builder.add_node("synthesize", synthesize_information)
    builder.add_node("self_check", self_check_information)
    builder.add_node("corrective_query", build_corrective_query)
    builder.add_node("validate", validate_information)
    builder.add_node("extract_sources", extract_sources)
    builder.add_node("clean_output", clean_output)

    builder.add_edge(START, "build_query")
    builder.add_edge("build_query", "search")
    builder.add_edge("search", "evaluate_sources")
    builder.add_edge("evaluate_sources", "synthesize")
    builder.add_edge("synthesize", "self_check")

    builder.add_conditional_edges(
        "self_check",
        self_check_router,
        {"correct": "corrective_query", "validate": "validate"},
    )

    builder.add_edge("corrective_query", "search")
    builder.add_edge("validate", "extract_sources")
    builder.add_edge("extract_sources", "clean_output")
    builder.add_edge("clean_output", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def research_medicine(medicine: MedicineIdentity) -> dict:
    """Run the complete medicine research agent and return the graph state."""
    agent = _build_agent()
    return agent.invoke({
        "medicine": medicine,
        "search_query": "",
        "search_results": [],
        "selected_results": [],
        "medicine_information": None,
        "self_check": None,
        "validation": None,
        "retry_count": 0,
        "final_sources": [],
        "final_output": {},
    })
