"""Shared Pydantic schemas for MedScanAI."""

from __future__ import annotations

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field


Confidence = Literal["high", "medium", "low"]


# ---------------------------------------------------------------------------
# Medicine identity (produced by OCR + identifier)
# ---------------------------------------------------------------------------

class Ingredient(BaseModel):
    name: str
    strength: Optional[str] = None


class MedicineIdentity(BaseModel):
    brand_name: Optional[str] = None
    active_ingredients: list[Ingredient] = Field(default_factory=list)
    manufacturer: Optional[str] = None
    dosage_form: Optional[str] = None
    prescription_status: Optional[str] = None
    confidence: Optional[Confidence] = "low"


# ---------------------------------------------------------------------------
# Source tracking
# ---------------------------------------------------------------------------

class SourceInfo(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    domain: Optional[str] = None


# ---------------------------------------------------------------------------
# Agent internal schemas
# ---------------------------------------------------------------------------

class SourceEvaluation(BaseModel):
    selected_indices: list[int] = Field(default_factory=list)
    reason: Optional[str] = None


class MedicineInformation(BaseModel):
    medicine_name: Optional[str] = None
    generic_name: Optional[str] = None
    strength: Optional[str] = None
    dosage_form: Optional[str] = None
    what_is_it: Optional[str] = None
    uses: list[str] = Field(default_factory=list)
    how_it_works: Optional[str] = None
    how_to_take: Optional[str] = None
    important_precautions: list[str] = Field(default_factory=list)
    possible_side_effects: list[str] = Field(default_factory=list)
    drug_interactions: list[str] = Field(default_factory=list)
    overdose_risks: list[str] = Field(default_factory=list)
    when_to_seek_medical_help: list[str] = Field(default_factory=list)
    storage: Optional[str] = None
    confidence: Confidence = "low"


class SelfCheckResult(BaseModel):
    is_sufficient: bool
    unsupported_fields: list[str] = Field(default_factory=list)
    missing_important_fields: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    corrective_search_terms: list[str] = Field(default_factory=list)
    confidence: Confidence = "low"


class ValidationResult(BaseModel):
    is_sufficient: bool
    unsupported_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    confidence: Confidence = "low"


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class MedicineAgentState(TypedDict):
    medicine: MedicineIdentity
    search_query: str
    search_results: list[dict]
    selected_results: list[dict]
    medicine_information: Optional[MedicineInformation]
    self_check: Optional[SelfCheckResult]
    validation: Optional[ValidationResult]
    retry_count: int
    final_sources: list[SourceInfo]
    final_output: dict
