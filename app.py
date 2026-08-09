"""MedScanAI – Streamlit application entry point.

Upload a medicine package photo, extract text with EasyOCR,
identify the medicine with Gemini, research it with a LangGraph
agent, and display the results in a clean, simple layout.
"""

from __future__ import annotations

import logging
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page config – must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MedScanAI",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Minimal CSS – clean white/light layout, remove Streamlit padding
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 760px;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1.4rem;
        margin-bottom: 0.4rem;
        color: #1a1a1a;
    }
    .disclaimer-box {
        margin-top: 2rem;
        padding: 0.8rem 1rem;
        border-left: 3px solid #ccc;
        background: #f8f8f8;
        font-size: 0.82rem;
        color: #555;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Imports (after page config so Streamlit is initialized)
# ---------------------------------------------------------------------------

from medicine_identifier import identify_medicine
from medicine_agent import research_medicine
from ocr_service import extract_medicine_text

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("MedScanAI")
st.caption("Upload a medicine package photo to get reliable information about it.")

st.divider()

# ---------------------------------------------------------------------------
# Image input
# ---------------------------------------------------------------------------

input_mode = st.radio(
    "Input method",
    ["Upload photo", "Use camera"],
    index=0,
    horizontal=True,
    label_visibility="collapsed",
)

image_file = None

if input_mode == "Upload photo":
    image_file = st.file_uploader(
        "Upload medicine package photo",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="visible",
    )
else:
    col_cam, _ = st.columns([1, 1])
    with col_cam:
        image_file = st.camera_input("Take a photo of the medicine package")

# ---------------------------------------------------------------------------
# Preview + process button
# ---------------------------------------------------------------------------

if image_file:
    st.image(image_file, width=320, caption="Uploaded image")

    process_btn = st.button("Process Medicine", type="primary")

    if process_btn:
        image_path: str | None = None

        try:
            # ------------------------------------------------------------------
            # Save uploaded file to a temp location
            # ------------------------------------------------------------------
            suffix = os.path.splitext(image_file.name)[1] if hasattr(image_file, "name") else ".png"
            if not suffix:
                suffix = ".png"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(image_file.getbuffer())
                image_path = tmp.name

            # ------------------------------------------------------------------
            # Processing pipeline with live status
            # ------------------------------------------------------------------
            with st.status("Analyzing medicine…", expanded=True) as status:

                # Step 1 – OCR
                st.write("Reading medicine package…")
                ocr_text = extract_medicine_text(image_path)

                if not ocr_text.strip():
                    status.update(label="Could not read the image", state="error")
                    st.error(
                        "No readable text was detected in the image. "
                        "Please upload a clearer photo with visible packaging text."
                    )
                    st.stop()

                st.write("Text extracted from package.")

                # Step 2 – Identification
                st.write("Identifying medicine…")
                medicine = identify_medicine(ocr_text)

                if medicine is None:
                    status.update(label="Medicine could not be identified", state="error")
                    st.error(
                        "The medicine could not be identified reliably from the image. "
                        "Please try a clearer photo showing the medicine name and ingredients."
                    )
                    st.stop()

                st.write("Medicine identified.")

                # Step 3 – Research
                st.write("Researching medical sources…")
                result = research_medicine(medicine)
                final_output = result.get("final_output")

                if not final_output:
                    status.update(label="Could not retrieve reliable information", state="error")
                    st.error(
                        "Unable to retrieve reliable medical information for this medicine. "
                        "Please try again later."
                    )
                    st.stop()

                st.write("Medical information ready.")
                status.update(label="Analysis complete", state="complete")

            # ------------------------------------------------------------------
            # Display results
            # ------------------------------------------------------------------

            st.divider()

            # Medicine identity summary
            brand = medicine.brand_name or ""
            ingredients = []
            for ing in medicine.active_ingredients:
                label = ing.name
                if ing.strength:
                    label += f" {ing.strength}"
                ingredients.append(label)
            ingredients_str = ", ".join(ingredients)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Medicine", final_output.get("medicine_name") or brand or "—")
            with col_b:
                st.metric("Generic name", final_output.get("generic_name") or ingredients_str or "—")
            with col_c:
                st.metric("Dosage form", final_output.get("dosage_form") or medicine.dosage_form or "—")

            strength_val = final_output.get("strength")
            if not strength_val and ingredients:
                # Extract strength from first identified ingredient as fallback
                first = medicine.active_ingredients[0] if medicine.active_ingredients else None
                if first and first.strength:
                    strength_val = first.strength

            if strength_val:
                st.write(f"**Strength:** {strength_val}")

            if medicine.manufacturer:
                st.write(f"**Manufacturer:** {medicine.manufacturer}")

            # ------------------------------------------------------------------
            # Sections helper
            # ------------------------------------------------------------------

            def section(title: str, key: str, render: str = "text") -> None:
                """Render a result section if the field is non-empty."""
                value = final_output.get(key)
                if not value:
                    return
                st.subheader(title)
                if render == "text":
                    st.write(value)
                elif render == "info":
                    st.info(value)
                elif render == "list":
                    for item in value:
                        st.markdown(f"- {item}")
                elif render == "warn_list":
                    for item in value:
                        st.warning(item)
                elif render == "error_list":
                    for item in value:
                        st.error(item)

            section("What is this medicine?", "what_is_it", "text")
            section("Uses", "uses", "list")
            section("How it works", "how_it_works", "text")
            section("How to take", "how_to_take", "info")
            section("Important precautions", "important_precautions", "warn_list")
            section("Possible side effects", "possible_side_effects", "list")
            section("Drug interactions", "drug_interactions", "list")

            # Overdose risks – special heading per spec
            overdose = final_output.get("overdose_risks")
            if overdose:
                st.subheader("What Can Happen If You Take More Than Recommended")
                for item in overdose:
                    st.error(item)

            section("When to seek medical help", "when_to_seek_medical_help", "warn_list")
            section("Storage", "storage", "text")

            # Sources
            sources = final_output.get("sources", [])
            if sources:
                st.subheader("Sources")
                for src in sources:
                    title = src.get("title") or src.get("domain") or "Medical source"
                    url = src.get("url", "")
                    if url:
                        st.markdown(f"- [{title}]({url})")
                    else:
                        st.markdown(f"- {title}")

            # Disclaimer
            st.markdown(
                """
                <div class="disclaimer-box">
                <strong>Medical Disclaimer:</strong> MedScanAI provides general educational
                information gathered from publicly available medical sources. It does not
                provide medical diagnosis, personalized treatment, or a substitute for advice
                from a qualified healthcare professional. Always check the medicine packaging
                or official patient information leaflet and consult a doctor or pharmacist
                when necessary.
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as exc:
            logging.exception("Unexpected error during processing.")
            st.error(
                "An unexpected error occurred while processing the medicine. "
                "Please try again with a clearer image."
            )
            with st.expander("Technical details (for developers)"):
                st.exception(exc)

        finally:
            if image_path:
                try:
                    os.remove(image_path)
                except OSError:
                    pass
