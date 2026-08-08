import os
import tempfile

import streamlit as st

from ocr_service import extract_medicine_text
from medicine_identifier import identify_medicine
from medicine_agent import research_medicine


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MedScan AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Remove unnecessary top spacing */
    .block-container {
        padding-top: 5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Fixed header */
    .medscan-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 64px;
        z-index: 999999;
        background: #0e1117;
        border-bottom: 1px solid rgba(255,255,255,0.10);
        display: flex;
        align-items: center;
        padding: 0 28px;
    }

    .medscan-header-inner {
        width: 1200px;
        max-width: calc(100% - 40px);
        margin: 0 auto;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
    }

    .brand-icon {
        font-size: 1.45rem;
    }

    .header-tag {
        font-size: 0.82rem;
        color: #aeb4c0;
    }

    /* Hero */
    .hero {
        text-align: center;
        padding: 1.2rem 0 2rem 0;
    }

    .hero h1 {
        font-size: 2.7rem;
        margin: 0;
        font-weight: 750;
        color: #ffffff;
    }

    .hero p {
        margin-top: 0.6rem;
        font-size: 1.05rem;
        color: #aeb4c0;
    }

    /* Input section */
    .input-card {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 1.4rem;
        background: rgba(255,255,255,0.025);
        margin-bottom: 1.5rem;
    }

    /* Section headings */
    .section-heading {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.7rem;
    }

    /* Medicine identity */
    .medicine-card {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 1.3rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        background: rgba(255,255,255,0.025);
    }

    .medicine-name {
        font-size: 1.8rem;
        font-weight: 750;
        margin-bottom: 0.3rem;
    }

    .medicine-subtitle {
        color: #aeb4c0;
        font-size: 1rem;
    }

    /* Disclaimer */
    .disclaimer {
        margin-top: 2.5rem;
        padding: 1rem 1.2rem;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 12px;
        color: #aeb4c0;
        font-size: 0.82rem;
        line-height: 1.6;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #707783;
        font-size: 0.78rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FIXED HEADER
# ============================================================

st.title("💊 MedScan AI")

# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Scan. Identify. Understand.</h1>
        <p>
            Get reliable, evidence-based information about
            your medicine from a simple package photo.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="input-card">',
    unsafe_allow_html=True
)

st.subheader("Provide Medicine Image")

# Upload is DEFAULT
input_method = st.radio(
    "Choose how you want to provide the medicine image:",
    [
        "📁 Upload Photo",
        "📷 Scan Medicine"
    ],
    index=0,
    horizontal=True,
    label_visibility="visible"
)

image_file = None


# ============================================================
# UPLOAD PHOTO
# ============================================================

if input_method == "📁 Upload Photo":

    image_file = st.file_uploader(
        "Upload a clear medicine package photo",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp"
        ],
        label_visibility="visible"
    )


# ============================================================
# CAMERA
# ============================================================

else:

    camera_col, empty_col = st.columns(
        [1, 2]
    )

    with camera_col:

        image_file = st.camera_input(
            "Take a clear photo of the medicine package",
            resolution="480p",
            width=480
        )


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if image_file:

    st.markdown("---")

    preview_col, details_col = st.columns(
        [1, 1],
        gap="large"
    )


    # ========================================================
    # IMAGE PREVIEW
    # ========================================================

    with preview_col:

        st.subheader("Image Preview")

        st.image(
            image_file,
            width=450
        )


    # ========================================================
    # SCAN BUTTON
    # ========================================================

    with details_col:

        st.subheader("Ready to Scan")

        st.write(
            "Make sure the medicine name, ingredients, "
            "and packaging text are clearly visible."
        )

        st.write("")

        scan_button = st.button(
            "🔍 Scan Medicine",
            type="primary",
            width="stretch"
        )


    # ========================================================
    # START PROCESSING
    # ========================================================

    if scan_button:

        image_path = None

        try:

            # ==================================================
            # SAVE IMAGE
            # ==================================================

            file_extension = os.path.splitext(
                image_file.name
            )[1]

            if not file_extension:
                file_extension = ".png"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_extension
            ) as temp_file:

                temp_file.write(
                    image_file.getbuffer()
                )

                image_path = temp_file.name


            # ==================================================
            # PROCESSING STATUS
            # ==================================================

            with st.status(
                "Analyzing medicine...",
                expanded=True
            ) as status:

                # ----------------------------------------------
                # OCR
                # ----------------------------------------------

                st.write(
                    "🔎 Reading medicine package..."
                )

                ocr_text = extract_medicine_text(
                    image_path
                )

                if not ocr_text.strip():

                    status.update(
                        label="Could not read medicine package",
                        state="error"
                    )

                    st.error(
                        "No readable text was detected "
                        "in the image. Please upload a clearer "
                        "photo."
                    )

                    st.stop()


                st.write(
                    "✓ Package text extracted"
                )


                # ----------------------------------------------
                # IDENTIFICATION
                # ----------------------------------------------

                st.write(
                    "🧠 Identifying medicine..."
                )

                medicine = identify_medicine(
                    ocr_text
                )

                if medicine is None:

                    status.update(
                        label="Medicine identification failed",
                        state="error"
                    )

                    st.error(
                        "The medicine could not be "
                        "identified reliably."
                    )

                    st.stop()


                st.write(
                    "✓ Medicine identified"
                )


                # ----------------------------------------------
                # RESEARCH AGENT
                # ----------------------------------------------

                st.write(
                    "🌐 Researching medical sources..."
                )

                result = research_medicine(
                    medicine
                )

                final_output = result.get(
                    "final_output"
                )

                if not final_output:

                    status.update(
                        label="Could not generate reliable information",
                        state="error"
                    )

                    st.error(
                        "Reliable medicine information "
                        "could not be generated."
                    )

                    st.stop()


                st.write(
                    "✓ Medical information generated"
                )


                # ----------------------------------------------
                # COMPLETE
                # ----------------------------------------------

                status.update(
                    label="Analysis completed",
                    state="complete"
                )


            # ==================================================
            # MEDICINE IDENTITY
            # ==================================================

            st.markdown("---")

            st.header("Medicine Identified")


            brand_name = (
                medicine.brand_name
                or "Unknown"
            )


            ingredients = []

            for ingredient in (
                medicine.active_ingredients
            ):

                text = ingredient.name

                if ingredient.strength:

                    text += (
                        f" ({ingredient.strength})"
                    )

                ingredients.append(
                    text
                )


            ingredients_text = (
                ", ".join(ingredients)
                if ingredients
                else "Unknown"
            )


            # ==================================================
            # MEDICINE CARD
            # ==================================================

            st.markdown(
                f"""
                <div class="medicine-card">

                    <div class="medicine-name">
                        {brand_name}
                    </div>

                    <div class="medicine-subtitle">
                        {ingredients_text}
                        &nbsp; • &nbsp;
                        {medicine.dosage_form or "Dosage form unavailable"}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # ==================================================
            # BASIC DETAILS
            # ==================================================

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Medicine",
                    brand_name
                )

            with col2:

                st.metric(
                    "Active Ingredient(s)",
                    ingredients_text
                )

            with col3:

                st.metric(
                    "Dosage Form",
                    medicine.dosage_form
                    or "Unknown"
                )


            if medicine.manufacturer:

                st.write(
                    f"**Manufacturer:** "
                    f"{medicine.manufacturer}"
                )


            # ==================================================
            # WHAT IS IT?
            # ==================================================

            if final_output.get(
                "what_is_it"
            ):

                st.markdown(
                    '<div class="section-heading">'
                    'What is this medicine?'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.write(
                    final_output[
                        "what_is_it"
                    ]
                )


            # ==================================================
            # USES
            # ==================================================

            if final_output.get(
                "uses"
            ):

                st.markdown(
                    '<div class="section-heading">'
                    'Uses'
                    '</div>',
                    unsafe_allow_html=True
                )

                for item in final_output[
                    "uses"
                ]:

                    st.markdown(
                        f"- {item}"
                    )


            # ==================================================
            # HOW IT WORKS
            # ==================================================

            if final_output.get(
                "how_it_works"
            ):

                st.markdown(
                    '<div class="section-heading">'
                    'How It Works'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.write(
                    final_output[
                        "how_it_works"
                    ]
                )


            # ==================================================
            # HOW TO TAKE
            # ==================================================

            if final_output.get(
                "how_to_take"
            ):

                st.markdown(
                    '<div class="section-heading">'
                    'How to Take'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.info(
                    final_output[
                        "how_to_take"
                    ]
                )


            # ==================================================
            # PRECAUTIONS
            # ==================================================

            if final_output.get(
                "important_precautions"
            ):

                st.markdown(
                    '<div class="section-heading">'
                    'Important Precautions'
                    '</div>',
                    unsafe_allow_html=True
                )

                for item in final_output[
                    "important_precautions"
                ]:

                    st.warning(
                        item
                    )


            # ==================================================
            # SIDE EFFECTS
            # ==================================================

            if final_output.get(
                "possible_side_effects"
            ):

                st.markdown(
                    '<div class="section-heading">'
                    'Possible Side Effects'
                    '</div>',
                    unsafe_allow_html=True
                )

                for item in final_output[
                    "possible_side_effects"
                ]:

                    st.markdown(
                        f"- {item}"
                    )


            # ==================================================
            # DRUG INTERACTIONS
            # ==================================================

            if final_output.get(
                "drug_interactions"
            ):

                st.markdown(
                    '<div class="section-heading">'
                    'Drug Interactions'
                    '</div>',
                    unsafe_allow_html=True
                )

                for item in final_output[
                    "drug_interactions"
                ]:

                    st.markdown(
                        f"- {item}"
                    )


            # ==================================================
            # OVERDOSE
            # ==================================================

            if final_output.get(
                "overdose_risks"
            ):

                st.markdown(
                    '<div class="section-heading">'
                    'If You Take More Than the '
                    'Recommended Amount'
                    '</div>',
                    unsafe_allow_html=True
                )

                for item in final_output[
                    "overdose_risks"
                ]:

                    st.error(
                        item
                    )


            # ==================================================
            # WHEN TO SEEK MEDICAL HELP
            # ==================================================

            if final_output.get(
                "when_to_seek_medical_help"
            ):

                st.markdown(
                    '<div class="section-heading">'
                    'When to Seek Medical Help'
                    '</div>',
                    unsafe_allow_html=True
                )

                for item in final_output[
                    "when_to_seek_medical_help"
                ]:

                    st.warning(
                        item
                    )


            # ==================================================
            # STORAGE
            # ==================================================

            if final_output.get(
                "storage"
            ):

                st.markdown(
                    '<div class="section-heading">'
                    'Storage'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.write(
                    final_output[
                        "storage"
                    ]
                )


            # ==================================================
            # SOURCES
            # ==================================================

            sources = final_output.get(
                "sources",
                []
            )


            if sources:

                st.markdown(
                    '<div class="section-heading">'
                    'Sources'
                    '</div>',
                    unsafe_allow_html=True
                )

                for source in sources:

                    title = source.get(
                        "title",
                        "Medical Source"
                    )

                    url = source.get(
                        "url",
                        ""
                    )

                    if url:

                        st.markdown(
                            f"- [{title}]({url})"
                        )

                    else:

                        st.markdown(
                            f"- {title}"
                        )


            # ==================================================
            # DISCLAIMER
            # ==================================================

            st.markdown(
                """
                <div class="disclaimer">

                <strong>Medical Disclaimer:</strong><br>

                MedScan AI provides general educational
                information gathered from external medical
                sources. It does not provide medical diagnosis,
                personalized treatment, or a substitute for
                advice from a qualified healthcare professional.

                Always check the medicine packaging or official
                patient information leaflet and consult a doctor
                or pharmacist when necessary.

                </div>
                """,
                unsafe_allow_html=True
            )


            # ==================================================
            # FOOTER
            # ==================================================

            st.markdown(
                """
                <div class="footer">
                    MedScan AI • AI-powered medicine information
                </div>
                """,
                unsafe_allow_html=True
            )


        except Exception as e:

            st.error(
                "An unexpected error occurred "
                "while processing the medicine."
            )

            with st.expander(
                "Technical details"
            ):

                st.exception(e)


        finally:

            if image_path:

                try:

                    os.remove(
                        image_path
                    )

                except OSError:

                    pass