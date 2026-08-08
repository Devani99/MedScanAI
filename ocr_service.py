import streamlit as st
from paddleocr import PaddleOCR
import os


@st.cache_resource
def get_ocr():

    use_gpu = os.getenv(
        "MEDSCAN_USE_GPU",
        "false"
    ).lower() == "true"

    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en",
        use_gpu=use_gpu
    )



def extract_medicine_text(image_path: str) -> str:

    ocr = get_ocr()

    result = ocr.predict(
        image_path
    )

    extracted_text = []

    for item in result:

        data = item

        if hasattr(item, "json"):
            data = item.json

        if isinstance(data, str):

            import json

            data = json.loads(data)

        if not isinstance(data, dict):
            continue

        data = data.get(
            "res",
            data
        )

        texts = data.get(
            "rec_texts",
            []
        )

        scores = data.get(
            "rec_scores",
            []
        )

        for i, text in enumerate(texts):

            text = str(
                text
            ).strip()

            if not text:
                continue

            score = (
                float(scores[i])
                if i < len(scores)
                else None
            )

            if (
                score is None
                or score >= 0.50
            ):

                extracted_text.append(
                    text
                )

    return "\n".join(
        extracted_text
    )   