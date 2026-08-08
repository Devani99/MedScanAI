import json
import streamlit as st
from paddleocr import PaddleOCR


@st.cache_resource
def get_ocr():
    return PaddleOCR(
        lang="en"
    )


def extract_medicine_text(image_path: str) -> str:

    ocr = get_ocr()

    try:
        results = ocr.predict(image_path)

        extracted_text = []

        for result in results:

            data = result.json

            if callable(data):
                data = data()

            if isinstance(data, str):
                data = json.loads(data)

            texts = data.get("rec_texts", [])

            scores = data.get(
                "rec_scores",
                []
            )

            for i, text in enumerate(texts):

                text = str(text).strip()

                if not text:
                    continue

                score = 1.0

                if i < len(scores):
                    score = float(scores[i])

                if score >= 0.50:
                    extracted_text.append(text)

        return "\n".join(extracted_text)

    except Exception as e:

        st.error(
            f"OCR processing failed: {str(e)}"
        )

        raise
