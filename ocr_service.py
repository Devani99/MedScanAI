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

    result = ocr.ocr(
        image_path,
        cls=False
    )

    extracted_text = []

    if not result:
        return ""

    for page in result:

        if not page:
            continue

        for line in page:

            if len(line) < 2:
                continue

            text_info = line[1]

            if not text_info:
                continue

            text = text_info[0]

            score = float(
                text_info[1]
            )

            text = str(
                text
            ).strip()

            if not text:
                continue

            if score >= 0.50:

                extracted_text.append(
                    text
                )

    return "\n".join(
        extracted_text
    )
