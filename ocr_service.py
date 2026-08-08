import streamlit as st
from paddleocr import PaddleOCR


@st.cache_resource
def get_ocr():

    return PaddleOCR(
        lang="en",
        use_angle_cls=True
    )


def extract_medicine_text(image_path: str) -> str:

    ocr = get_ocr()

    try:

        result = ocr.ocr(
            image_path,
            cls=True
        )

        if not result:
            return ""

        extracted_text = []

        for page in result:

            if not page:
                continue

            for line in page:

                if not line or len(line) < 2:
                    continue

                text_info = line[1]

                if not text_info:
                    continue

                text = str(
                    text_info[0]
                ).strip()

                score = float(
                    text_info[1]
                )

                if (
                    text
                    and score >= 0.50
                ):
                    extracted_text.append(
                        text
                    )

        return "\n".join(
            extracted_text
        )

    except Exception as e:

        st.error(
            f"OCR processing failed: {str(e)}"
        )

        raise
