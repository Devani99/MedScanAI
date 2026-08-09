"""OCR service for MedScanAI.

Uses EasyOCR for reliable, dependency-light text extraction from medicine
package images. EasyOCR is pure Python, has no Linux system-package
dependencies, and is compatible with Python 3.12 and Streamlit Community Cloud.

The reader is cached with @st.cache_resource so it is initialized only once
per Streamlit session – the first call downloads small language model files
(~50 MB) and subsequent calls are fast.
"""

from __future__ import annotations

import logging

import streamlit as st

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def _get_reader():
    """Initialize and cache the EasyOCR reader."""
    import easyocr  # imported here so the module can be imported without easyocr installed (CI)
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return reader


def extract_medicine_text(image_path: str) -> str:
    """Extract text from a medicine package image.

    Parameters
    ----------
    image_path:
        Absolute path to the image file.

    Returns
    -------
    str
        Clean extracted text, one detected line per row.
        Returns an empty string when no text is found.
    """
    try:
        reader = _get_reader()
        results = reader.readtext(image_path, detail=1)
    except Exception as exc:
        logger.error("EasyOCR failed: %s", exc)
        return ""

    lines: list[str] = []
    for _bbox, text, confidence in results:
        text = str(text).strip()
        if not text:
            continue
        # Keep text with confidence >= 0.4 to reduce noise
        if confidence >= 0.4:
            lines.append(text)

    return "\n".join(lines)
