import streamlit as st

st.title("MedScanAI")

try:
    import cv2

    st.success(
        f"OpenCV works: {cv2.__version__}"
    )

except Exception as e:

    st.error("OpenCV failed")

    st.exception(e)
