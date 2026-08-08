import streamlit as st

st.title("MedScanAI")

try:
    import cv2
    st.success(f"OpenCV works: {cv2.__version__}")
except Exception as e:
    st.error("OpenCV failed")
    st.exception(e)
    st.stop()

try:
    import paddle
    st.success(f"PaddlePaddle works: {paddle.__version__}")
except Exception as e:
    st.error("PaddlePaddle failed")
    st.exception(e)
    st.stop()

try:
    from paddleocr import PaddleOCR
    st.success("PaddleOCR imported successfully")
except Exception as e:
    st.error("PaddleOCR failed")
    st.exception(e)
    st.stop()
