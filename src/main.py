import streamlit as st
import cv2
import numpy as np
import time

try:
    from image_compressor import compress_image
except ModuleNotFoundError:
    from src.image_compressor import compress_image

st.set_page_config(page_title="Tucker Tensor Compressor", layout="wide")

st.title("Sistem de Compresie Tensorială (Tucker Decomposition & HOSVD)")

compressionFactor = st.sidebar.slider("Factor de compresie", 0.05, 1.0, 0.5, 0.05)
blockSize = st.sidebar.selectbox("Dimensiune bloc (Pixeli)", [8, 16, 32], index=1)

uploadedFile = st.file_uploader("Selectați imaginea sursă pentru compresare", type=["jpg", "jpeg", "png"])

if uploadedFile is not None:
    tempPath = "temp_input.png"
    with open(tempPath, "wb") as f:
        f.write(uploadedFile.getbuffer())
        
    try:
        startTime = time.time()
        imgCompressed, compRate, relativeError, absoluteError = compress_image(
            tempPath,
            compressionFactor=compressionFactor,
            blockSize=blockSize
        )
        execTime = time.time() - startTime
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(cv2.imread(tempPath), cv2.COLOR_BGR2RGB), caption="Imagine originală", use_column_width=True)
        with col2:
            st.image(imgCompressed, caption="Reconstrucție Tucker", use_column_width=True)
            
        st.write(f"**Raport de compresie stocare:** {compRate:.2f}x")
        st.write(f"**Eroare relativă de reconstrucție (Norma Frobenius):** {relativeError:.2f}%")
        st.write(f"**Timp de execuție:** {execTime:.3f} s")
        
        imgCompressedBgr = cv2.cvtColor(imgCompressed, cv2.COLOR_RGB2BGR)
        isSuccess, buffer = cv2.imencode(".jpg", imgCompressedBgr)
        if isSuccess:
            st.download_button(
                label="Descărcare rezultat reconstrucție (Format JPG)",
                data=buffer.tobytes(),
                file_name="reconstructie_tucker.jpg",
                mime="image/jpg"
            )
    except Exception as e:
        st.error(f"Eroare întâmpinată: {e}")