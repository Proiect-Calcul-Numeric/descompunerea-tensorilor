import streamlit as st
import cv2
import numpy as np
import time
import os
import io
from image_compressor import compress_image

st.set_page_config(
    page_title="Comprimare de Imagini",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Compresarea imaginilor folosind HOSVD si Tucker Decomposition")

compressionFactor = st.sidebar.slider(
    "Factor de compresie",
    min_value=0.05,
    max_value=1.0,
    value=0.5,
    step=0.05,
)

blockSize = st.sidebar.selectbox(
    "Dimensiune bloc (Pixeli)",
    options=[4,8, 16, 32, 64],
    index=1,
)



uploadedFile = st.file_uploader(
    "Selectați imaginea pentru compresie",
    type=["jpg", "jpeg", "png"],
)

if "compressedImage" not in st.session_state:
    st.session_state.compressedImage = None
    st.session_state.compRate = None
    st.session_state.relativeError = None
    st.session_state.execTime = None
    st.session_state.gList = None
    st.session_state.usList = None

if uploadedFile is not None:
    tempPath = "temp_input.png"
    fileIdentifier = f"{uploadedFile.name}_{uploadedFile.size}"
    
    if "fileIdentifier" not in st.session_state or st.session_state.fileIdentifier != fileIdentifier:
        st.session_state.fileIdentifier = fileIdentifier
        st.session_state.compressedImage = None
        
    with open(tempPath, "wb") as f:
        f.write(uploadedFile.getbuffer())
        
    if st.button("Execută compresia"):
        try:
            with st.spinner("Se comprimă imaginea.."):
                startTime = time.time()
                progressBar = st.progress(0.0)
                
                def updateProgress(percent):
                    progressBar.progress(percent)
                
                imgCompressed, compRate, relativeError, absoluteError, gList, usList = compress_image(
                    tempPath,
                    compressionFactor=compressionFactor,
                    blockSize=blockSize,
                    progressCallback=updateProgress
                )
                progressBar.empty()
                execTime = time.time() - startTime
                
                st.session_state.compressedImage = imgCompressed
                st.session_state.compRate = compRate
                st.session_state.relativeError = relativeError
                st.session_state.execTime = execTime
                st.session_state.gList = gList
                st.session_state.usList = usList
        except Exception as e:
            st.error(f"Eroare întâmpinată în timpul compresării: {e}")

    if st.session_state.compressedImage is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original")
            st.image(uploadedFile, width='stretch')
            
        with col2:
            st.subheader("Comprimat")
            st.image(st.session_state.compressedImage, width='stretch')
            
        st.write(f"**Raport de compresie stocare (teoretic):** {st.session_state.compRate:.2f}x")
        st.write(f"**Eroare relativă de reconstrucție (Norma Frobenius):** {st.session_state.relativeError:.2f}%")
        st.write(f"**Timp de execuție:** {st.session_state.execTime:.3f} s")
        
        gArray = np.stack([G.astype(np.float16) for G in st.session_state.gList])
        u1Array = np.stack([Us[0].astype(np.float16) for Us in st.session_state.usList])
        u2Array = np.stack([Us[1].astype(np.float16) for Us in st.session_state.usList])
        u3Array = np.stack([Us[2].astype(np.float16) for Us in st.session_state.usList])

        npzBuffer = io.BytesIO()
        np.savez_compressed(
            npzBuffer,
            G=gArray,
            U1=u1Array,
            U2=u2Array,
            U3=u3Array
        )
        npzBytes = npzBuffer.getvalue()
        
        originalSizeKb = os.path.getsize(tempPath) / 1024.0
        compressedSizeKb = len(npzBytes) / 1024.0
        
        
        st.write(f"**Dimensiune imagine originală:** {originalSizeKb:.2f} KB")
        
        
        
            
        st.markdown("**Opțiuni Descărcare:**")
        
        imgCompressedBgr = cv2.cvtColor(st.session_state.compressedImage, cv2.COLOR_RGB2BGR)
        isSuccess, jpegBuffer = cv2.imencode(".jpg", imgCompressedBgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        if isSuccess:
            st.download_button(
                label="Descărcare imagine reconstituită (Format JPG)",
                data=jpegBuffer.tobytes(),
                file_name="reconstructie_tucker.jpg",
                mime="image/jpeg"
            )
            
        st.download_button(
            label="Descărcare date Tucker comprimate (Format NPZ)",
            data=npzBytes,
            file_name="date_tucker_comprimate.npz",
            mime="application/octet-stream"
        )