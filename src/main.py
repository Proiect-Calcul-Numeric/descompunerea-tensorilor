import streamlit as st
import cv2
import numpy as np
import time
import os
from image_compressor import compress_image

# Configurare pagina Streamlit
st.set_page_config(
    page_title="Image Compressor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Sistem de Compresie Tensorială (Tucker Decomposition & HOSVD)")

compressionFactor = st.sidebar.slider(
    "Factor de compresie spectrală (Calitate)",
    min_value=0.05,
    max_value=1.0,
    value=0.5,
    step=0.05,
)

blockSize = st.sidebar.selectbox(
    "Dimensiune bloc spațial (Pixeli)",
    options=[8, 16, 32],
    index=1,
    help="Imaginea este împărțită în blocuri de BxB pixeli. Dimensiunea 16x16 este optimă."
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Cum funcționează?**
Aplicația împarte imaginea în blocuri pătrate, aplică descompunerea HOSVD custom pe fiecare bloc și trunchiază valorile singulare mai puțin importante, reducând drastic spațiul ocupat.
""")

# --- ZONA PRINCIPALĂ ---
uploadedFile = st.file_uploader(
    "Selectați imaginea sursă pentru analiză spectrală",
    type=["jpg", "jpeg", "png"],
    help="Suportă fișiere JPG, JPEG și PNG."
)

if "compressedImage" not in st.session_state:
    st.session_state.compressedImage = None
    st.session_state.compRate = None
    st.session_state.relativeError = None
    st.session_state.execTime = None

if uploadedFile is not None:
    # Salvăm fișierul încărcat într-o locație temporară fixă din spațiul de lucru
    # Folosim o cale fixă pentru a evita blocarea fișierelor de către Windows în procese concurente
    tempPath = "temp_input.png"
    fileBytes = uploadedFile.getvalue()
    fileHash = hash(fileBytes)
    
    if "fileHash" not in st.session_state or st.session_state.fileHash != fileHash:
        st.session_state.fileHash = fileHash
        st.session_state.compressedImage = None
        
    with open(tempPath, "wb") as f:
        f.write(fileBytes)
        
    if st.button("Execută compresia tensorială"):
        try:
            # Spinner de încărcare în timpul rulării HOSVD
            with st.spinner("Se comprimă imaginea.."):
                startTime = time.time()
                progressBar = st.progress(0.0)
                
                def updateProgress(percent):
                    progressBar.progress(percent)
                
                # Apelăm funcția din image_compressor.py
                imgCompressed, compRate, relativeError, absoluteError = compress_image(
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
        except Exception as e:
            st.error(f"Eroare întâmpinată în timpul procesării tensorului: {e}")

    if st.session_state.compressedImage is not None:
        # Afișare imagini Side-by-Side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original")
            # Citim imaginea originală pentru a o afișa corect în RGB
            imgOrig = cv2.imread(tempPath)
            imgOrigRgb = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB)
            st.image(imgOrigRgb, use_column_width=True)
            
        with col2:
            st.subheader("Comprimat")
            st.image(st.session_state.compressedImage, use_column_width=True)
            
        st.write(f"**Raport de compresie stocare:** {st.session_state.compRate:.2f}x")
        
        # Procentul de informație păstrată = 100 - eroarea relativă
        infoConserved = max(0.0, 100.0 - st.session_state.relativeError)
        st.write(f"**Eroare relativă de reconstrucție (Norma Frobenius):** {st.session_state.relativeError:.2f}%")
        st.write(f"**Timp de execuție algoritm tensorial:** {st.session_state.execTime:.3f} s")
        
        # Oferim opțiunea de descărcare a imaginii comprimate
        # Convertim imaginea înapoi în BGR pentru a fi codificată corect de OpenCV în format PNG
        imgCompressedBgr = cv2.cvtColor(st.session_state.compressedImage, cv2.COLOR_RGB2BGR)
        isSuccess, buffer = cv2.imencode(".png", imgCompressedBgr)
        
        if isSuccess:
            st.download_button(
                label="Descărcare rezultat reconstrucție (Format PNG)",
                data=buffer.tobytes(),
                file_name="reconstructie_tucker.png",
                mime="image/png"
            )
            
else:
    # Mesaj de întâmpinare când nu este încărcată nicio imagine
    st.info("Pornire rapidă: Încarcă o imagine din panoul de mai sus pentru a începe compresia.")