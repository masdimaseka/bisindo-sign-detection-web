import streamlit as st
from ultralytics import YOLO
import os
import cv2

st.title("Deteksi Kata BISINDO secara Real Time")


MODEL_DIR = 'weights'
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pt')]

st.sidebar.header("Pengaturan Model")
selected_model = st.sidebar.selectbox("Pilih Model", model_files)
confidence_threshold = st.sidebar.slider("Tentukan Nilai Confidence", 0.0, 1.0, 0.7, 0.01)

model_path = os.path.join(MODEL_DIR, selected_model)
model = YOLO(model_path)


if "detecting" not in st.session_state:
    st.session_state.detecting = False

if st.sidebar.button("Mulai/Stop Deteksi", type="primary"):
    st.session_state.detecting = not st.session_state.detecting

stframe = st.empty()

if st.session_state.detecting:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Webcam tidak dapat diakses.")
    else:
        while st.session_state.detecting:
            ret, frame = cap.read()
            if not ret:
                st.warning("Gagal membaca frame dari webcam.")
                break

            results = model(frame, conf=confidence_threshold)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="BGR")
            
        

    cap.release()
    cv2.destroyAllWindows()
