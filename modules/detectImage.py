import streamlit as st
from ultralytics import YOLO
import os
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

st.title("Deteksi Kata BISNDO dari Gambar")
    
MODEL_DIR = 'weights'
    
st.sidebar.header("Model Configuration")
model_type = st.sidebar.selectbox("Pilih Model", [f for f in os.listdir(MODEL_DIR) if f.endswith('.pt')])
confidence_threshold = st.sidebar.slider("Tentukan Nilai Confidence", 0.0, 1.0, 0.7, 0.01)

uploaded_image = st.sidebar.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    
    st.subheader("Gambar Asli")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.sidebar.button("Deteksi Kata", type="primary"):
        st.subheader("Hasil Deteksi")
        
        try:
            model = YOLO(os.path.join(MODEL_DIR, model_type))
            result = model.predict(image, conf=confidence_threshold)
            
            result_plotted = result[0].plot()[:, :, ::-1]

            boxes = result[0].boxes
            predicted_classes = boxes.cls.tolist() if boxes else []
            class_names = result[0].names
            labels = [class_names[int(c)] for c in predicted_classes]
            detected_text = " ".join(labels).strip() if labels else "Tidak ada yang terdeteksi"

            image_with_text = image.copy()
            draw = ImageDraw.Draw(image_with_text)
            
            font = ImageFont.truetype("arial.ttf", size=32)
            
            bbox = draw.textbbox((0, 0), detected_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            position = ((image.width - text_width) // 2, image.height - text_height - 40)
            draw.text(position, detected_text, fill="yellow", stroke_fill="black", stroke_width=1 , font=font)
            
            st.info(f"{detected_text}")
        
            col1, col2 = st.columns(2)

            with col1:
                st.image(result_plotted, caption="BoundingBox", use_container_width=True)
            with col2:
                st.image(image_with_text, caption="Caption", use_container_width=True)
            
            buffer = BytesIO()
            image_with_text.save(buffer, format="JPEG")
            buffer.seek(0)
            
            st.download_button(
                label="Download Hasil Prediksi",
                data=buffer,
                file_name="hasil-deteksi.jpg",
                mime="image/jpeg",
                type="primary"
            )
        except Exception as e:
            st.error(f"Error during detection: {e}")
            st.stop()
else:
    st.warning("Silakan upload gambar terlebih dahulu.")
