import streamlit as st
from ultralytics import YOLO
import os
import tempfile
import cv2
from io import BytesIO

st.title("Deteksi Kata BISINDO dari Video")

MODEL_DIR = 'weights'
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pt')]

st.sidebar.header("Pengaturan Model")
selected_model = st.sidebar.selectbox("Pilih Model", model_files)
confidence_threshold = st.sidebar.slider("Tentukan Nilai Confidence", 0.0, 1.0, 0.7, 0.01)
uploaded_video = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    st.video(video_path)

    stframe = st.empty()
    caption_placeholder = st.empty()

    if st.sidebar.button("Deteksi Kata", type="primary"):

        try:
            model = YOLO(os.path.join(MODEL_DIR, selected_model))
            cap = cv2.VideoCapture(video_path)

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

            output_path_caption = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            output_caption = cv2.VideoWriter(output_path_caption, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

            detected_words = set()
            all_detected_words_per_frame = []

            st.subheader("Hasil Deteksi")

            for _ in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                result = model.predict(frame, conf=confidence_threshold)
                boxes = result[0].boxes
                predicted_classes = boxes.cls.tolist() if boxes else []
                class_names = result[0].names
                labels = [class_names[int(c)] for c in predicted_classes]

                detected_words.update(labels)

                if labels:
                    all_detected_words_per_frame.extend(labels)

                text_overlay = " ".join(sorted(set(labels)))
                result_img = result[0].plot() 
              
                stframe.image(result_img, channels="BGR")
                out.write(result_img)

                caption_frame = frame.copy()
                if labels:
                    text_size, _ = cv2.getTextSize(text_overlay, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
                    text_width = text_size[0]
                    x = (frame_width - text_width) // 2
                    y = frame_height - 50  # tetap di bawah frame, 50px dari bawah
                    cv2.putText(caption_frame, text_overlay, (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)


                output_caption.write(caption_frame)

            cap.release()
            out.release()
            output_caption.release()

            final_text = " ".join(sorted(detected_words)).strip()

            filtered_sequence = []
            prev_word = None
            for word in all_detected_words_per_frame:
                if word != prev_word:
                    filtered_sequence.append(word)
                prev_word = word

            final_sequence_text = " ".join(filtered_sequence).strip()
            st.info(f"{final_sequence_text}")

            caption_buffer = BytesIO()
            with open(output_path_caption, "rb") as f:
                caption_buffer.write(f.read())
            caption_buffer.seek(0)

            st.download_button(
                label="Download Hasil Prediksi",
                data=caption_buffer,
                file_name="hasil_deteksi.mp4",
                mime="video/mp4",
                type="primary"
            )


        except Exception as e:
            st.error(f"Terjadi kesalahan saat proses deteksi: {e}")

else:
    st.warning("Silakan upload video terlebih dahulu.")
