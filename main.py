# Impor library yang dibutuhkan
import cv2
import streamlit as st
from pathlib import Path
import sys
from ultralytics import YOLO
from PIL import Image

# Mendapatkan path absolut dari file ini
FILE = Path(__file__).resolve()
try:
    ROOT = FILE.parent.relative_to(Path.cwd())
except ValueError:
    ROOT = FILE.parent

# Menambahkan path ke sys.path jika belum ada
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) 

# Konstanta untuk jenis input
IMAGE = 'Image'
VIDEO = 'Video'
LIVE = 'Live'

SOURCES_LIST = [IMAGE, VIDEO, LIVE]

# Path direktori
IMAGES_DIR = ROOT / 'images'

VIDEOS_DIR = ROOT / 'videos'

MODEL_DIR = ROOT / 'weights'

Y11N_DETECTION_MODEL = MODEL_DIR / 'bisindo-detection-yolo11n.pt'
Y11X_DETECTION_MODEL = MODEL_DIR / 'bisindo-detection-yolo11x.pt'

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="BISINDO Detector",
    page_icon="🙌",
)

st.header("Sign Length Detector with YOLO11")

# Sidebar: Konfigurasi Model
st.sidebar.header("Model Configuration")
model_type = st.sidebar.radio("Select Model Type", ["YOLO11n", "YOLO11x"])
confidence_threshold = float(st.sidebar.slider("Set Confidence Threshold", 20, 100, 60)) / 100

# Load model
try:
    if model_type == "YOLO11n":
        model = YOLO(Y11N_DETECTION_MODEL)
    else:
        model = YOLO(Y11X_DETECTION_MODEL)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar: Konfigurasi Input
st.sidebar.header("Input Configuration")
source_radio = st.sidebar.radio("Select Input Source", SOURCES_LIST)

source_image = None
if source_radio == IMAGE:
    source_image = st.sidebar.file_uploader(
        "Choose an Image....", type = ("jpg", "png", "jpeg", "bmp", "webp")
    )
    col1, col2 = st.columns(2)
    with col1:
        try:
            if source_image is not None:
                uploaded_image  =Image.open(source_image)
                st.image(source_image, caption = "Uploaded Image", use_column_width = True)
            else:
                st.error("Please Upload an Image")
        except Exception as e:
            st.error("Error Occured While Opening the Image")
            st.error(e)
    with col2:
        try:
            if source_image is not None:
                if st.sidebar.button("Detect Objects"):
                    result = model.predict(uploaded_image, conf = confidence_threshold)
                    boxes = result[0].boxes
                    result_plotted = result[0].plot()[:,:,::-1]
                    st.image(result_plotted, caption = "Detected Image", use_column_width = True)

                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.data)
                    except Exception as e:
                        st.error(e)
            else:
                st.error("Please Upload an Image")
        except Exception as e:
            st.error("Error Occured While Opening the Image")
            st.error(e)

elif source_radio == VIDEO:
    uploaded_video = st.sidebar.file_uploader("Upload a Video...", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Tampilkan video
        st.video(uploaded_video)

        if st.sidebar.button("Detect Video Objects"):
            try:
                # Simpan video ke file sementara
                with open("temp_video.mp4", "wb") as f:
                    f.write(uploaded_video.read())

                video_cap = cv2.VideoCapture("temp_video.mp4")
                st_frame = st.empty()

                while video_cap.isOpened():
                    success, image = video_cap.read()
                    if success:
                        image = cv2.resize(image, (720, int(720 * (9/16))))
                        # Predict the objects in the image using YOLO
                        result = model.predict(image, conf=confidence_threshold)
                        # Plot the detected objects on the video frame
                        result_plotted = result[0].plot()
                        st_frame.image(result_plotted, caption="Detected Video",
                                       channels="BGR",
                                       use_column_width=True)
                    else:
                        video_cap.release()
                        break
            except Exception as e:
                st.sidebar.error("Error processing video: " + str(e))

