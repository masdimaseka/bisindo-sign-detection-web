import streamlit as st

st.set_page_config(
    page_title="Deteksi BISINDO",
    page_icon="🙌",
    layout="centered",
    initial_sidebar_state="auto"
)   

pages = {
            "Menu": [
                st.Page("modules/home.py", title="Home", icon="🏡"),
                st.Page("modules/listKata.py", title="List Kata BISINDO", icon="📚"), 
            ],
            "Metode deteksi": [
                st.Page("modules/detectImage.py", title="Deteksi dari Gambar", icon="📷"),
                st.Page("modules/detectVideo.py", title="Deteksi dari Video", icon="📹"), 
                st.Page("modules/detectRealTime.py", title="Deteksi Real Time", icon="📽️"),
            ]
        }

pg = st.navigation(pages)
pg.run()