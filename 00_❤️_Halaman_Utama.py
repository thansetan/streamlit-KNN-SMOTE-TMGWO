import numpy as np
import streamlit as st

np.random.seed(42)
if "cm" not in st.session_state:
    st.session_state.cm = {}
if "acc" not in st.session_state:
    st.session_state.acc = {}
judul = "Implementasi SMOTE dan TMGWO pada Deteksi Dini Penyakit Diabetes menggunakan KNN"
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title=judul,
    layout="centered",
    page_icon="random",
)

col1, col2, col3 = st.columns(3)
with col2:
    st.image("images/unnes.png")
st.markdown(
    "<h1 style='text-align: center'>Skripsi</h1>", unsafe_allow_html=True
)
st.markdown(
    f"<h1 style='text-align: center'>{judul}</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h2 style='text-align: center'>Disusun oleh: Fathan Arsyadani</h2>",
    unsafe_allow_html=True,
)
