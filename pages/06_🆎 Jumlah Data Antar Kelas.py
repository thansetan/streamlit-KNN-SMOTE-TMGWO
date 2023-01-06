import numpy as np
import streamlit as st

from scripts.skripsi import get_data_antar_kelas, plot_data_antar_kelas

np.random.seed(42)

if "X" not in st.session_state:
    st.session_state.X = None
if "y" not in st.session_state:
    st.session_state.y = None
if "X_sf" not in st.session_state:
    st.session_state.X_sf = None
if "preprocessed" not in st.session_state:
    st.session_state.preprocessed = False
if "splitted" not in st.session_state:
    st.session_state.splitted = False
if "posneg" not in st.session_state:
    st.session_state.posneg = None
if "hasil_smote" not in st.session_state:
    st.session_state.hasil_smote = None
if st.session_state.splitted:
    X_sf = st.session_state.X_sf
    y = st.session_state.y
    st.set_page_config(
        initial_sidebar_state="collapsed",
        page_title="Jumlah data antar kelas",
        layout="centered",
        page_icon="random",
    )
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 10px'>Jumlah Data Antar Kelas</h1>",
        unsafe_allow_html=True,
    )
    posneg, hasil_smote = get_data_antar_kelas(X_sf, y)
    st.session_state.posneg = posneg
    st.session_state.hasil_smote = hasil_smote
    for key, val in posneg.items():
        st.pyplot(plot_data_antar_kelas(val['before_smote'], key), use_column_width=True)
else:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 80px'>Jumlah Data Antar Kelas</h1>",
        unsafe_allow_html=True,
    )
    st.warning(
        "Tolong bukanya dari atas-bawah, jangan dilewatin. terima kasih.", icon="⚠️"
    )
