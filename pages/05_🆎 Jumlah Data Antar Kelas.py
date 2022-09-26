import numpy as np
import streamlit as st
from scripts.skripsi import data_antar_kelas, split_preprocessed

np.random.seed(42)

if "X" not in st.session_state:
    st.session_state.X = None
if "y" not in st.session_state:
    st.session_state.y = None
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_train" not in st.session_state:
    st.session_state.y_train = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "preprocessed" not in st.session_state:
    st.session_state.preprocessed = False
if "splitted" not in st.session_state:
    st.session_state.splitted = False
if st.session_state.preprocessed:
    X, y, X_train, X_test, y_train, y_test = split_preprocessed(st.session_state.preprocessed_data)
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.splitted = True
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
    st.pyplot(data_antar_kelas(y_train), use_column_width=True)
else:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 80px'>Jumlah Data Antar Kelas</h1>",
        unsafe_allow_html=True,
    )
    st.warning("Silakan lakukan preprocessing terlebih dahulu.", icon="⚠️")
