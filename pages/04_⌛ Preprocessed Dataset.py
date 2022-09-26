from time import sleep

import numpy as np
import streamlit as st

np.random.seed(42)
import pandas as pd
from scripts.skripsi import preprocess

if "dataset" not in st.session_state:
    st.session_state.dataset = None

st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Preprocessed Dataset",
    layout="centered",
    page_icon="random",
)
if "preprocessed" not in st.session_state:
    st.session_state.preprocessed = False
if "preprocessed_data" not in st.session_state:
    st.session_state.preprocessed_data = None
if st.session_state.dataset:
    data = pd.read_csv(f"datasets/{st.session_state.dataset}")
    data.index += 1
    if not st.session_state.preprocessed:
        col1, col2, col3 = st.columns(3)
        with col2:
            with st.spinner("Preprocessing data..."):
                sleep(1)
            st.session_state.preprocessed = True
    preprocessed_data = preprocess(data)
    st.session_state.preprocessed_data = preprocessed_data
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 10px'>Preprocessed Dataset</h1>",
        unsafe_allow_html=True,
    )
    st.write(
        "Dataset yang berisi data integer/mumerik/kontinyu/opolah telah dinormalisasi dengan MinMaxScaler dan data yang bersifat categorical telah diubah menjadi mumerik"
    )
    st.dataframe(preprocessed_data)
else:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 80px'>Preprocessed Dataset</h1>",
        unsafe_allow_html=True,
    )
    st.warning("Silakan pilih dataset terlebih dahulu.", icon="⚠️")
