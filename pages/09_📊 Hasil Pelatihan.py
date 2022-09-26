import numpy as np
import pandas as pd
import streamlit as st
from scripts.skripsi import hasil, plot_highest_accuracy

np.random.seed(42)
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Perbandingan akurasi",
    layout="centered",
    page_icon="random",
)
akurasi_dict = {
    "Algoritma": [model for model in hasil["akurasi"].keys()],
    "Akurasi": [f"{akurasi} %" for akurasi in hasil["akurasi"].values()],
}
if "trained" not in st.session_state:
    st.session_state.trained = False
if st.session_state.trained and hasil["akurasi"]:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 10px'>Perbandingan Akurasi</h1>",
        unsafe_allow_html=True,
)
    st.write(
        "Berikut adalah perbandingan akurasi dari setiap algoritma yang digunakan:"
    )
    akurasi = pd.DataFrame.from_dict(akurasi_dict)
    akurasi.index += 1
    st.table(akurasi)
    st.pyplot(plot_highest_accuracy(hasil), use_column_width=True)
else:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 80px'>Perbandingan Akurasi</h1>",
        unsafe_allow_html=True,
    )
    st.warning("Silakan latih model terlebih dahulu.", icon="⚠️")
