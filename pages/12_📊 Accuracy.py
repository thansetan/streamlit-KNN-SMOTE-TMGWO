import numpy as np
import pandas as pd
import streamlit as st
from scripts.skripsi import plot_acc, trained

np.random.seed(42)
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Perbandingan akurasi",
    layout="centered",
    page_icon="random",
)
akurasi_dict = {
    "Algoritma": [model for model in trained["akurasi"].keys()],
    "Akurasi": [
        f"{round(akurasi*100, 2)} %" for akurasi in trained["akurasi"].values()
    ],
}
if "trained" not in st.session_state:
    st.session_state.trained = False
if st.session_state.trained and trained["akurasi"]:
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
    st.pyplot(plot_acc(trained["akurasi"]), use_column_width=True)
else:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 80px'>Perbandingan Akurasi</h1>",
        unsafe_allow_html=True,
    )
    st.warning("Silakan latih model terlebih dahulu.", icon="⚠️")
