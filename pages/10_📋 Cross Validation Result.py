import numpy as np
import pandas as pd
import streamlit as st
from scripts.skripsi import trained

np.random.seed(42)
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Confusion Matrix",
    layout="centered",
    page_icon="random",
)

akurasi_dict = {}
time_dict = {}
try:
    for key in trained["result"].keys():
        akurasi_dict[key] = np.array(
            [akurasi * 100 for akurasi in trained["result"][key]["test_score"]]
        ).round(2)
        time_dict[key] = np.array(
            [time for time in trained["result"][key]["score_time"]]
        ).round(4)
except:
    pass
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "trained" not in st.session_state:
    st.session_state.trained = False

if st.session_state.trained and akurasi_dict:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 10px'>Hasil Cross Validation</h1>",
        unsafe_allow_html=True,
    )
    akurasi_df = pd.DataFrame(akurasi_dict)
    akurasi_df.index += 1
    time_df = pd.DataFrame(time_dict)
    time_df.index += 1
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("# Akurasi")
        st.dataframe(
            akurasi_df.T.style.format("{:.2f}")
            .highlight_max(axis=1, color="#FFCB42")
            .highlight_min(axis=1, color="#EE6983")
        )
        st.write("# Waktu")
        st.dataframe(
            time_df.T.style.format("{:.4f}")
            .highlight_min(axis=1, color="#FFCB42")
            .highlight_max(axis=1, color="#EE6983")
        )
    with col2:
        st.write("# Rata-rata", unsafe_allow_html=True)
        ratarata_akurasi = akurasi_df.mean(axis=0).to_frame()
        ratarata_akurasi.rename(columns={0: "Rata-Rata"}, inplace=True)
        st.dataframe(ratarata_akurasi.style.format("{:.2f}"))
        st.write("# Rata-rata", unsafe_allow_html=True)
        ratarata_waktu = time_df.mean(axis=0).to_frame()
        ratarata_waktu.rename(columns={0: "Rata-Rata"}, inplace=True)
        st.dataframe(ratarata_waktu.style.format("{:.4f}"))
else:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 80px'>Hasil Cross Validation</h1>",
        unsafe_allow_html=True,
    )
    st.warning("Silakan latih model terlebih dahulu.", icon="⚠️")
