import numpy as np
import pandas as pd
import streamlit as st
from scripts.skripsi import hasil

np.random.seed(42)
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Confusion Matrix",
    layout="centered",
    page_icon="random",
)

akurasi_dict = {}
try:
    for key in hasil['cross_val']['akurasi'].keys():
        akurasi_dict[key] = [akurasi for akurasi in hasil['cross_val']['akurasi'][key]]
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
    datacek = pd.read_csv(f"datasets/{st.session_state.dataset}")
    if datacek.columns[0] != "Age":
        st.error("Ndak tahu krn bkn data diabetes", icon="🗿")
    else:
        akurasi_df = pd.DataFrame(akurasi_dict).round(2)
        akurasi_df.index+=1
        col1,col2 = st.columns([2,1])
        with col1:
            st.write("# Akurasi")
            st.dataframe(akurasi_df.T.style.format("{:.2f}").highlight_max(axis=1,color='#FFCB42').highlight_min(axis=1, color='#EE6983'))
            if st.session_state.knn_smote_tmgwo_trained:
                st.write("# Jumlah Fitur Terpilih")
                num_features = pd.DataFrame(hasil["cross_val"]["num_sf"])
                num_features.index+=1
                num_features.rename(columns = {0:'Jumlah Fitur'}, inplace = True)
                st.dataframe(num_features.T)
        with col2:
            st.write("# Rata-rata", unsafe_allow_html=True)
            ratarata = akurasi_df.mean(axis=0).to_frame()
            ratarata.rename(columns = {0:'Rata-Rata'}, inplace = True)
            st.dataframe(ratarata.style.format("{:.2f}"))
            if st.session_state.knn_smote_tmgwo_trained:
                st.write("# Rata-rata", unsafe_allow_html=True)
                ratarata_features = num_features.mean(axis=0).to_frame()
                ratarata_features.rename(columns = {0:'Rata-Rata'}, inplace = True)
                st.dataframe(ratarata_features.style.format("{:.1f}"))
elif st.session_state.trained and not akurasi_dict:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 80px'>Hasil Cross Validation</h1>",
        unsafe_allow_html=True,
    )
    st.warning("Ndak ada hasil cross validation", icon="⚠️")
else:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 80px'>Hasil Cross Validation</h1>",
        unsafe_allow_html=True,
    )
    st.warning("Silakan latih model terlebih dahulu.", icon="⚠️")
