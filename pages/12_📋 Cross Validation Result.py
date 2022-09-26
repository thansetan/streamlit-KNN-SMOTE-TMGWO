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

akurasi_df = pd.DataFrame()
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "knn_trained" not in st.session_state or not hasil['akurasi']:
    st.session_state.knn_trained = False
if "knn_smote_trained" not in st.session_state or not hasil['akurasi']:
    st.session_state.knn_smote_trained = False
if "knn_smote_tmgwo_trained" not in st.session_state or not hasil['akurasi']:
    st.session_state.knn_smote_tmgwo_trained = False
if "trained" not in st.session_state:
    st.session_state.trained = False

if st.session_state.knn_trained:
    akurasi_df = akurasi_df.assign(KNN=pd.read_csv("cross_val_results/knn_acc.csv"))
if st.session_state.knn_smote_trained:
    akurasi_df = akurasi_df.assign(KNN_SMOTE=pd.read_csv("cross_val_results/knn_smote_acc.csv"))
    akurasi_df.rename(columns={"KNN_SMOTE": "KNN+SMOTE"}, inplace=True)
if st.session_state.knn_smote_tmgwo_trained:
    akurasi_df = akurasi_df.assign(KNN_SMOTE_TMGWO=pd.read_csv("cross_val_results/knn_smote_tmgwo_acc.csv"))
    akurasi_df.rename(columns={"KNN_SMOTE_TMGWO": "KNN+SMOTE+TMGWO"}, inplace=True)
    num_features = pd.read_csv("cross_val_results/num_selected_features.csv")
if st.session_state.trained and hasil['akurasi']:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 10px'>Hasil Cross Validation</h1>",
        unsafe_allow_html=True,
)
    datacek = pd.read_csv(f"datasets/{st.session_state.dataset}")
    if datacek.columns[0] != "Age":
        st.error("Ndak tahu krn bkn data diabetes", icon="🗿")
    else:
        akurasi_df = akurasi_df.round(2)
        akurasi_df.index+=1
        col1,col2 = st.columns([2,1])
        with col1:
            st.write("# Akurasi")
            st.dataframe(akurasi_df.T.style.format("{:.2f}").highlight_max(axis=1,color='#FFCB42').highlight_min(axis=1, color='#EE6983'))
            if st.session_state.knn_smote_tmgwo_trained:
                st.write("# Jumlah Fitur Terpilih")
                num_features.index+=1
                num_features.rename(columns = {"0":'Jumlah Fitur'}, inplace = True)
                st.dataframe(num_features.T.style.highlight_max(axis=1,color='#FFCB42').highlight_min(axis=1, color='#EE6983'))
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
else:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 80px'>Hasil Cross Validation</h1>",
        unsafe_allow_html=True,
    )
    st.warning("Silakan latih model terlebih dahulu.", icon="⚠️")
