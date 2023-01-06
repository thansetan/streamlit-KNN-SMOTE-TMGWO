import time

import numpy as np

np.random.seed(42)
import streamlit as st

from scripts.skripsi import train_model, trained

st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Hasil Pelatihan Algoritma KNN",
    layout="centered",
    page_icon="random",
)
if "oversampled" not in st.session_state:
    st.session_state.oversampled = False
if "trained" not in st.session_state:
    st.session_state.trained = False
if st.session_state.oversampled:
    X = st.session_state.X
    y = st.session_state.y
    X_sf = st.session_state.X_sf
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 80px'>Latih Model</h1>",
        unsafe_allow_html=True,
    )
    if "knn_trained" not in st.session_state or not trained["akurasi"]:
        st.session_state.knn_trained = False
    if "knn_tmgwo_trained" not in st.session_state or not trained["akurasi"]:
        st.session_state.knn_tmgwo_trained = False
    if "knn_smote_trained" not in st.session_state or not trained["akurasi"]:
        st.session_state.knn_smote_trained = False
    if "knn_smote_tmgwo_trained" not in st.session_state or not trained["akurasi"]:
        st.session_state.knn_smote_tmgwo_trained = False
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        placeholder1 = st.empty()
        if placeholder1.button(
            "Latih Model K-Nearest Neighbors",
            key="KNN",
            disabled=True if st.session_state.knn_trained else False,
        ):
            placeholder1.empty()
            st.button(
                "Latih Model K-Nearest Neighbors",
                key="KNN_disabled",
                disabled=True,
            )
            st.session_state.knn_trained = True
            with st.spinner("melatih model dengan KNN..."):
                time.sleep(1)
                model_knn = train_model(
                    X,
                    y,
                    "KNN",
                )

            placeholder1.empty()
        if st.session_state.knn_trained:
            st.success("Selesai")

    with col2:
        placeholder2 = st.empty()
        if placeholder2.button(
            "Latih Model KNN dengan TMGWO",
            key="KNN_TMGWO",
            disabled=True if st.session_state.knn_tmgwo_trained else False,
        ):
            placeholder2.empty()
            st.button(
                "Latih Model KNN dengan TMGWO",
                key="KNN_TMGWO_disabled",
                disabled=True,
            )
            st.session_state.knn_tmgwo_trained = True
            with st.spinner("melatih model dengan KNN dan TMGWO..."):
                time.sleep(1)
                model_knn_tmgwo = train_model(
                    X_sf,
                    y,
                    "KNN+TMGWO",
                )
            placeholder2.empty()
        if st.session_state.knn_tmgwo_trained:
            st.success("Selesai")
    
    with col3:
        placeholder3 = st.empty()
        if placeholder3.button(
            "Latih Model KNN dengan SMOTE",
            key="KNN_SMOTE",
            disabled=True if st.session_state.knn_smote_trained else False,
        ):
            placeholder3.empty()
            st.button(
                "Latih Model KNN dengan SMOTE",
                key="KNN_SMOTE_disabled",
                disabled=True,
            )
            st.session_state.knn_smote_trained = True
            with st.spinner("melatih model dengan KNN dan SMOTE..."):
                time.sleep(1)
                model_knn_smote = train_model(
                    X,
                    y,
                    "KNN+SMOTE",
                )
            placeholder3.empty()
        if st.session_state.knn_smote_trained:
            st.success("Selesai")
    
    with col4:
        placeholder4 = st.empty()
        if placeholder4.button(
            "Latih Model KNN dengan SMOTE dan TMGWO",
            key="KNN_SMOTE_TMGWO",
            disabled=True if st.session_state.knn_smote_tmgwo_trained else False,
        ):
            st.button(
                "Latih Model KNN dengan SMOTE dan TMGWO",
                key="KNN_SMOTE_TMGWO_Disabled",
                disabled=True,
            )
            placeholder4.empty()
            st.session_state.knn_smote_tmgwo_trained = True
            with st.spinner("melatih model dengan KNN dan SMOTE dan TMGWO..."):
                time.sleep(1)
                cr_knn_smote_tmgwo = train_model(
                    X_sf,
                    y,
                    "KNN+SMOTE+TMGWO",
                )
            placeholder4.empty()
        if st.session_state.knn_smote_tmgwo_trained:
            st.success("Selesai")
        if (
            st.session_state.knn_trained
            or st.session_state.knn_smote_trained
            or st.session_state.knn_smote_tmgwo_trained
        ):
            st.session_state.trained = True
else:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 80px'>Latih Model</h1>",
        unsafe_allow_html=True,
    )
    st.warning(
        "Silakan lakukan pengolahan data (feature selection, oversampling) terlebih dahulu.",
        icon="⚠️",
    )
