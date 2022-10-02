import time

import numpy as np

np.random.seed(42)
import streamlit as st
from scripts.skripsi import hasil, train_model

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
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    X_train_smote = st.session_state.X_train_smote
    y_train_smote = st.session_state.y_train_smote
    selected_features = st.session_state.selected_features
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 80px'>Latih Model</h1>",
        unsafe_allow_html=True,
    )
    if "knn_trained" not in st.session_state or not hasil['akurasi']:
        st.session_state.knn_trained = False
    if "knn_crossval" not in st.session_state:
        st.session_state.knn_crossval = False
    if "knn_smote_crossval" not in st.session_state:
        st.session_state.knn_smote_crossval = False
    if "knn_smote_tmgwo_crossval" not in st.session_state:
        st.session_state.knn_smote_tmgwo_crossval = False
    if "knn_smote_trained" not in st.session_state or not hasil['akurasi']:
        st.session_state.knn_smote_trained = False
    if "knn_smote_tmgwo_trained" not in st.session_state or not hasil['akurasi']:
        st.session_state.knn_smote_tmgwo_trained = False    
    # crossval = st.checkbox("Lakukan Cross-Validation juga", 
    #                         value=st.session_state.do_crossval, 
    #                         disabled=True if st.session_state.knn_trained or 
    #                                          st.session_state.knn_smote_trained or
    #                                          st.session_state.knn_smote_tmgwo_trained else False,
    #                         key="cv", 
    #                         help="Jika dicentang, maka akan dilakukan cross-validation juga dan pelatihan akan memakan waktu lebih lama")
    # st.session_state.do_crossval = crossval
    col1, col2, col3 = st.columns(3)
    with col1:
        placeholder1 = st.empty()
        knn_crossval = st.checkbox("Cross-Validation", 
                                    value=st.session_state.knn_crossval, 
                                    disabled=True if st.session_state.knn_trained else False,
                                    key="knn_cv", 
                                    help="Jika dicentang, maka akan dilakukan cross-validation juga dan pelatihan akan memakan waktu lebih lama")
        st.session_state.knn_crossval = knn_crossval
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
                model_knn, score_knn, cm_knn, cr_knn = train_model(X, y, X_train, X_test, y_train, y_test, "KNN",cv=st.session_state.knn_crossval)

            placeholder1.empty()
        if st.session_state.knn_trained:
            st.success("Selesai")

    with col2:
        placeholder2 = st.empty()
        knn_smote_crossval = st.checkbox("Cross-Validation", 
                                    value=st.session_state.knn_smote_crossval, 
                                    disabled=True if st.session_state.knn_smote_trained else False,
                                    key="knn_smote_cv", 
                                    help="Jika dicentang, maka akan dilakukan cross-validation juga dan pelatihan akan memakan waktu lebih lama")
        st.session_state.knn_smote_crossval = knn_smote_crossval
        if placeholder2.button(
            "Latih Model KNN dengan SMOTE",
            key="KNN_SMOTE",
            disabled=True if st.session_state.knn_smote_trained else False,
        ):
            placeholder2.empty()
            st.button(
                "Latih Model KNN dengan SMOTE",
                key="KNN_SMOTE_disabled",
                disabled=True,
            )
            st.session_state.knn_smote_trained = True
            with st.spinner("melatih model dengan KNN dan SMOTE..."):
                time.sleep(1)
                model_knn_smote, score_knn_smote, cm_knn_smote, cr_knn_smote = train_model(
                    X, y, X_train_smote, X_test, y_train_smote, y_test,"KNN+SMOTE", cv=st.session_state.knn_smote_crossval
                )
            placeholder2.empty()
        if st.session_state.knn_smote_trained:
            st.success("Selesai")
    with col3:
        placeholder3 = st.empty()
        knn_smote_tmgwo_crossval = st.checkbox("Cross-Validation", 
                                    value=st.session_state.knn_smote_tmgwo_crossval, 
                                    disabled=True if st.session_state.knn_smote_tmgwo_trained else False,
                                    help="Jika dicentang, maka akan dilakukan cross-validation juga dan pelatihan akan memakan waktu lebih lama")
        st.session_state.knn_smote_tmgwo_crossval = knn_smote_tmgwo_crossval
        if placeholder3.button(
            "Latih Model KNN dengan SMOTE dan TMGWO",
            key="KNN_SMOTE_TMGWO",
            disabled=True if st.session_state.knn_smote_tmgwo_trained else False,
        ):
            st.button(
                "Latih Model KNN dengan SMOTE dan TMGWO",
                key="KNN_SMOTE_TMGWO_Disabled",
                disabled=True,
            )
            placeholder3.empty()
            st.session_state.knn_smote_tmgwo_trained = True
            with st.spinner("melatih model dengan KNN dan SMOTE dan TMGWO..."):
                time.sleep(1)
                model_knn_smote_tmgwo, score_knn_smote_tmgwo, cm_knn_smote_tmgwo, cr_knn_smote_tmgwo = train_model(
                    X, y, 
                    X_train_smote[:, selected_features],
                    X_test[:, selected_features],
                    y_train_smote,
                    y_test,
                    "KNN+SMOTE+TMGWO",
                    cv=st.session_state.knn_smote_tmgwo_crossval
                )
            placeholder3.empty()
        if st.session_state.knn_smote_tmgwo_trained:
            st.success("Selesai")
        if st.session_state.knn_trained or st.session_state.knn_smote_trained or st.session_state.knn_smote_tmgwo_trained:
            st.session_state.trained = True
else:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 80px'>Latih Model</h1>",
        unsafe_allow_html=True,
    )
    st.warning("Silakan lakukan pengolahan data (feature selection, oversampling) terlebih dahulu.", icon="⚠️")
