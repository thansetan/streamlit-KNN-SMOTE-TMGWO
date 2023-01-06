import time

import numpy as np

np.random.seed(42)
import numpy as np
import streamlit as st
from scripts.skripsi import plot_data_antar_kelas, synthetic_data

st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="SMOTE",
    layout="centered",
    page_icon="random",
)
if "smote" not in st.session_state:
    st.session_state.smote = False
if "X" not in st.session_state:
    st.session_state.X = None
if "y" not in st.session_state:
    st.session_state.y = None
if "X_sf" not in st.session_state:
    st.session_state.X_sf = None
if "feature_selection" not in st.session_state:
    st.session_state.feature_selection = False
if "oversampled" not in st.session_state:
    st.session_state.oversampled = False
if "posneg" not in st.session_state:
    st.session_state.posneg = None
if "hasil_smote" not in st.session_state:
    st.session_state.hasil_smote = None
if st.session_state.posneg:
    cols = st.session_state.column_names
    posneg = st.session_state.posneg
    hasil_smote = st.session_state.hasil_smote
    sf = st.session_state.selected_features
    col1, col2, col3 = st.columns(3)
    with col2:
        if not st.session_state.smote:
            st.session_state.smote = True
            with st.spinner("Performing SMOTE on training data..."):
                time.sleep(1)
                st.session_state.oversampled = True
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 10px'>Oversampling dengan SMOTE</h1>",
        unsafe_allow_html=True,
    )
    for key, val in posneg.items():
        st.pyplot(plot_data_antar_kelas(val["after_smote"], key), use_column_width=True)
        st.markdown(
            "<h1 style='text-align: center; margin-bottom: 10px'>Synthetic Data yang Dibuat</h1>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            synthetic_data(
                hasil_smote[key],
                cols[np.where(sf)].tolist()+[cols[-1]]
            )
        )
    # sf = list(st.session_state.selected_features)
    # sf.append(True)
    # st.dataframe(
    #     synthetic_data(
    #         X_train, st.session_state.X_train_smote_fs, st.session_state.y_train_smote_fs, st.session_state.column_names[np.where(sf)])
    #     )
    st.session_state.oversampled = True
else:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 80px'>Oversampling dengan SMOTE</h1>",
        unsafe_allow_html=True,
    )
    st.warning("Liat yg blm di-oversample dulu ya anjing.", icon="⚠️")
