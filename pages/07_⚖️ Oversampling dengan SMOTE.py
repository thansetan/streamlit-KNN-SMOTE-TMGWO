import time

import numpy as np

np.random.seed(42)
import numpy as np
import streamlit as st
from scripts.skripsi import do_smote, plot_smote

st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="SMOTE",
    layout="centered",
    page_icon="random",
)
if "smote" not in st.session_state:
    st.session_state.smote = False
if "X_train_smote" not in st.session_state:
    st.session_state.X_train_smote = None
if "y_train_smote" not in st.session_state:
    st.session_state.y_train_smote = None
if "feature_selection" not in st.session_state:
    st.session_state.feature_selection = False
if "oversampled" not in st.session_state:
    st.session_state.oversampled = False
if st.session_state.feature_selection:
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    col1, col2, col3 = st.columns(3)
    with col2:
        if not st.session_state.smote:
            st.session_state.smote = True
            with st.spinner("Performing SMOTE on training data..."):
                time.sleep(1)
                (
                    st.session_state.X_train_smote,
                    st.session_state.y_train_smote,
                ) = do_smote(X_train, y_train)
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 10px'>Oversampling dengan SMOTE</h1>",
        unsafe_allow_html=True,
    )
    st.pyplot(plot_smote(st.session_state.y_train_smote), use_column_width=True)
    st.session_state.oversampled = True
else:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 80px'>Oversampling dengan SMOTE</h1>",
        unsafe_allow_html=True,
    )
    st.warning("Silakan lakukan seleksi fitur terlebih dahulu.", icon="⚠️")
