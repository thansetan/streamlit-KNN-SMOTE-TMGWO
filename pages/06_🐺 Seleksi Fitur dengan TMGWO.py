import numpy as np
import pandas as pd

np.random.seed(42)
import streamlit as st
from scripts.skripsi import feature_selection, plot_selected_features

st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Seleksi Fitur",
    layout="centered",
    page_icon="random",
)

if "feature_selection" not in st.session_state:
    st.session_state.feature_selection = False
if "selected_features" not in st.session_state:
    st.session_state.selected_features = None
if "splitted" not in st.session_state:
    st.session_state.splitted = False
if st.session_state.splitted:
    X = st.session_state.X
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    data = st.session_state.preprocessed_data
    col1, col2, col3 = st.columns(3)
    with col2:
        if not st.session_state.feature_selection:
            st.session_state.feature_selection = True
            with st.spinner("Performing feature selection..."):
                st.session_state.selected_features = feature_selection(X_train, X_test, y_train, y_test)
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 10px'>Seleksi Fitur dengan TMGWO</h1>",
        unsafe_allow_html=True,
    )
    st.pyplot(plot_selected_features(X, st.session_state.selected_features))

    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 10px'>Daftar Fitur</h1>",
        unsafe_allow_html=True,
    )
    feature_list = [list(data.columns[:-1])][0]
    selected_features_list = [col for col, selected in zip(feature_list, st.session_state.selected_features) if selected]
    selected_features_dict = {"Features": feature_list,
            "Keterangan": [
                "Terpilih" if feature in selected_features_list else "Tidak Terpilih"
                for feature in feature_list
            ],
        }
    selected_features_dict = pd.DataFrame(selected_features_dict)
    selected_features_dict.index+=1
    st.table(selected_features_dict)
else:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 80px'>Seleksi Fitur dengan TMGWO</h1>",
        unsafe_allow_html=True,
    )
    st.warning("Tolong bukanya dari atas-bawah, jangan dilewatin. terima kasih.", icon="⚠️")
