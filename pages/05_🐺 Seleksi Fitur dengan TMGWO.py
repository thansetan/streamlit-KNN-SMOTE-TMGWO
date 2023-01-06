import numpy as np
import pandas as pd

np.random.seed(8047)
import streamlit as st

from scripts.skripsi import (feature_selection, plot_selected_features,
                             split_X_y)

st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Seleksi Fitur",
    layout="centered",
    page_icon="random",
)
if "X" not in st.session_state:
    st.session_state.X = None
if "y" not in st.session_state:
    st.session_state.y = None
if "preprocessed" not in st.session_state:
    st.session_state.preprocessed = False
if "feature_selection" not in st.session_state:
    st.session_state.feature_selection = False
if "selected_features" not in st.session_state:
    st.session_state.selected_features = None
if "splitted" not in st.session_state:
    st.session_state.splitted = False
if "X_sf" not in st.session_state:
    st.session_state.X_sf = None
if st.session_state.preprocessed:
    X, y = split_X_y(st.session_state.preprocessed_data)
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.splitted = True
    data = st.session_state.preprocessed_data
    col1, col2, col3 = st.columns(3)
    with col2:
        if not st.session_state.feature_selection:
            st.session_state.feature_selection = True
            with st.spinner("Performing feature selection..."):
                st.session_state.selected_features = feature_selection(X, y)
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
    selected_features_list = [
        col
        for col, selected in zip(feature_list, st.session_state.selected_features)
        if selected
    ]
    selected_features_dict = {
        "Features": feature_list,
        "Keterangan": [
            "Terpilih" if feature in selected_features_list else "Tidak Terpilih"
            for feature in feature_list
        ],
    }
    st.session_state.X_sf = X[:, st.session_state.selected_features]
    selected_features_dict = pd.DataFrame(selected_features_dict)
    selected_features_dict.index += 1
    st.dataframe(selected_features_dict, use_container_width=True)
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 10px'>Dataset with Only Selected Features</h1>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        data[
            data.columns[np.where(st.session_state.selected_features)].append(
                pd.Index([data.columns[-1]])
            )
        ],
        use_container_width=True,
    )
else:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 80px'>Seleksi Fitur dengan TMGWO</h1>",
        unsafe_allow_html=True,
    )
    st.warning("Silakan lakukan preprocessing terlebih dahulu.", icon="⚠️")
