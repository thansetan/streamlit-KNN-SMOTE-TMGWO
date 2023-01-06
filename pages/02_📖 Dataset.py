import os

import numpy as np
import pandas as pd
import streamlit as st

np.random.seed(42)
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Diabetes Dataset",
    layout="centered",
    page_icon="random",
)


def save_uploadedfile(uploadedfile):
    with open(os.path.join("datasets", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())


uploaded_file = None
placeholder = st.empty()
if "column_names" not in st.session_state:
    st.session_state.column_names = None
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "isDiabetes" not in st.session_state:
    st.session_state.isDiabetes = True
if st.session_state.dataset == None:
    uploaded_file = placeholder.file_uploader("Upload dataset", type="csv")
if uploaded_file or st.session_state.dataset:
    placeholder.empty()
    try:
        st.session_state.dataset = uploaded_file.name
        data = pd.read_csv(uploaded_file)
        save_uploadedfile(uploaded_file)
        st.session_state.column_names = data.columns
    except:
        data = pd.read_csv(f"datasets/{st.session_state.dataset}")
    if (
        data.columns[0] != "Age"
        and data.columns[2] != "Polyuria"
    ):
        st.markdown(
            "<h1 style='text-align: center; margin-bottom: 10px'>Dataset</h1>",
            unsafe_allow_html=True,
        )
        st.session_state.isDiabetes = False
        data.index += 1
        st.dataframe(data)
    else:
        st.markdown(
            "<h1 style='text-align: center; margin-bottom: 10px'>Diabetes Dataset</h1>",
            unsafe_allow_html=True,
        )
        st.write(
            'Dataset yang digunakan pada penelitian ini merupakan dataset "[Early Stage Diabetes Risk Prediction Dataset](https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.)" yang diperoleh dari UCI Machine Learning Repository'
        )
        data.index += 1
        st.dataframe(data)
else:
    st.warning("Silakan upload dataset terlebih dahulu.", icon="⚠️")
