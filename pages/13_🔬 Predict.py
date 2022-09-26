
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scripts.skripsi import hasil

np.random.seed(42)
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="SMOTE",
    layout="centered",
    page_icon="random",
)
if hasil['akurasi']:
    data = pd.read_csv(f"datasets/{st.session_state.dataset}")
    int_features = data.select_dtypes(include="int64").columns.to_list()
    object_features = data.select_dtypes(include="object").columns.to_list()
    selected_features = st.session_state.selected_features
    feature_names = list(data[data.columns[np.where(selected_features)]])
    features = list(data[data.columns[np.where(selected_features)]])
    st.markdown("""
    <style>
    .css-ocqkz7{
        margin-left: 20px;
    </style>
    """, unsafe_allow_html=True)
    st.markdown(
            "<h1 style='text-align: center;'>Deteksi Dini Resiko Diabetes</h1>",
            unsafe_allow_html=True,
    )
    feature_labels = features.copy()
    for i in range(len(features)):
        if features[i] in int_features:
            features[i] = st.number_input(f"{feature_labels[i]}", min_value=0, max_value=100, value=30, step=1, key=feature_labels[i])
        elif features[i] in object_features:
            features[i] = st.radio(f"{features[i]}", options=reversed(sorted(data[features[i]].unique())), key=features[i], index=1, horizontal=True)
        # else:
        #     features[i] = st.radio(f"{features[i]}", options=data[features[i]].unique(), key=features[i], index=1, horizontal=True)

    prediction = None
    col4, col5, col6, col7, col8 = st.columns(5)
    with col6:
        if st.button("Prediksi", key="prediksi"):
            model = joblib.load("models/KNN+SMOTE+TMGWO.pkl")
            scaler = joblib.load("models/scaler.pkl")
            X = pd.DataFrame([features], columns=feature_names)
            X = X.replace({"Yes": 1, "No": 0,"Female": 0, "Male": 1})
            try:
                X["Age"] = scaler.transform([X["Age"]])
            except KeyError:
                pass
            prediction = model.predict(X)
    col9,col10,col11=st.columns([1,3,1])
    with col10:
        if prediction is not None:
            col10.text(f"Prediksi: {'Beresiko Terkena Diabetes' if prediction[0]==1 else 'Tidak Beresiko Terkena Diabetes'}")
        else:
            pass
else:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 80px'>Prediksi</h1>",
        unsafe_allow_html=True,
    )
    st.warning("Silakan latih model terlebih dahulu.", icon="⚠️")
