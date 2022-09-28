import numpy as np
import pandas as pd
import streamlit as st

np.random.seed(42)
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Related Works",
    layout="centered",
    page_icon="random",
)
related_works_dict = {
    "Judul": [
        "Data Mining Techniques for Early Diagnosis of Diabetes: A Comparative Study",
        "Data-Driven Machine-Learning Methods for Diabetes Risk Prediction",
        "Early Risk Prediction of Diabetes Based on GA-Stacking",
        "Proposed Method"
    ],
    "Penulis": [
        "Chaves & Marques",
        "Dristas & Trigka",
        "Yaqi Tan, et al.",
        "Fathan Arsyadani"
    ],
    "Tahun": [
        "2021",
        "2022",
        "2022",
        "2022"
    ],
    "Algoritma": [
        "Artificial Neural Neural Network, Information Gain",
        "K-Nearest Neighbors + SMOTE",
        "Genetic Algorithm, Stacking",
        "KNN + SMOTE + TMGWO"
    ],
    "Dataset": [
        "Early stage diabetes risk prediction dataset",
        "Early stage diabetes risk prediction dataset",
        "Early stage diabetes risk prediction dataset",
        "Early stage diabetes risk prediction dataset"
    ],
    "Akurasi": [
        "98.08 %",
        "99.22 %",
        "98.71 %",
        "99.04 %"
    ]
}
st.markdown(
    "<h1 style='text-align: center; margin-bottom: 10px'>Daftar Penelitian Terkait</h1>",
    unsafe_allow_html=True,
)
related_works = pd.DataFrame(related_works_dict)
related_works.index+=1
st.dataframe(related_works)
