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
        "Early Risk Prediction of Diabetes Based on GA-Stacking",
        "Proposed Method"
    ],
    "Penulis": [
        "Yaqi Tan, et al.",
        "Fathan Arsyadani"
    ],
    "Tahun": [
        "2022",
        "2022"
    ],
    "Algoritma": [
        "Genetic Algorithm, Stacking",
        "KNN + SMOTE + TMGWO"
    ],
    "Dataset": [
        "Early stage diabetes risk prediction dataset",
        "Early stage diabetes risk prediction dataset"
    ],
    "Akurasi": [
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
