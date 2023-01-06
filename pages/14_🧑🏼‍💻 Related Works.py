import matplotlib.pyplot as plt
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
        "A Novel Wrapper-Based Feature Selection for Early Diabetes Prediction Enhanced with a Metaheuristic",
        "Machine Learning Algorithms for Diabetes Detection: A Comparative Evaluation of Performance of Algorithms",
        "Proposed Method",
    ],
    "Penulis": [
        "Chaves & Marques",
        "Le, et al.",
        "Saxena, et al.",
        "Fathan Arsyadani",
    ],
    "Tahun": ["2021", "2021", "2021", "2022"],
    "Algoritme": [
        "ANN, Information Gain",
        "MLP + APGWO",
        "Stacked Ensemble + MDI",
        "KNN + TMGWO + SMOTE",
    ],
    "Dataset": [
        "Early stage diabetes risk prediction dataset",
        "Early stage diabetes risk prediction dataset",
        "Early stage diabetes risk prediction dataset",
        "Early stage diabetes risk prediction dataset",
    ],
    "Akurasi": ["98.08 %", "97.12 %", "97 %", "98.85 %"],
}
st.markdown(
    "<h1 style='text-align: center; margin-bottom: 10px'>Daftar Penelitian Terkait</h1>",
    unsafe_allow_html=True,
)
related_works = pd.DataFrame(related_works_dict)
related_works.index += 1
st.dataframe(related_works)
fig, ax = plt.subplots()
peneliti = ["Chaves & Marques", "Le, et al.", "Saxena, et al.", "Fathan Arsyadani"]
akurasi = [98.08, 97.12, 97, 98.85]
for i in range(len(peneliti)):
    ax.bar(peneliti[i], akurasi[i], label=peneliti[i])
    ax.annotate(f"{akurasi[i]} %", (peneliti[i], akurasi[i] / 2), ha="center")
ax.set_title("Perbandingan Akurasi")
ax.set_xlabel("Peneliti")
ax.set_ylabel("Akurasi (%)")
st.pyplot(fig, use_column_width=True)
