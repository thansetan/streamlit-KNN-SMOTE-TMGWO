import numpy as np
import pandas as pd
import streamlit as st
from scripts.skripsi import hasil

np.random.seed(42)
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Classification Metrisc",
    layout="centered",
    page_icon="random",
)
cr_dict = {
    "Algoritma": [model for model in hasil["cr"].keys()],
    "Classification_Report": [hasil["cr"][model] for model in hasil["cr"].keys()],
}
if "trained" not in st.session_state:
    st.session_state.trained = False
if st.session_state.trained and hasil["cr"]:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 10px'>Classification Report Tiap Algoritma</h1>",
        unsafe_allow_html=True,
)
    for i in range(len(cr_dict["Algoritma"])):
        st.markdown(
        f"<h2 style='text-align: center; margin-bottom: 10px'>{cr_dict['Algoritma'][i]}</h2>",
        unsafe_allow_html=True,
)
        df = pd.DataFrame(cr_dict["Classification_Report"][i]).transpose()
        st.table(df)
else:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 80px'>Classification Report Tiap Algoritma</h1>",
        unsafe_allow_html=True,
    )
    st.warning("Silakan latih model terlebih dahulu.", icon="⚠️")
