import numpy as np
import pandas as pd
import streamlit as st
from scripts.skripsi import trained
from sklearn.metrics import classification_report

np.random.seed(42)
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Classification Metrisc",
    layout="centered",
    page_icon="random",
)
cr_dict = {
    "Algoritma": [model for model in trained["ypred"].keys()],
    "Classification_Report": [
        trained["ypred"][model] for model in trained["ypred"].keys()
    ],
}
if "trained" not in st.session_state:
    st.session_state.trained = False
if st.session_state.trained and trained["ypred"]:
    y = st.session_state.y
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 10px'>Classification Report Tiap Algoritma</h1>",
        unsafe_allow_html=True,
    )
    for i in range(len(cr_dict["Algoritma"])):
        st.markdown(
            f"<h2 style='text-align: center; margin-bottom: 10px'>{cr_dict['Algoritma'][i]}</h2>",
            unsafe_allow_html=True,
        )
        cr = classification_report(
            y,
            cr_dict["Classification_Report"][i],
            output_dict=True,
            target_names=["Negative", "Positive"],
        )
        df = pd.DataFrame(cr).transpose()
        df[df.columns[:-1]] = df[df.columns[:-1]].applymap(lambda x: x * 100)
        df[df.columns[:-1]] = df[df.columns[:-1]].applymap(lambda x: f"{x:.2f}")
        df[df.columns[-1]] = df[df.columns[-1]].apply(lambda x: int(x))
        df.drop("accuracy", axis=0, inplace=True)
        st.table(df)
else:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 80px'>Classification Report Tiap Algoritma</h1>",
        unsafe_allow_html=True,
    )
    st.warning("Silakan latih model terlebih dahulu.", icon="⚠️")
