import numpy as np
import pandas as pd
import streamlit as st
from scripts.skripsi import plot_cm, trained
from sklearn.metrics import confusion_matrix

np.random.seed(42)
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Confusion Matrix",
    layout="centered",
    page_icon="random",
)

ypred_dict = trained['ypred']
if "trained" not in st.session_state:
    st.session_state.trained = False
if "y" not in st.session_state:
    st.session_state.y = None
if st.session_state.trained and ypred_dict:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 10px'>Confusion Matrix</h1>",
        unsafe_allow_html=True,
)
    y = st.session_state.y
    for algo, ypred in ypred_dict.items():
        cm = confusion_matrix(y, ypred)
        col1, col2 = st.columns([5,2])
        with col1:
            st.pyplot(plot_cm(cm, algo), use_column_width=True)
        with col2:
            TP = cm[1][1]
            TN = cm[0][0]
            FP = cm[1][0]
            FN = cm[0][1]
            st.markdown("<div style='margin-top:55px'></div>", unsafe_allow_html=True)
            st.latex(r"Akurasi = \frac{TP+TN}{TP+TN+FP+FN}\times100")
            st.latex(r"Akurasi = \frac{"+str(TP)+"+"+str(TN)+"}{"+str(TP)+"+"+str(TN)+"+"+str(FP)+"+"+str(FN)+r"}\times100")
            st.latex(r"Akurasi = \frac{"+str(TP+TN)+"}{"+str(TP+TN+FP+FN)+r"}\times100")
            st.latex(r"Akurasi = "+"{:.4f}".format((TP+TN)/(TP+TN+FP+FN))+r"\times100")
            st.latex(r"Akurasi = "+"{:.2f}".format((TP+TN)/(TP+TN+FP+FN)*100)+"\;\%")
        
else:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 80px'>Confusion Matrix</h1>",
        unsafe_allow_html=True,
    )
    st.warning("Silakan latih model terlebih dahulu.", icon="⚠️")
