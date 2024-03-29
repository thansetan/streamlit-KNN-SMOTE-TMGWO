import numpy as np
import streamlit as st

np.random.seed(42)
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Flowchart Penelitian",
    layout="centered",
    page_icon="random",
)
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.markdown(
        "<h1 style='text-align: center'>Flowchart Penelitian</h1>", unsafe_allow_html=True
    )
    st.image("images/flowchart.png", use_column_width=True)
