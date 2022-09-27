
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
name_dict = {
    'Age': 'Berapa usia anda? (tahun)',
    'Gender': 'Jenis kelamin anda?',
    'Polyuria': 'Apakah belakangan ini anda lebih sering buang air kecil dibanding hari biasanya?',
    'Polydipsia': 'Apakah sering tetap merasa haus meskipun sudah minum dalam jumlah yang cukup?',
    'sudden weight loss': 'Apakah anda mengalami penurunan berat badan secara tiba-tiba?',
    'weakness': 'Apakah anda sering merasa lemas?',
    'Polyphagia': 'Apakah anda sering tetap merasa lapar meskipun sudah makan dalam porsi yang cukup?',
    'Genital thrush': 'Apakah anda mengalami infeksi jamur di alat kelamin?',
    'visual blurring': 'Apakah anda sering mengalami pandangan mata kabur secara tiba-tiba?',
    'Itching': 'Apakah anda sering merasa gatal/geli tanpa sebab?',
    'Irritability': 'Apakah anda sering merasa mudah tersinggung/marah?',
    'delayed healing': 'Apakah ketika anda memiliki luka fisik, luka tersebut tak kunjung sembuh/kering?',
    'partial paresis': 'Apakah anda sering mengalami rasa lemah/lemas pada otot/sebagian anggota badan?',
    'muscle stiffness': 'Apakah anda sering mengalami rasa kaku pada otot?',
    'Alopecia': 'Apakah anda mengalami kerontokan rambut?',
    'Obesity': 'Apakah anda mengalami obesitas/berat badan berlebih? (BMI>30)',
}
if "knn_smote_tmgwo_trained" not in st.session_state or not hasil['akurasi']:
    st.session_state.knn_smote_tmgwo_trained = False
if hasil['akurasi'] and st.session_state.knn_smote_tmgwo_trained:
    data = pd.read_csv(f"datasets/{st.session_state.dataset}")
    if data.columns[0] != "Age":
        st.error("Ndak tahu krn bkn data diabetes", icon="🗿")
    else:
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
                "<h1 style='text-align: center; margin-bottom:20px;'>Deteksi Dini Resiko Diabetes</h1>",
                unsafe_allow_html=True,
        )
        feature_labels = features.copy()
        no = 1
        for i in range(len(features)):
            if features[i] in int_features:
                col1, col2, col3 = st.columns([1,5,1])
                with col2:
                    features[i] = st.number_input(f"{no}. {name_dict[feature_labels[i]]}", min_value=0, max_value=100, value=30, step=1, key=feature_labels[i])
            elif features[i] in object_features:
                col1, col2, col3 = st.columns([1,5,1])
                with col2:
                    features[i] = st.radio(f"{no}. {name_dict[feature_labels[i]]}", options=reversed(sorted(data[features[i]].unique())), key=features[i], index=1, horizontal=True, help="Pilih salah satu")
            no+=1
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
    st.warning("Silakan latih model KNN-SMOTE-TMGWO terlebih dahulu.", icon="⚠️")
