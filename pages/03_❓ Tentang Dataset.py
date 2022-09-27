import numpy as np
import pandas as pd
import streamlit as st

np.random.seed(42)
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Informasi Dataset",
    layout="centered",
    page_icon="random",
)

if "dataset" not in st.session_state:
    st.session_state.dataset = None
try:
    data = pd.read_csv("datasets/diabetes.csv")
    keterangan = {
        "Atribut": [col for col in data.columns],
        "Keterangan": [
            "Usia",
            "Jenis Kelamin",
            "Produksi urine berlebih (sering buang air kecil)",
            "Selalu merasa haus, mulut terasa kering",
            "Penurunan berat badan secara tiba-tiba",
            "Sering merasa lemas",
            "Rasa lapar berlebihan",
            "Infeksi jamur di alat kelamin",
            "Pandangan mata kabur",
            "Sering merasa gatal/geli",
            "mudah tersinggung/marah",
            "Luka tak kunjung kering/sembuh",
            "Mengalami rasa lemah/lemas pada otot/sebagian anggota badan",
            "Sering mengalami kaku otot",
            "Mengalami kerontokan rambut",
            "Obesitas/Berat badan berlebih",
            "Hasil diagnosis",
        ],
        "Nilai": [
            ", ".join(reversed(sorted(data[col].unique())))
            if col != "Age"
            else "Usia (dalam tahun)"
            for col in data.columns
        ],
    }
except:
    pass
if st.session_state.dataset:
    datacek = pd.read_csv(f"datasets/{st.session_state.dataset}")
    if datacek.columns[0] != "Age":
        st.markdown(
            "<h1 style='text-align: center; margin-bottom: 10px'>Informasi Dataset</h1>",
            unsafe_allow_html=True,
        )
        st.write("Ndak tahu krn bkn data diabetes")
    else:
        st.markdown(
            "<h1 style='text-align: center; margin-bottom: 10px'>Daftar Fitur</h1>",
            unsafe_allow_html=True,
        )
        st.write(
            f"Dataset yang digunakan terdiri dari {len(data)} baris data dan {len(data.columns)} kolom (16 fitur dan 1 target). Dengan keterangan tiap kolom sebagai berikut: "
        )
        keterangan = pd.DataFrame.from_dict(keterangan)
        keterangan.index += 1
        st.table(keterangan)
else:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 80px'>Informasi Dataset</h1>",unsafe_allow_html=True
    )
    st.warning("Silakan pilih dataset terlebih dahulu.", icon="⚠️")
