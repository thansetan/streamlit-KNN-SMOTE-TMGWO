import warnings

warnings.filterwarnings("ignore")
import numpy as np

np.random.seed(42)
import functools

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from scripts.TMGWO import TMGWO


def load_dataset(dataset):
    data = pd.read_csv(dataset)
    return data


def preprocess(data):
    attributes = data.drop(data.columns[-1], axis=1)
    label = data.columns[-1]
    labelType = data.dtypes[label]
    objectList = attributes.select_dtypes(include="object").columns
    numList = attributes.select_dtypes(include=["int", "float"]).columns
    le = LabelEncoder()
    scaler = MinMaxScaler()
    if labelType == "O":
        data[label] = le.fit_transform(data[label])
    for feature in objectList:
        data[feature] = le.fit_transform(data[feature])
    for feature in numList:
        data[[feature]] = scaler.fit_transform(data[[feature]])
    joblib.dump(scaler, "models/scaler.pkl")
    return data


def save_preprocessed(preprocessed):
    preprocessed.to_csv("datasets/diabetes_preprocessed.csv", index=False)


def split_preprocessed(preprocessed):
    X = preprocessed[preprocessed.columns[:-1]].values
    y = preprocessed[preprocessed.columns[-1]].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2, shuffle=True, stratify=y
    )
    return X, y, X_train, X_test, y_train, y_test


def fitness(x, X_train, y_train, X_test, y_test):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    loss = np.zeros(x.shape[0])
    model = KNeighborsClassifier(metric="jaccard", n_neighbors=2)
    for i in range(x.shape[0]):
        if np.sum(x[i, :]) > 0:
            model.fit(X_train[:, x[i, :].astype(bool)], y_train)
            score = accuracy_score(
                model.predict(X_test[:, x[i, :].astype(bool)]), y_test
            )
            loss[i] = 0.99 * (1 - score) + 0.01 * (np.sum(x[i, :]) / X_train.shape[1])
        else:
            loss[i] = np.inf
    return loss


def data_antar_kelas(y_train):
    positive = list(y_train).count(1)
    negative = list(y_train).count(0)
    fig = plt.figure()
    plt.title("Data antar kelas pada training data")
    plt.bar("Positive", positive)
    plt.annotate(positive, (0, positive / 2), ha="center")
    plt.bar("Negative", negative)
    plt.annotate(negative, (1, negative / 2), ha="center")
    plt.xlabel("Kelas")
    plt.ylabel("Jumlah data")
    plt.show()
    return fig


def feature_selection(X_train, X_test, y_train, y_test):
    lossfunc = functools.partial(
        fitness, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    optimizer = TMGWO(fitness=lossfunc, D=X_train.shape[1], P=8, G=70)
    optimizer.optimize()
    selected_features = optimizer.gBest_X > 0
    return selected_features


def plot_selected_features(X, selected_features):
    fig = plt.figure()
    plt.title("Jumlah Fitur Sebelum dan Sesudah Seleksi Fitur")
    plt.bar("before", X.shape[1])
    plt.annotate(X.shape[1], (0, X.shape[1] / 2), ha="center")
    plt.bar("after", X[:, selected_features].shape[1])
    plt.annotate(
        X[:, selected_features].shape[1],
        (1, X[:, selected_features].shape[1] / 2),
        ha="center",
    )
    plt.ylabel("Num of features")
    return fig


def do_smote(X_train, y_train):
    try:
        smote = SMOTENC(
            random_state=42,
            categorical_features=[
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
        )
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    except:
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote


def plot_smote(y_train_smote):
    positive = list(y_train_smote).count(1)
    negative = list(y_train_smote).count(0)
    fig = plt.figure()
    plt.title("Data antar kelas pada training data setelah SMOTE")
    plt.bar("Positive", height=positive)
    plt.annotate(positive, (0, positive / 2), ha="center", fontsize=20)
    plt.bar("Negative", height=negative)
    plt.annotate(negative, (1, negative / 2), ha="center", fontsize=20)
    plt.xlabel("Kelas")
    plt.ylabel("Jumlah data")
    return fig


hasil = {}
hasil["akurasi"] = {}
hasil["cm"] = {}
hasil["cr"] = {}


def train_model(X_train, X_test, y_train, y_test, algoritma):
    model = KNeighborsClassifier(metric="jaccard", n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    hasil["akurasi"][algoritma] = float(
        "{:.2f}".format(accuracy_score(y_test, y_pred) * 100)
    )
    hasil["cm"][algoritma] = confusion_matrix(y_test, y_pred)
    hasil["cr"][algoritma] = classification_report(
        y_test, y_pred, output_dict=True, target_names=["Negative", "Positive"]
    )
    score = hasil["akurasi"][algoritma]
    cm = hasil["cm"][algoritma]
    cr = hasil["cr"][algoritma]
    joblib.dump(model, f"models/{algoritma}.pkl")
    return model, score, cm, cr


def get_highest_acc_index(hasil):
    index = {}
    for key, val in hasil["akurasi"].items():
        index[key] = val.index(max(val))
    return index


def plot_cm(cm, algo):
    fig, ax = plt.subplots()
    cm = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
    cm.plot(ax=ax)
    plt.title(f"Confusion Matrix {algo}")
    return fig


def plot_highest_accuracy(hasil):
    fig = plt.figure()
    acc = [acc for acc in hasil["akurasi"].values()]
    label = [algoritma for algoritma in hasil["akurasi"].keys()]
    juml = len(acc)
    plt.title("Hasil Model")
    for i in range(juml):
        plt.bar(i, acc[i], label="Ori")
        plt.annotate(f"{acc[i]} %", (i, acc[i] / 2), ha="center")
    plt.ylabel("Akurasi")
    plt.xticks([i for i in range(juml)], label)
    plt.xlabel("Algoritma")
    plt.ylim(0, 100)
    return fig
