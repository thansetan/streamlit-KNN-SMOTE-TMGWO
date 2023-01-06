import warnings

warnings.filterwarnings("ignore")
import numpy as np

np.random.seed(42)
import functools

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import make_pipeline
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import KFold, cross_val_predict, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from scripts.TMGWO import TMGWO

kf = KFold(n_splits=10, random_state=0, shuffle=True)


def load_dataset(dataset):
    data = pd.read_csv(dataset)
    return data


def preprocess(data):
    attributes = data.drop(data.columns[-1], axis=1)
    label = data.columns[-1]
    labelType = data.dtypes[label]
    objectList = attributes.select_dtypes(include="object").columns
    numList = attributes.select_dtypes(include=["int", "float"]).columns
    le_f_gender, le_f, le_l = LabelEncoder(), LabelEncoder(), LabelEncoder()
    scaler = MinMaxScaler()
    if labelType == "O":
        data[label] = le_l.fit_transform(data[label])
    for feature in objectList:
        if feature == "Gender":
            data[feature] = le_f_gender.fit_transform(data[feature])
        else:
            data[feature] = le_f.fit_transform(data[feature])
    for feature in numList:
        data[[feature]] = scaler.fit_transform(data[[feature]])
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(le_f, "models/le_f.pkl")
    joblib.dump(le_f_gender, "models/le_f_gender.pkl")
    joblib.dump(le_l, "models/le_l.pkl")
    return data


def split_X_y(data):
    X = data[data.columns[:-1]].values
    y = data[data.columns[-1]].values
    return X, y


def fitness(x, X, y):
    alpha = 0.99
    beta = 1 - alpha
    if x.ndim == 1:
        x = x.reshape(1, -1)
    loss = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        if np.sum(x[i, :]) > 0:
            model = KNeighborsClassifier(
                metric="manhattan", n_neighbors=3, weights="distance"
            )
            y_pred = cross_val_predict(model, X[:, x[i, :].astype(bool)], y, cv=kf)
            acc = accuracy_score(y, y_pred)
            error_rate = 1 - acc
            loss[i] = alpha * error_rate + beta * (
                np.sum(x[i, :]) / X.shape[1]
            )  # eq 3.2
        else:
            loss[i] = np.inf
    return loss


def feature_selection(X, y):
    lossfunc = functools.partial(fitness, X=X, y=y)
    optimizer = TMGWO(fitness=lossfunc, D=X.shape[1], P=8, G=40, Mp=0.07)
    optimizer.optimize()
    selected_features = optimizer.gBest_X.astype(bool)
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


def get_data_antar_kelas(X, y):
    posneg = {}
    hasil_smote = {}
    for i, (train, test) in enumerate(kf.split(X, y), start=1):
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]
        posneg[i] = {
            "before_smote": {
                "positive": list(y_train).count(1),
                "negative": list(y_train).count(0),
            }
        }
        smotenc = SMOTENC(
            categorical_features=[i for i in range(1, X.shape[1])],
            random_state=505,
            k_neighbors=5,
        )
        X_train_oversampled, y_train_oversampled = smotenc.fit_resample(
            X_train, y_train
        )
        posneg[i]["after_smote"] = {
            "positive": list(y_train_oversampled).count(1),
            "negative": list(y_train_oversampled).count(0),
        }
        hasil_smote[i] = np.concatenate(
            (
                X_train_oversampled[X_train.shape[0] :],
                y_train_oversampled[y_train.shape[0] :].reshape(-1, 1),
            ),
            axis=1,
        )
    return posneg, hasil_smote


def plot_data_antar_kelas(y, i):
    positive = y["positive"]
    negative = y["negative"]
    fig = plt.figure()
    plt.title(f"Data antar kelas pada training data pada iterasi ke-{i}")
    plt.bar("Positive", positive)
    plt.annotate(positive, (0, positive / 2), ha="center")
    plt.bar("Negative", negative)
    plt.annotate(negative, (1, negative / 2), ha="center")
    plt.xlabel("Kelas")
    plt.ylabel("Jumlah data")
    plt.show()
    return fig


def synthetic_data(data, column_names):
    synthetic_only = pd.DataFrame(data, columns=column_names)
    synthetic_only[synthetic_only.columns[1:]] = synthetic_only[
        synthetic_only.columns[1:]
    ].applymap(np.int64)
    synthetic_only.index += 1
    return synthetic_only


trained = {}
trained["result"] = {}
trained["ypred"] = {}
trained["akurasi"] = {}


def train_model(X, y, algo):
    knn = KNeighborsClassifier(n_neighbors=3, metric="manhattan", weights="distance")
    knn_smotenc = make_pipeline(
        SMOTENC(
            categorical_features=[i for i in range(1, X.shape[1])],
            random_state=72,
            k_neighbors=3,
        ),
        knn,
    )
    if algo == "KNN" or algo=="KNN+TMGWO":
        model = knn
    else:
        model = knn_smotenc
    trained["result"][algo] = cross_validate(model, X, y, cv=kf, scoring="accuracy")
    trained["ypred"][algo] = cross_val_predict(model, X, y, cv=kf)
    trained["akurasi"][algo] = accuracy_score(y, trained["ypred"][algo])
    model.fit(X, y)
    joblib.dump(model, f"models/{algo}.pkl")
    return model


def plot_cm(cm, algo):
    fig, ax = plt.subplots()
    cm = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
    cm.plot(ax=ax)
    plt.title(f"Confusion Matrix {algo}")
    return fig


def plot_acc(hasil):
    fig = plt.figure()
    acc = [round(acc * 100, 2) for acc in hasil.values()]
    label = [algoritma for algoritma in hasil.keys()]
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
