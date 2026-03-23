import sys
from pathlib import Path
import re
import json
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import fasttext

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


label_qual = "qualitative"
label_quant = "quantitative"


# Function to convert a pandas DataFrame into the format required by fastText.
def write_fasttext(df, text_col, label_col, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            txt = str(row[text_col]).replace("\n", " ").strip()
            lbl = str(row[label_col]).replace("__label__", "").strip()
            if txt and lbl:
                f.write(f"__label__{lbl} {txt}\n")

#Function to train a fastText model using parameters saved in a dictionary
def train_fasttext(train_txt, model_out, cfg):
    model = fasttext.train_supervised(
        input=str(train_txt),
        epoch=int(cfg["epoch"]),
        lr=float(cfg["lr"]),
        wordNgrams=int(cfg["wordNgrams"]),
        dim=int(cfg["dim"]),
        loss=str(cfg["loss"]),
        minCount=int(cfg["minCount"]),
        minCountLabel=int(cfg["minCountLabel"]),
        ws=int(cfg["ws"]),
        thread=int(cfg["thread"]),
    )
    model.save_model(str(model_out))
    return model

def train_svm(df_train):
    X_train = df_train["abstract"].fillna("").astype(str)
    y_train = df_train["label"].astype(str).str.strip().str.lower()

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b'
        )),
        ("clf", LinearSVC(class_weight="balanced", random_state=42))
    ])

    model.fit(X_train, y_train)
    return model



# Function to address class imbalance by undersampling the quantitative class.
def undersample_quantitative(df, qual_label, quant_label, ratio=1.5, random_state=42):
    s = df.copy()
    s["label_norm"] = s["label"].astype(str).str.strip().str.lower()
    qual = s[s["label_norm"] == str(qual_label).lower()].copy()
    quant = s[s["label_norm"] == str(quant_label).lower()].copy()

    # If class is missing, return dataset unchanged
    if len(qual) == 0 or len(quant) == 0:
        return s.drop(columns=["label_norm"], errors="ignore")

    # Determine maximum allowed quantitative samples based on ratio
    max_quant = int(np.floor(len(qual) * ratio))
    
    # Randomly sample the quantitative class if it exceeds the allowed ratio
    if len(quant) > max_quant:
        quant = quant.sample(n=max_quant, random_state=random_state)
    # Combine the balanced dataset and shuffle
    out = pd.concat([qual, quant], ignore_index=True)
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    return out.drop(columns=["label_norm"], errors="ignore")



def select_for_iteration_svm(model, df_to_score, min_distance=1.0):
    s = df_to_score[["doc_id", "title", "abstract"]].copy()
    text = s["abstract"].fillna("").astype(str).str.replace("\n", " ", regex=False)

    decision = model.decision_function(text)
    pred_label = model.predict(text)

    s["pred_label"] = pred_label
    s["pred_norm"] = pd.Series(pred_label, index=s.index).astype(str).str.strip().str.lower()
    s["distance"] = np.abs(decision)

    auto = s[s["distance"] >= min_distance].copy()
    manual = s.drop(auto.index).copy()

    return auto, manual

def select_for_iteration(model, df_to_score, tau=0.90):
    s = df_to_score[["doc_id", "title", "abstract"]].copy()

    pred = s["abstract"].fillna("").astype(str).apply(
        lambda t: model.predict(t.replace("\n", " "), k=1)
    )

    s["pred_label"] = pred.apply(lambda x: x[0][0] if len(x[0]) > 0 else np.nan)
    s["confidence"] = pred.apply(lambda x: x[1][0] if len(x[1]) > 0 else np.nan)
    s["pred_norm"] = (
        s["pred_label"]
        .astype(str)
        .str.replace("__label__", "", regex=False)
        .str.strip()
        .str.lower()
    )

    auto = s[s["confidence"] >= tau].copy()
    manual = s.drop(auto.index).copy()

    return auto, manual

# Function to adjust parameters during iterative training.
def ramp(it, start, end, total):
    t = min(1.0, (it - 1) / max(1, total - 1))
    return start + t * (end - start)

def normalize_doc_id(df):
    df = df.copy()
    df["doc_id"] = df["doc_id"].astype(str).str.strip()
    bad = df["doc_id"].isin(["", "nan", "None", "<NA>"])
    if bad.any():
        h = pd.util.hash_pandas_object(df.loc[bad, ["title", "abstract"]].fillna(""), index=False).astype(str)
        df.loc[bad, "doc_id"] = "MISS_" + h
    return df

def eval_metrics_ft(model, df_eval, y_col="y_true"):
    if len(df_eval) == 0:
        return {
            "acc": np.nan,
            "f1_macro": np.nan,
            "f1_qual": np.nan,
            "f1_quant": np.nan,
            "precision_macro": np.nan,
            "recall_macro": np.nan,
        }

    out = df_eval["abstract"].apply(lambda t: model.predict(str(t).replace("\n", " "), k=1))
    y_pred = out.apply(lambda r: r[0][0].replace("__label__", "").strip().lower())
    y_true = df_eval[y_col].astype(str).str.strip().str.lower()

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, labels=[label_qual, label_quant], average="macro", zero_division=0)

    rpt = classification_report(
        y_true,
        y_pred,
        labels=[label_qual, label_quant],
        output_dict=True,
        zero_division=0,
    )

    return {
        "acc": float(acc),
        "f1_macro": float(f1m),
        "f1_qual": float(rpt.get(label_qual, {}).get("f1-score", 0.0)),
        "f1_quant": float(rpt.get(label_quant, {}).get("f1-score", 0.0)),
        "precision_macro": float(rpt.get("macro avg", {}).get("precision", 0.0)),
        "recall_macro": float(rpt.get("macro avg", {}).get("recall", 0.0)),
    }

def eval_metrics_svm(model, df_eval, y_col="y_true"):

    # Handle empty eval set
    if len(df_eval) == 0:
        return {
            "acc": np.nan,
            "f1_macro": np.nan,
            "f1_qual": np.nan,
            "f1_quant": np.nan,
            "precision_macro": np.nan,
            "recall_macro": np.nan,
        }

    # --- Prepare inputs ---
    X_eval = df_eval["abstract"].fillna("").astype(str)

    # Predictions (KEEP SAME INDEX)
    y_pred = pd.Series(
        model.predict(X_eval),
        index=df_eval.index
    ).astype(str).str.strip().str.lower()

    # Ground truth
    y_true = df_eval[y_col].astype(str).str.strip().str.lower()

    # --- Convert to arrays (avoid pandas alignment issues) ---
    y_true_arr = y_true.to_numpy()
    y_pred_arr = y_pred.to_numpy()

    # --- Metrics ---
    acc = accuracy_score(y_true_arr, y_pred_arr)

    f1m = f1_score(
        y_true_arr,
        y_pred_arr,
        labels=[label_qual, label_quant],
        average="macro",
        zero_division=0
    )

    rpt = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=[label_qual, label_quant],
        output_dict=True,
        zero_division=0,
    )

    return {
        "acc": float(acc),
        "f1_macro": float(f1m),
        "f1_qual": float(rpt.get(label_qual, {}).get("f1-score", 0.0)),
        "f1_quant": float(rpt.get(label_quant, {}).get("f1-score", 0.0)),
        "precision_macro": float(rpt.get("macro avg", {}).get("precision", 0.0)),
        "recall_macro": float(rpt.get("macro avg", {}).get("recall", 0.0)),
    }

def pick_by_pos(df, pos_list):
    pos = pd.Series(pos_list, dtype="int64")
    pos0 = pos - 1
    valid = pos0[(pos0 >= 0) & (pos0 < len(df))].drop_duplicates().tolist()
    return df.iloc[valid].copy()

