# ==========================================
# Sentiment Analysis Pipeline (TF-IDF + XGBoost + Smart Caching)
# ==========================================

import os
import re
import joblib
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

MODEL_PATH = "sentiment_model_xgb.pkl"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

label_map = {
    # airline.tsv
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",

    # eco_news.tsv
    "yes": "positive",
    "no": "negative",
    "not sure": "neutral",

    # glo_warm.tsv
    "y": "positive",
    "n": "negative",

    # text_emo.tsv
    "happiness": "positive",
    "enthusiasm": "positive",
    "love": "positive",
    "fun": "positive",
    "relief": "positive",
    "surprise": "positive",
    "anger": "negative",
    "hate": "negative",
    "sadness": "negative",
    "worry": "negative",
    "boredom": "negative",
    "neutral": "neutral",
    "empty": "neutral",
}

def load_and_clean(path):
    df = pd.read_csv(path, sep="\t", encoding="latin1")
    df.columns = [c.lower() for c in df.columns]

    # detect text and label columns dynamically
    text_col = [c for c in df.columns if "text" in c or "tweet" in c or "message" in c][0]
    label_col = [c for c in df.columns if "label" in c or "sentiment" in c or "emotion" in c][0]

    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]
    df["label"] = df["label"].astype(str).str.lower().map(label_map)
    df = df[df["label"].isin(["positive", "negative", "neutral"])]
    df["text"] = df["text"].apply(clean_text)
    df.drop_duplicates(subset="text", inplace=True)
    return df

def load_all_datasets():
    print("📂 Loading datasets...")

    df_air = load_and_clean("../data/pick/airline.tsv")
    df_eco = load_and_clean("../data/pick/eco_news.tsv")
    df_glo = load_and_clean("../data/pick/glo_warm.tsv")
    df_txt = load_and_clean("../data/pick/text_emo.tsv")

    df = pd.concat([df_air, df_eco, df_glo, df_txt], ignore_index=True)
    df.dropna(inplace=True)
    print("✅ Combined Dataset Shape:", df.shape)
    print(df["label"].value_counts())
    return df

def train_model():
    df = load_all_datasets()

    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label_encoded"],
        test_size=0.2, stratify=df["label_encoded"], random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("\n🚀 Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        use_label_encoder=False,
        tree_method="hist",      # 🚀 GPU support (fallback to CPU if unavailable)
        predictor="cpu_predictor"
    )

    model.fit(X_train_vec.toarray(), y_train)

    preds = model.predict(X_test_vec.toarray())
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, preds, target_names=le.classes_))
    print("✅ Accuracy:", round(accuracy_score(y_test, preds), 3))

    # Save everything
    joblib.dump((vectorizer, model, le), MODEL_PATH)
    print(f"💾 Model saved successfully as '{MODEL_PATH}'")

    return vectorizer, model, le

def load_or_train_model(force_retrain=False):
    if not force_retrain and os.path.exists(MODEL_PATH):
        print("✅ Found existing model. Loading...")
        vectorizer, model, le = joblib.load(MODEL_PATH)
    else:
        print("⚙️ No existing model found or retraining forced.")
        vectorizer, model, le = train_model()
    return vectorizer, model, le

def predict_sentiments(text, vectorizer, model, le):
    clean = clean_text(text)
    vec = vectorizer.transform([clean]).toarray()
    probs = model.predict_proba(vec)[0]
    result = sorted(zip(le.classes_, probs), key=lambda x: x[1], reverse=True)
    return result
    # return {"input": text, "predicted": result[0][0], "probabilities": result}

if __name__ == "__main__":
    vectorizer, model, le = load_or_train_model(force_retrain=False)

    # Example usage
    sample = "I hate this life"
    print("\nExample Prediction:")
    print(predict_sentiments(sample, vectorizer, model, le)[0])
