# sentiment_analysis_pipeline_v2.py

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 1️⃣ Cleaning Function
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

# ---------------------------
# 2️⃣ Unified Label Mapping
# ---------------------------
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

    # text_emo.tsv (emotion-based)
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

# ---------------------------
# 3️⃣ Generic Loader Function
# ---------------------------
def load_and_clean(path):
    df = pd.read_csv(path, sep="\t", encoding="latin1")
    df.columns = [c.lower() for c in df.columns]
    # Find likely text and label columns
    text_col = [c for c in df.columns if "text" in c or "tweet" in c or "message" in c][0]
    label_col = [c for c in df.columns if "label" in c or "sentiment" in c or "emotion" in c][0]
    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]
    df["label"] = df["label"].astype(str).str.lower().map(label_map)
    df = df[df["label"].isin(["positive", "negative", "neutral"])]
    df["text"] = df["text"].apply(clean_text)
    df.drop_duplicates(subset="text", inplace=True)
    return df

# ---------------------------
# 4️⃣ Load and Merge Datasets
# ---------------------------
print("📂 Loading datasets...")

df_air = load_and_clean("../data/pick/airline.tsv")
df_eco = load_and_clean("../data/pick/eco_news.tsv")
df_glo = load_and_clean("../data/pick/glo_warm.tsv")
df_txt = load_and_clean("../data/pick/text_emo.tsv")

df = pd.concat([df_air, df_eco, df_glo, df_txt], ignore_index=True)
print("✅ Combined Dataset Shape:", df.shape)
print(df["label"].value_counts())

# ---------------------------
# 5️⃣ Encode Labels + Split
# ---------------------------
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label_encoded"],
    test_size=0.2, random_state=42, stratify=df["label_encoded"]
)

# ---------------------------
# 6️⃣ TF-IDF Vectorization
# ---------------------------
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------
# 7️⃣ Model Training
# ---------------------------
model = LogisticRegression(max_iter=400, class_weight='balanced', solver='lbfgs')
model.fit(X_train_vec, y_train)

# ---------------------------
# 8️⃣ Evaluation
# ---------------------------
y_pred = model.predict(X_test_vec)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred), 3))

# ---------------------------
# 9️⃣ Prediction Function
# ---------------------------
def predict_sentiment(text):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    probs = model.predict_proba(vec)[0]
    result = sorted(zip(le.classes_, probs), key=lambda x: x[1], reverse=True)
    top_label = result[0][0]
    return {"input": text, "predicted": top_label, "probabilities": result}

# Example Prediction
sample_text = "I absolutely loved the new policy changes, they are great!"
print("\n🔍 Example Prediction:\n")
print(predict_sentiment(sample_text))

# ---------------------------
# 🔟 Save Model
# ---------------------------
joblib.dump((vectorizer, model, le), "sentiment_model.pkl")
print("\n💾 Model saved as 'sentiment_model.pkl'")
