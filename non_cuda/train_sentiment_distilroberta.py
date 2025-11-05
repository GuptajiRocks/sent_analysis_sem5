# ==========================================
# Sentiment Analysis Pipeline (DistilRoBERTa on CPU)
# ==========================================

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from torch.nn.functional import softmax
import joblib
import os

# Ensure we use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")
print(f"⚙️ Running on: {device}")

# ==========================================
# 1️⃣ Load all sentiment datasets
# ==========================================

airline = pd.read_csv("airline.tsv", sep="\t", encoding="latin1")
eco = pd.read_csv("eco_news.tsv", sep="\t", encoding="latin1")
glo = pd.read_csv("glo_warm.tsv", sep="\t", encoding="latin1")
emo = pd.read_csv("text_emo.tsv", sep="\t", encoding="latin1")

# Identify text and label columns dynamically
def detect_columns(df):
    text_col = df.columns[0]
    label_col = df.columns[1]
    return text_col, label_col

datasets = [airline, eco, glo, emo]
all_data = []

for data in datasets:
    text_col, label_col = detect_columns(data)
    df = data[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]
    all_data.append(df)

df = pd.concat(all_data, ignore_index=True)
df.dropna(inplace=True)

# ==========================================
# 2️⃣ Normalize and clean labels
# ==========================================

def normalize_label(label):
    label = str(label).lower().strip()
    if label in ["positive", "yes", "y", "happiness", "happy", "enthusiasm", "love", "fun", "surprise", "relief"]:
        return "positive"
    elif label in ["negative", "no", "n", "hate", "anger", "fear", "sadness", "boredom", "worry"]:
        return "negative"
    else:
        return "neutral"

df["label"] = df["label"].apply(normalize_label)
df = df[df["text"].astype(str).str.strip() != ""]
df.reset_index(drop=True, inplace=True)

print("✅ Data prepared. Label distribution:")
print(df["label"].value_counts())

# ==========================================
# 3️⃣ Encode labels & Split
# ==========================================

le = LabelEncoder()
df["label_enc"] = le.fit_transform(df["label"])

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label_enc"], test_size=0.2, stratify=df["label_enc"], random_state=42
)

train_df = pd.DataFrame({"text": X_train, "labels": y_train})
test_df  = pd.DataFrame({"text": X_test, "labels": y_test})

train_ds = Dataset.from_pandas(train_df)
test_ds  = Dataset.from_pandas(test_df)

# ==========================================
# 4️⃣ Tokenization (DistilRoBERTa)
# ==========================================

model_name = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
test_ds  = test_ds.map(tokenize, batched=True)

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ==========================================
# 5️⃣ Model & Training Setup
# ==========================================

num_labels = len(le.classes_)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

training_args = TrainingArguments(
    output_dir="./sentiment_distilroberta",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,   # smaller for CPU
    per_device_eval_batch_size=8,
    num_train_epochs=2,              # shorter for CPU
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
)

# ==========================================
# 6️⃣ Train the Model
# ==========================================

print("\n🚀 Starting training (CPU)... this will take a while...")
trainer.train()

# ==========================================
# 7️⃣ Evaluate the Model
# ==========================================

preds = trainer.predict(test_ds)
y_pred = np.argmax(preds.predictions, axis=1)
y_true = preds.label_ids

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))
print("✅ Accuracy:", round(accuracy_score(y_true, y_pred), 4))

# ==========================================
# 8️⃣ Save Model & Label Encoder
# ==========================================

save_dir = "./sentiment_distilroberta_finetuned"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
joblib.dump(le, os.path.join(save_dir, "label_encoder.pkl"))

print(f"\n💾 Model and tokenizer saved successfully to: {save_dir}")

# ==========================================
# 9️⃣ Prediction Helper
# ==========================================

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=-1).flatten().numpy()
    results = sorted(zip(le.classes_, probs), key=lambda x: x[1], reverse=True)
    return {"input": text, "predicted": results[0][0], "probabilities": results}

# Example
example = "The company's new product launch was a complete success!"
print("\n🔮 Prediction Example:")
print(predict_sentiment(example))
