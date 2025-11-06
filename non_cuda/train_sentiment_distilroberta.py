# ==========================================
# Sentiment Analysis (DistilRoBERTa) - CPU Safe Version
# ==========================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

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

# ==========================================
# 1️⃣ Environment setup
# ==========================================
device = torch.device("cpu")
print(f"⚙️ Running on: {device}")

# ==========================================
# 2️⃣ Load all datasets
# ==========================================

def load_tsv(path):
    df = pd.read_csv(path, sep="\t")
    # try to detect text and label columns
    text_col, label_col = df.columns[0], df.columns[1]
    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]
    return df

# Load all provided datasets
airline = pd.read_csv("airline.tsv", sep="\t", encoding="latin1")
eco = pd.read_csv("eco_news.tsv", sep="\t", encoding="latin1")
glo = pd.read_csv("glo_warm.tsv", sep="\t", encoding="latin1")
emo = pd.read_csv("text_emo.tsv", sep="\t", encoding="latin1")

# Combine
df = pd.concat([airline, eco, glo, emo], ignore_index=True)
df.dropna(inplace=True)

# ==========================================
# 3️⃣ Normalize labels
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
df = df[df["text"].astype(str).str.strip() != ""].reset_index(drop=True)

print("✅ Data prepared. Label distribution:")
print(df["label"].value_counts())

# ==========================================
# 4️⃣ Encode labels and split
# ==========================================

le = LabelEncoder()
df["label_enc"] = le.fit_transform(df["label"])

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label_enc"], test_size=0.2, stratify=df["label_enc"], random_state=42
)

train_df = pd.DataFrame({"text": X_train, "labels": y_train})
test_df = pd.DataFrame({"text": X_test, "labels": y_test})

train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# ==========================================
# 5️⃣ Tokenizer (DistilRoBERTa)
# ==========================================

model_name = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ==========================================
# 6️⃣ Model setup
# ==========================================

num_labels = len(le.classes_)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# ==========================================
# 7️⃣ Training configuration
# ==========================================

training_args = TrainingArguments(
    output_dir="./sentiment_distilroberta",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
)

# ==========================================
# 8️⃣ Train the model
# ==========================================

print("\n🚀 Starting training (CPU)... This may take some time.")
trainer.train()

# ==========================================
# 9️⃣ Evaluate
# ==========================================

preds = trainer.predict(test_ds)
y_pred = np.argmax(preds.predictions, axis=1)
y_true = preds.label_ids

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))
print("✅ Accuracy:", round(accuracy_score(y_true, y_pred), 4))

# ==========================================
# 🔟 Save model and tokenizer
# ==========================================

save_dir = "./sentiment_distilroberta_finetuned"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
joblib.dump(le, os.path.join(save_dir, "label_encoder.pkl"))
print(f"\n💾 Model saved to: {save_dir}")

# ==========================================
# 1️⃣1️⃣ Prediction function
# ==========================================

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=-1).flatten().numpy()
    results = sorted(zip(le.classes_, probs), key=lambda x: x[1], reverse=True)
    return {"text": text, "predicted": results[0][0], "probabilities": results}

# Example usage
example = "I was using the product, it has a critical issue while using it for the first time, thought it was the defect of the product, but turned out its my defect, but the product is very good."
print("\n🔮 Prediction Example:")
print(predict_sentiment(example))
