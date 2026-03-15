# ============================================================
# 📊 SENTIMENT ANALYSIS — FULL EDA SCRIPT
# Works for: airline.tsv, eco_news.tsv, glo_warm.tsv, text_emo.tsv
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------
# 1️⃣ Load All Datasets
# ---------------------------------------------
# Adjust the file paths if needed
files = {
    "airline": "non_cuda/airline.tsv",
    "eco_news": "non_cuda/eco_news.tsv",
    "glo_warm": "non_cuda/glo_warm.tsv",
    "text_emo": "non_cuda/text_emo.tsv"
}

dataframes = []
for name, path in files.items():
    try:
        df_temp = pd.read_csv(path, sep='\t', encoding="latin1")
        df_temp['source'] = name
        dataframes.append(df_temp)
        print(f"✅ Loaded {name} — {df_temp.shape}")
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")

# Merge all datasets
df = pd.concat(dataframes, ignore_index=True)
print("\n🔹 Total merged shape:", df.shape)

# ---------------------------------------------
# 2️⃣ Clean and Normalize
# ---------------------------------------------
df.columns = [c.lower().strip() for c in df.columns]
if 'text' not in df.columns:
    # Try to auto-detect text-like column
    for col in df.columns:
        if 'tweet' in col or 'content' in col or 'sentence' in col:
            df.rename(columns={col: 'text'}, inplace=True)
            break

if 'label' not in df.columns:
    # Try to auto-detect label column
    for col in df.columns:
        if 'sentiment' in col or 'emotion' in col or 'category' in col:
            df.rename(columns={col: 'label'}, inplace=True)
            break

# Drop missing values
df = df.dropna(subset=['text', 'label']).reset_index(drop=True)

# Normalize labels
df['label'] = df['label'].astype(str).str.lower().str.strip()
print("\n🧹 Cleaned data preview:")
print(df.head())

# ---------------------------------------------
# 3️⃣ Text Length Analysis
# ---------------------------------------------
df['text_len'] = df['text'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(7,5))
sns.countplot(x="label", data=df, order=df["label"].value_counts().index, palette="Set2")
plt.title("Label Distribution Across All Datasets")
plt.xlabel("Sentiment Label")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["text_len"], bins=50, kde=True, color="skyblue")
plt.title("Distribution of Text Lengths")
plt.xlabel("Number of Words per Text")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x="label", y="text_len", data=df, palette="coolwarm")
plt.title("Text Length by Sentiment Label")
plt.xlabel("Label")
plt.ylabel("Number of Words")
plt.show()

# ---------------------------------------------
# 4️⃣ WordClouds per Sentiment
# ---------------------------------------------
for sentiment in df['label'].unique():
    text = " ".join(df[df['label'] == sentiment]['text'])
    if len(text.strip()) == 0:
        continue
    wordcloud = WordCloud(
        width=1000, height=500,
        background_color="white",
        colormap="viridis",
        max_words=150
    ).generate(text)
    plt.figure(figsize=(10,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"WordCloud for '{sentiment}' Sentiment", fontsize=16)
    plt.show()

# ---------------------------------------------
# 5️⃣ Top Words by Label
# ---------------------------------------------
def get_top_n_words(corpus, n=15):
    words = " ".join(corpus).split()
    common = Counter(words).most_common(n)
    return pd.DataFrame(common, columns=["word", "count"])

for sentiment in df["label"].unique():
    top_words = get_top_n_words(df[df["label"] == sentiment]["text"], n=15)
    plt.figure(figsize=(8,4))
    sns.barplot(x="count", y="word", data=top_words, palette="Set1")
    plt.title(f"Top Words in '{sentiment}' Class")
    plt.xlabel("Count")
    plt.ylabel("Word")
    plt.show()

# ---------------------------------------------
# 6️⃣ TF-IDF Feature Correlation (Top 20)
# ---------------------------------------------
vectorizer = TfidfVectorizer(max_features=20)
tfidf_sample = vectorizer.fit_transform(df["text"])
tfidf_df = pd.DataFrame(tfidf_sample.toarray(), columns=vectorizer.get_feature_names_out())

plt.figure(figsize=(10,8))
sns.heatmap(tfidf_df.corr(), cmap="coolwarm", linewidths=0.5)
plt.title("TF-IDF Feature Correlation Heatmap (Top 20 Terms)")
plt.show()

# ---------------------------------------------
# 7️⃣ Dataset Source Comparison
# ---------------------------------------------
if "source" in df.columns:
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x="source", hue="label", palette="Spectral")
    plt.title("Label Distribution per Dataset Source")
    plt.xlabel("Dataset Source")
    plt.ylabel("Count")
    plt.legend(title="Sentiment")
    plt.show()

print("\n✅ EDA Complete!")
