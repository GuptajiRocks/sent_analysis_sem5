import pandas as pd
from datasets import load_dataset
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Load local TSV datasets ---
paths = {
    "airline": "airline.tsv",
    "eco_news": "eco_news.tsv",
    "glo_warm": "glo_warm.tsv",
    "text_emo": "text_emo.tsv"
}

dfs = []
for name, path in paths.items():
    df = pd.read_csv(path, sep="\t", encoding="latin1")
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if "text" not in df or "label" not in df:
        raise ValueError(f"Missing text/label columns in {name}")
    df["source"] = name
    dfs.append(df[["text", "label", "source"]])

# --- Load HF emotion datasets ---
hf_emotion = load_dataset("dair-ai/emotion")
hf_tweeteval = load_dataset("tweet_eval", "sentiment")

def hf_to_df(ds, source):
    rows = []
    for split in ds.keys():
        for ex in ds[split]:
            rows.append({"text": ex["text"], "label": str(ex["label"]), "source": source})
    return pd.DataFrame(rows)

dfs.append(hf_to_df(hf_emotion, "HF_Emotion"))
dfs.append(hf_to_df(hf_tweeteval, "TweetEval"))

# --- Combine all datasets ---
combined_df = pd.concat(dfs, ignore_index=True)

# --- Display unique class labels ---
# unique_labels = sorted(combined_df["label"].astype(str).str.strip().str.lower().unique())
# print("Total unique classes:", len(unique_labels))
# print("Classes:\n", unique_labels)

full_text = " ".join(combined_df["text"].astype(str).tolist())

wc = WordCloud(
    width=1600,
    height=900,
    background_color="white",
    max_words=300,
    collocations=False
).generate(full_text)

plt.figure(figsize=(16,9))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()