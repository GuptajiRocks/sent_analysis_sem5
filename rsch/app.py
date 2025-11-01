import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

airline_df = pd.read_csv("../data/crowdflower/airline-sentiment/train.tsv", sep="\t", encoding="latin1")
corpmsg_df = pd.read_csv("../data/crowdflower/corporate-messaging/train.tsv", sep="\t", encoding="latin1")
econews_df = pd.read_csv("../data/crowdflower/economic-news/train.tsv", sep="\t", encoding="latin1")
def uni_label(df):
    print(df["label"].unique())

def print_all():
    uni_label(airline_df)
    uni_label(corpmsg_df)
    uni_label(econews_df)

def calc_all():
    findf = pd.concat([airline_df, corpmsg_df, econews_df])
    #print(findf.head())

    #print(findf["label"].unique())

    return findf

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


fdf = calc_all()

def cleaning_try():
    econews_df["clean"] = econews_df["text"].apply(clean_text)
    print(econews_df["text"].iloc[0])
    print(econews_df["clean"].iloc[0])

cleaning_try()
