import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from helper import *

airline_df = pd.read_csv("../data/crowdflower/airline-sentiment/train.tsv", sep="\t", encoding="latin1")
econews_df = pd.read_csv("../data/crowdflower/economic-news/train.tsv", sep="\t", encoding="latin1")
text_emo_df = pd.read_csv("../data/pick/text_emo.tsv", sep="\t", encoding="latin1")
glo_warm_df = pd.read_csv("../data/pick/glo_warm.tsv", sep="\t", encoding="latin1")

allDs = [airline_df, econews_df, text_emo_df, glo_warm_df]


def print_all():
    for m in allDs:
        uni_label(m)

print_all()
#cleaning_try()
#print_all()