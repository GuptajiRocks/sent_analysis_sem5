from helper import clean_text
import pandas as pd

airline_df = pd.read_csv("../data/crowdflower/airline-sentiment/train.tsv", sep="\t", encoding="latin1")

airline_df["clean"] = airline_df["text"].apply(clean_text)
airline_df.drop("text", axis=1, inplace=True)
print(airline_df.head())
