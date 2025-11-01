import pandas as pd

df = pd.read_csv('../data/crowdflower/tweet_global_warming/train.tsv', sep='\t', encoding="latin1")
print(df.tail())