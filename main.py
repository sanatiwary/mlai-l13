import pandas as pd
from sklearn.model_selection import train_test_split

sentimentDF = pd.read_csv("train.csv", encoding='unicode_escape')
print(sentimentDF.head(5))
print(sentimentDF["sentiment"].value_counts())

sentimentDF["sentiment"] = sentimentDF["sentiment"].replace(
    {"positive" : 0, "negative" : 1, "neutral" : 2}
)

x = sentimentDF["text"]
y = sentimentDF["sentiment"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=1)
print(yTrain.value_counts())
print(yTest.value_counts())

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_https_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords")