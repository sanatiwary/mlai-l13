import pandas as pd
from sklearn.model_selection import train_test_split

sentimentDF = pd.read_csv("train.csv", encoding='unicode_escape')
print(sentimentDF.head(5))
print(sentimentDF["sentiment"].value_counts())

sentimentDF["sentiment"] = sentimentDF["sentiment"].replace(
    {"positive" : 0, "negative" : 1, "neutral" : 2}
)

x = sentimentDF["selected_text"]
y = sentimentDF["sentiment"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=1)
print("xtrain: ", xTrain.head(5))

print(yTrain.value_counts())
print(yTest.value_counts())

import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

wnl = WordNetLemmatizer()

def transform(data):
    corpus = []

    for i in data:
        newi = re.sub("[^a-zA-Z]", " ", i)
        newi = newi.lower()
        newi = newi.split()
        list1 = [wnl.lemmatize(word) for word in newi if word not in stopwords.words("english")]
        corpus.append(" ".join(list1))

    return corpus

xTrainCorpus = transform(xTrain)
xTestCorpus = transform(xTest)

print(xTrainCorpus[0:5])

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range=(1, 2))
xTrainNew = cv.fit_transform(xTrainCorpus)
xTestNew = cv.fit_transform(xTestCorpus)
print(xTrainNew.shape, xTestNew.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(xTrainNew, xTestNew)
trainPred = rfc.predict(xTrainNew)
testPred = rfc.predict(xTestNew)

def analyzeSentiment(text):
    trText = transform(text)
    trText = cv.transform(trText)
    textPred = rfc.predict(trText)

    if textPred[0] == 0:
        print("positive")
    elif textPred[0] == 1:
        print("negative")
    else:
        print("neutral")

ex0 = ["i love the new dress you're wearing"]
ex1 = ["i don't like the color of your hair"]
ex2 = ["i got a letter in the mail today"]

analyzeSentiment(ex0)
analyzeSentiment(ex1)
analyzeSentiment(ex2)