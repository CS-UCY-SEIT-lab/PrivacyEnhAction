import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


import requests
from bs4 import BeautifulSoup
import re

from sentence_splitter import SentenceSplitter


from nltk.stem.snowball import SnowballStemmer

import sys
import warnings

import pickle

dataset = os.path.join('uploads', 'all_devices.csv')



df = pd.read_csv(dataset)


data = df

if not sys.warnoptions:
    warnings.simplefilter("ignore")
def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext
def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned
def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent
data['policy'] = data['policy'].str.lower()
data['policy'] = data['policy'].apply(cleanHtml)
data['policy'] = data['policy'].apply(cleanPunc)
data['policy'] = data['policy'].apply(keepAlpha)



from sklearn.utils import shuffle



stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence
data['policy'] = data['policy'].apply(stemming)

data = shuffle(data)

X = data["policy"]
y = np.asarray(data[data.columns[1:]])
    



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42,shuffle=True)

vetorizar = TfidfVectorizer(
    stop_words='english',
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{2,}',  #vectorize 2-character words or more
    ngram_range=(1, 1),
    max_features=30000)


vetorizar.fit(X_train)


X_train_tfidf = vetorizar.transform(X_train)
X_test_tfidf = vetorizar.transform(X_test)

from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC

# initialize LabelPowerset multi-label classifier with a RandomForest
classifier = BinaryRelevance(
    classifier = SVC(),
    require_dense = [False, True])

# train
classifier.fit(X_train_tfidf, y_train)

# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))

