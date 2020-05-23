import re
import sklearn
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from math import log

# Create lemmatizer and stopwords list

mystem = Mystem()
russian_stopwords = stopwords.words("russian")


# Preprocess function
def preprocess_text(text):
    text = re.sub("[^А-Яа-я]", " ", text)
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords \
              and token != " " \
              and token.strip() not in punctuation]
    # text = " ".join(tokens)
    return tokens


def tf_idf(d, w, allDoc):
    tf = d.count(w) / len(d)
    #print('count of words', d.count(w), 'len of d', len(d), 'tf = ', tf)
    df = 0
    for D in allDoc:
        if w in D:
            df += 1
    #print('df = ', df)
    idf = log((len(allDoc) + 1) / (df + 1))
    #print('len of all Doc = ', len(allDoc), 'idf = ', idf)
    vec = tf * idf
    return '{:.3f}'.format(vec)


with open('user1.json') as json_file:
    comp = json.load(json_file)

with open('user2.json') as json_file:
    user = json.load(json_file)
corpusUser = []
for i in range(0, len(user)):
    text = preprocess_text(user[i])
    corpusUser.append(text)
corpusUser = [item for sublist in corpusUser for item in sublist]

corpusComp = []
for i in range(0, len(comp)):
    text = preprocess_text(comp[i])
    corpusComp.append(text)

corpusComp = [item for sublist in corpusComp for item in sublist]

# need to create a vocab
merged = list(corpusUser + corpusComp)
vocab = list(set(merged))


VecUser = [tf_idf(corpusUser, w, merged[1:]) for w in vocab]
VecComp = [tf_idf(corpusComp, w, merged[:1]) for w in vocab]

#create Vector of tf_idfs
import numpy as np
Vec1 = [tf_idf(corpusUser, w, merged[1:]) for w in vocab]

Vec2 = [tf_idf(corpusComp, w, merged[:1]) for w in vocab]


Vec1=np.array(Vec1,dtype=float).reshape(1, len(vocab))
Vec2=np.array(Vec2,dtype=float).reshape(1, len(vocab))


sim = sklearn.metrics.pairwise.cosine_similarity(Vec1, Vec2)
print('similarity is ', sim)