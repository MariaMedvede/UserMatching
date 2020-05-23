import json
import pandas as pd
import re
import nltk
from nltk import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

#nltk.download('punkt')
#nltk.download('stopwords')

with open('data.txt') as json_file:
    data = json.load(json_file)
    #print(data)

#preproccesing
def clean_text(text):
    """
    Receives a raw review and clean it using the following steps:
    1. Remove all non-words
    2. Transform the review in lower case
    3. Remove all stop words
    4. Perform stemming

    Args:
        review: the review that iwill be cleaned
    Returns:
        a clean review using the mentioned steps above.
    """

    text = re.sub("[^А-Яа-я]", " ", text)
    text = text.lower()
    text = word_tokenize(text)
    stemmer = SnowballStemmer("russian")
    text = [stemmer.stem(word) for word in text if word not in set(stopwords.words("russian"))]
    text = " ".join(text)
    return text


print(data[8])

cleaned_data = clean_text(data[8])
print('cleaned data', cleaned_data)

corpus = []
for i in range(0, len(data)):
    text = clean_text(data[i])
    corpus.append(text)

#print(corpus)

tfidf_vectorizer = TfidfVectorizer()
values = tfidf_vectorizer.fit_transform(corpus)

# Show the Model as a pandas DataFrame
feature_names = tfidf_vectorizer.get_feature_names()
print(pd.DataFrame(values.toarray(), columns = feature_names))