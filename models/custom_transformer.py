import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def tokenize(text):
    '''
    Working:
        Takes the text to normalize case, remove punctuation,
        tokenize, lemmatize and remove english stop words
    Input:
        text: Message text
    Output:
        tokens: Message text after performing the above operations
    '''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

class TextLength(BaseEstimator, TransformerMixin):
    '''
    Custom made transformer class for text length feature to be used in ML Pipeline
    that returns the length of the text documents.
    '''
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return np.array([len(text) for text in x]).reshape(-1, 1)
