# DisasterResponsePipeline
Web app project with ETL, NPL and ML pipelines to handle real messages that were sent during disaster events with the purpose to be sent to an appropriate disaster relief agency. 

# Project Overview
This project is part of the Data Science NanoDegree from Udacity. With data from Figure Eight, the main goal is to analyze disaster data to build a model for an API that classifies disaster messages. The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

# Files

* process_data.py -> This file includes the code that loads the data, cleans the data and saves the data into a sql_database.

* train_classifier.py -> This file loads and split data into training X,y sets, it also builds a nlp pipeline and uses a gridsearch object to evaluate the model.

* custom_transformer.py -> This file contains the tokenize function that will handle the text processing of the messages, and it also contains the custom made transformer class TextLength for creating a text length feature to be used in the ML pipeline. 

* run.py -> This file has the flask web-app code that loads the model, display the visuals and receive user input text for modeling.

* disaster_categories.csv -> categories data

* disaster_messages.csv -> messages data

# Installations

re
pandas
numpy
nltk
pickle
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'])
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, recall_score
from custom_transformer import tokenize, TextLength


