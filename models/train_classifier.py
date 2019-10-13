import sys
# libraries
import re
import pandas as pd
import numpy as np
import nltk
import pickle
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


def load_data(database_filepath):
    """Loads the data from a database file and returns
    'X' assign to the message column, 'y' to the categories column
    and a list of column names for categories column.

    input -> str
    output -> DataFrame column, DataFrame column, DataFrame name list
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)
    # X contains the messages column
    X = df.message
    # Y contains all the categories
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    # names for categories columns
    category_names = list(Y)[1:]
    return X, Y, category_names



def build_model():
    """
    Assemble the model and creates the GridSearchCV

    """
    return Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            # custom tranformers StartingVerbExtractor and TextLength
            ('length', TextLength()),
        ])),

        ('clf', MultiOutputClassifier(MultinomialNB())),
    ])

    # Dictionary of parameters for grid search
    parameters = {'features__text_pipeline__vect__ngram_range':[(1,2),(2,2)],
                 'clf__estimator__n_estimators':[50, 100]
             }

    # Create GridSearchCV object
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs = -1, verbose=2)

    return cv




def evaluate_model(model, X_test, Y_test, category_names):
    '''
        Evaluate model performance using test data
        Prints the f1 score, precision and recall for each category
        Also print the Average F1-Score of all output categories

        input -> model, X_test, Y_test, str
        output -> f1-score
    '''

    predictions = model.predict(X_test)
    predictions = pd.DataFrame(predictions, columns = Y_test.columns)

    f1_scores=[]
    # Print the f1 score, precision and recall for each category
    for column in category_names:
        print("Output category:", column)
        print(classification_report(Y_test[column], predictions[column]))
        f1_scores.append(f1_score(Y_test[column], predictions[column], average='macro'))

    # Print the Average F1-Score of all output categories
    print("Average F1-Score of all output categories:", sum(f1_scores)/len(f1_scores))




def save_model(model, model_filepath):
    """Saves the model to a pickle file.
        input -> pipeline, str
        output -> pipeline file
    """
    pipeline_pkl = open(model_filepath, 'wb')
    pickle.dump(model, pipeline_pkl)
    pipeline_pkl.close()



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
