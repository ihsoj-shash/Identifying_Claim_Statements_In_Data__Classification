# PACKAGE IMPORTS:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# LIST OF CONSTANTS:
LANGUAGE = 'english'
COL_NAMES = ['TAGS', 'STATEMENT']
TAG_VALUES = ['nonClaim', 'claim']
TRAIN_FILE_PATH = 'cl_nc.csv'
PICKLED_MODEL_PATH = 'claimClassifier.pkl'
PICKLED_VECTORIZER_PATH = 'claimClassifierVectorizer.pkl'
DUMP_MODE = 'wb'
PERCENT_CONVERTER = 100
SEPARATOR = '\t'
# MIN_IGNORANCE=0.01 # ignore the terms appearing in less than (MIN_IGNORANCE*100)% of documents
# MAX_IGNORANCE=0.75 # ignore the terms appearing in more than (MAX_IGNORANCE*100)% of documents
VALIDATION_PERCENTAGE = 0.2
WORDS_IN_UNIGRAM = 1
WORDS_IN_TRIGRAM = 3
HEADER=0
RANDOM_STATE = 20

# PULLING DATA:
cnc = pd.read_csv(TRAIN_FILE_PATH, sep=SEPARATOR, header=HEADER, names=[COL_NAMES[0], COL_NAMES[1]])

# DATA CLEANING:
cnc.STATEMENT = cnc[COL_NAMES[1]].str.lower()
cnc.TAGS.replace(to_replace=[TAG_VALUES[0], TAG_VALUES[1]], value=[0, 1], inplace=True)

# FEATURE CREATION
vectorizer = CountVectorizer(stop_words=LANGUAGE, ngram_range=[WORDS_IN_UNIGRAM, WORDS_IN_TRIGRAM])
all_features = vectorizer.fit_transform(cnc.STATEMENT)

# TRAIN_VALIDATION SPLIT
X_train, X_val, y_train, y_val = train_test_split(all_features, cnc[COL_NAMES[0]], random_state=RANDOM_STATE,
                                                  test_size=VALIDATION_PERCENTAGE)

# MODEL CALL AND FITTING ON DATA
claim_classifier = MultinomialNB()
claim_classifier.fit(X_train, y_train)

# VALIDATION METRICS CALCULATION
val_nr_correct = (y_val == claim_classifier.predict(X_val)).sum()
val_nr_incorrect = y_val.size - val_nr_correct
val_fraction_wrong = val_nr_incorrect / (val_nr_correct + val_nr_incorrect)
val_accuracy = (1 - val_fraction_wrong) * PERCENT_CONVERTER

# PICKLING
pickle.dump(vectorizer, open(PICKLED_VECTORIZER_PATH, DUMP_MODE))
pickle.dump(claim_classifier, open(PICKLED_MODEL_PATH, DUMP_MODE))
