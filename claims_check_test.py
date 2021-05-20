# PACKAGE IMPORTS:
import pandas as pd
from test_data_preprocessing import TestData
import pickle

# LIST OF CONSTANTS:
PICKLED_MODEL_PATH = 'claimClassifier.pkl'
PICKLED_VECTORIZER_PATH = 'claimClassifierVectorizer.pkl'
COL_NAMES = ['TAGS', 'STATEMENT']
LOAD_MODE = 'rb'

testData = TestData()  # Create an object of TestData Class


# LOAD PICKLED MODEL
class PredictTestData:

    def __init__(self):
        self.pickled_model_path = PICKLED_MODEL_PATH
        self.pickled_vectorizer_path = PICKLED_VECTORIZER_PATH
        self.pickled_model_in_use = pickle.load(open(self.pickled_model_path, LOAD_MODE))
        self.pickled_classifier_in_use = pickle.load(open(self.pickled_vectorizer_path, LOAD_MODE))

    def make_prediction(self):
        new_pred = self.pickled_model_in_use.predict(
            self.pickled_classifier_in_use.transform(testData.create_test_data().STATEMENT))
        new_pred_proba = self.pickled_model_in_use.predict_proba(
            self.pickled_classifier_in_use.transform(testData.create_test_data().STATEMENT))

        new_pred_dict = {"STATEMENT": testData.create_test_data().STATEMENT, "PREDICTED_TAG": new_pred,
                         "PROBABILITY_NONCLAIM": new_pred_proba[:, 0], "PROBABILITY_CLAIM": new_pred_proba[:, 1]}
        test_data_prediction = pd.DataFrame(data=new_pred_dict)

        for idx in test_data_prediction.loc[test_data_prediction.PREDICTED_TAG == 1].index:
            print(testData.base_test_data.loc[idx, COL_NAMES[1]])
