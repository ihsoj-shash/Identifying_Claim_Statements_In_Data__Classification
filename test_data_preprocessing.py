# PACKAGE IMPORTS:
import pandas as pd

# LIST OF CONSTANTS:
TEST_DATA_PATH = 'test_claim_nonClaim.csv'
COL_NAMES = ['TAGS', 'STATEMENT']
TAG_VALUES = ['nonClaim', 'claim']
HEADER = 0


class TestData:
    def __init__(self):
        self.test_data_path = TEST_DATA_PATH
        self.base_test_data = pd.read_csv(self.test_data_path, header=HEADER, names=[COL_NAMES[0], COL_NAMES[1]])
        self.test_data = pd.read_csv(self.test_data_path, header=HEADER, names=[COL_NAMES[0], COL_NAMES[1]])

    def create_test_data(self):  # CLEAN NEW DATA BEING SUPPLIED FOR TESTING
        self.test_data.TAGS.replace(to_replace=[TAG_VALUES[0], TAG_VALUES[1]], value=[0, 1], inplace=True)
        self.test_data.STATEMENT = self.test_data[COL_NAMES[1]].str.lower()
        return self.test_data
