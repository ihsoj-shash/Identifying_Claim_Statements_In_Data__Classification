# Identifying_Claim_Statements_In_Data__Classification
The code creates a model, pickles it and then reuses it for classifying data by identifying claim statements in the data.

NOTE: It's Naive Bayes and not Navive Bayes .... Apologies for the typo. :)

Data Format in raw csv(both train and test) files: ([tag_col, statement_col], [row1col1, row1col2], [row2col1, row2col2]. ....... , [rowncol1, rowncol2])

If one wants to train the model on his data, new pickle files will have to be created by running base_model_with_pickle_creation.py again. Else main.py directly uses pickled files to make predictions on the test dataset.

File Description : 

1. base_model_with_pickle_creation.py --> Collects, cleans and preprocesses the training data. Creates a model, trains it and creates pickled files of both         Vectorizer and model which can be reused for future predictions on a different dataset(with same data domain and format).
2. test_data_preprocessing.py --> Preprocesses the test data.
3. claims_check_test.py --> Runs the Claims check on test data.
4. claims_check_main.py --> main file that calls files 2 and 3 to make predictions.



Data Curated from the reserach paper : https://www.researchgate.net/figure/List-of-ambiguous-promotional-claims-by-pharmaceutical-companies-in-the-light-of_tbl2_6936857/actions#reference --> Thanks to the authors :)

