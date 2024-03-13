import numpy as np
import pandas as pd
import logging
import initializer as init
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from cleaner import clean_data
from data_exploration import explore_data

SEED = 42

"""
Read the data from the input files

args:
    input_file: The file containing the training data
    labels_file: The file containing the training labels
    test_file: The file containing the test data

returns:
    x_train: The training data
    y_train: The training labels
    x_test: The test data
"""
def read_data(input_file, labels_file, test_file):
    print("----------------- READING DATA -----------------")
    x_train = pd.read_csv(input_file)
    y_train = pd.read_csv(labels_file)
    x_test = pd.read_csv(test_file)

    print("Training data shape:", x_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Test data shape:", x_test.shape)
    print("----------------- DATA READ -----------------\n")
    return x_train, y_train, x_test

"""
Write the predictions to the output file following the submission format

args:
    output_file: The file to save the test predictions
    predictions: The predictions to save
"""
def write_predictions(output_file, predictions):
    logging.info(f"Writing the predictions to {output_file}")
    np.savetxt(output_file, predictions, delimiter=",", fmt="%s")
    logging.info("Predictions written successfully")

def main():
    logging.basicConfig(level=logging.INFO)
    args = init.get_args()

    x_train, y_train, x_test = read_data(args.train_input_file, 
                                         args.train_labels_file, 
                                         args.test_input_file)
    model = init.get_model(args.model_type)

    # explore_data(x_train, y_train)

    x_train, x_test = clean_data(x_train, x_test)

    column_transformer = init.get_column_transformer(x_train, 
                                                     args.categorical_preprocessing,
                                                     args.numerical_preprocessing)

    train_pipeline = make_pipeline(column_transformer, model)

    print("----------------- TRAINING MODEL -----------------")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(train_pipeline, x_train, y_train["status_group"], cv=skf, scoring='accuracy')
    print(f"Cross-validated accuracy: {np.mean(scores)} Â± {np.std(scores)}")
    print("----------------- MODEL TRAINED -----------------\n")

    print("----------------- MAKING PREDICTIONS -----------------")
    train_pipeline.fit(x_train, y_train["status_group"])
    predictions = train_pipeline.predict(x_test)
    write_predictions(args.test_prediction_output_file, predictions)
    print("----------------- PREDICTIONS MADE -----------------\n")

if __name__ == "__main__":
    main()
