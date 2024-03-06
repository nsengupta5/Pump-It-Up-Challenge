import argparse
from enum import Enum
import pandas as pd
import logging
from data_exploration import explore_data

SEED = 42
Model = Enum('Model', ['LogisticRegression', 'RandomForestClassifier', 
                       'GradientBoostingClassifier', 'HistGradientBoostingClassifier', 
                       'MLPClassifier'])

NumPreprocessing = Enum('NumPreprocessing', ['None', 'StandardScaler'])
CatPreprocessing = Enum('CatPreprocessing', ['OneHotEncoder', 'OrdinalEncoder', 
                                             'TargetEncoder'])

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
    logging.info(f"Reading the data from {input_file}, {labels_file}, and {test_file}")
    
    x_train = pd.read_csv(input_file)
    y_train = pd.read_csv(labels_file)
    x_test = pd.read_csv(test_file)

    logging.info("Data read successfully")
    return x_train, y_train, x_test

"""
Get the arguments from the command line

returns:
    args: The arguments from the command line
"""
def get_args():
    parser = argparse.ArgumentParser(description="Predicting the status of a water pump")

    # Define the arguments
    parser.add_argument(
        "train_input_file",
        type=str,
        help="The file containing the training data",
    )
    parser.add_argument(
        "train_labels_file",
        type=str,
        help="The file containing the training labels",
    )
    parser.add_argument(
        "test_input_file",
        type=str,
        help="The file containing the test data",
    )
    # parser.add_argument(
    #     "numerical-preprocessing",
    #     type=str,
    #     choices=["None", "StandardScaler"],
    #     help="The type of scaling method to use for numerical features",
    # )
    # parser.add_argument(
    #     "categorical-preprocessing",
    #     type=str,
    #     choices=["OneHotEncoder", "OrdinalEncoder", "TargetEncoder"],
    #     help="The type of encoding method to use for categorical features",
    # )
    # parser.add_argument(
    #     "model-type",
    #     type=str,
    #     choices=["LogisticRegression", "RandomForestClassifier", 
    #              "GradientBoostingClassifier", "HistGradientBoostingClassifier", 
    #              "MLPClassifier"],
    #     help="The type of model to use for prediction",
    # )
    # parser.add_argument(
    #     "test-prediction-output-file",
    #     type=str,
    #     help="The file to save the test predictions. Must conform to the submission format",
    #     default="data/test_predictions.csv"
    # )

    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    x_train, y_train, x_test = read_data(args.train_input_file, 
                                         args.train_labels_file, 
                                         args.test_input_file)
    explore_data(x_train, y_train)

if __name__ == "__main__":
    main()
