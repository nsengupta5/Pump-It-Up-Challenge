import argparse

"""
Get the arguments from the command line
:return args: The arguments
"""
def get_args():
    parser = argparse.ArgumentParser(description="Predicting the status of a water pump")

    # Define the arguments
    parser.add_argument(
        "--train-input-file",
        type=str,
        help="The file containing the training data",
        required=True
    )
    parser.add_argument(
        "--train-labels-file",
        type=str,
        help="The file containing the training labels",
        required=True
    )
    parser.add_argument(
        "--test-input-file",
        type=str,
        help="The file containing the test data",
        required=True
    )
    parser.add_argument(
        "--numerical-preprocessing",
        type=str,
        choices=["None", "StandardScaler"],
        help="The type of scaling method to use for numerical features",
        required=True
    )
    parser.add_argument(
        "--categorical-preprocessing",
        type=str,
        choices=["OneHotEncoder", "OrdinalEncoder", "TargetEncoder"],
        help="The type of encoding method to use for categorical features",
        required=True
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["LogisticRegression", "RandomForestClassifier", 
                 "GradientBoostingClassifier", "HistGradientBoostingClassifier", 
                 "MLPClassifier"],
        help="The type of model to use for prediction",
        required=True
    )
    parser.add_argument(
        "--test-prediction-output-file",
        type=str,
        help="The file to save the test predictions. Must conform to the submission format",
        default="data/test_predictions.csv"
    )

    return parser.parse_args()

def main():
    args = get_args()

if __name__ == "__main__":
    main()
