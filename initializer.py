import argparse
import logging
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

"""
Get the column transformer based on the preprocessing method

args:
    train_data: The training data
    cat_preprocessing: The type of encoding method to use for categorical features
    num_preprocessing: The type of scaling method to use for numerical features

returns:
    column_transformer: The column transformer to use for preprocessing
"""
def get_column_transformer(train_data, cat_preprocessing, num_preprocessing):
    num_features = train_data.select_dtypes(include=["int64", "float64"]).columns
    cat_features = train_data.select_dtypes(include=["object"]).columns

    cat_encoder, num_encoder = get_encoder(cat_preprocessing, num_preprocessing)

    return ColumnTransformer(
        transformers=[
            ("num", num_encoder, num_features),
            ("cat", cat_encoder, cat_features),
        ]
    )
    
"""
Get the encoder based on the preprocessing method

args:
    cat_preprocessing: The type of encoding method to use for categorical features

returns:
    cat_encoder: The categorical encoder to use
"""
def get_cat_encoder(cat_preprocessing):
    if cat_preprocessing == "OneHotEncoder":
        return OneHotEncoder()
    elif cat_preprocessing == "OrdinalEncoder":
        return OrdinalEncoder()
    elif cat_preprocessing == "TargetEncoder":
        return TargetEncoder()
    else:
        raise ValueError(f"Invalid categorical preprocessing method: {cat_preprocessing}")

"""
Get the encoder based on the preprocessing method

args:
    num_preprocessing: The type of scaling method to use for numerical features

returns:
    num_encoder: The numerical encoder to use
"""
def get_num_encoder(num_preprocessing):
    if num_preprocessing == "None":
        return None
    elif num_preprocessing == "StandardScaler":
        return StandardScaler()
    else:
        raise ValueError(f"Invalid numerical preprocessing method: {num_preprocessing}")

"""
Get the encoder based on the preprocessing method
    
args:
    cat_preprocessing: The type of encoding method to use for categorical features
    num_preprocessing: The type of scaling method to use for numerical features

returns:
    cat_encoder: The categorical encoder to use
    num_encoder: The numerical encoder to use
"""
def get_encoder(cat_preprocessing, num_preprocessing):
    cat_encoder = get_cat_encoder(cat_preprocessing)
    num_encoder = get_num_encoder(num_preprocessing)
    return cat_encoder, num_encoder

"""
Get the model based on the model type

args:
    model_type: The type of model to use for prediction

returns:
    model: The model to use for prediction
"""
def get_model(model_type):
    if model_type == "LogisticRegression":
        return LogisticRegression()
    elif model_type == "RandomForestClassifier":
        return RandomForestClassifier()
    elif model_type == "GradientBoostingClassifier":
        return GradientBoostingClassifier()
    elif model_type == "HistGradientBoostingClassifier":
        return HistGradientBoostingClassifier()
    elif model_type == "MLPClassifier":
        return MLPClassifier()
    else:
        raise ValueError(f"Invalid model type: {model_type}")

"""
Write the predictions to the output file

args:
    output_file: The file to save the test predictions
    predictions: The predictions to save
"""
def write_predictions(output_file, predictions):
    logging.info(f"Writing the predictions to {output_file}")
    predictions.to_csv(output_file, index=False)
    logging.info("Predictions written successfully")

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
    parser.add_argument(
        "numerical_preprocessing",
        type=str,
        choices=["None", "StandardScaler"],
        help="The type of scaling method to use for numerical features",
    )
    parser.add_argument(
        "categorical_preprocessing",
        type=str,
        choices=["OneHotEncoder", "OrdinalEncoder", "TargetEncoder"],
        help="The type of encoding method to use for categorical features",
    )
    parser.add_argument(
        "model_type",
        type=str,
        choices=["LogisticRegression", "RandomForestClassifier", 
                 "GradientBoostingClassifier", "HistGradientBoostingClassifier", 
                 "MLPClassifier"],
        help="The type of model to use for prediction",
    )
    parser.add_argument(
        "test_prediction_output_file",
        type=str,
        help="The file to save the test predictions. Must conform to the submission format",
        default="data/test_predictions.csv"
    )

    return parser.parse_args()
