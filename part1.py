import numpy as np
import pandas as pd
import logging
import initializer as init
from cleaner import clean_data
from utils import print_header
from hpo import get_best_hyperparams
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

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
    print_header("READING DATA", True)
    x_train = pd.read_csv(input_file)
    y_train = pd.read_csv(labels_file)
    x_test = pd.read_csv(test_file)

    print("Training data shape:", x_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Test data shape:", x_test.shape)
    print_header("DATA READ", False)
    return x_train, y_train, x_test

"""
Write the predictions to the output file following the submission format

args:
    output_file: The file to save the test predictions
    predictions: The predictions to save
"""
def write_predictions(output_file, predictions):
    logging.info(f"Writing the predictions to {output_file}")
    # Add the header to the predictions
    predictions = np.vstack((["id", "status_group"], predictions))
    np.savetxt(output_file, predictions, delimiter=",", fmt="%s")
    logging.info("Predictions written successfully")

def main():
    logging.basicConfig(level=logging.INFO)
    args = init.get_args()

    x_train, y_train, x_test = read_data(args.train_input_file, 
                                         args.train_labels_file, 
                                         args.test_input_file)


    # Save the id from the test data to append to the predictions
    x_test_id = x_test["id"]
    x_train, x_test = clean_data(x_train, x_test)

    column_transformer, functional_transformer = init.get_transformers(x_train, 
                                                     args.categorical_preprocessing,
                                                     args.numerical_preprocessing)

    best_params = get_best_hyperparams(x_train, 
                                       y_train["status_group"],
                                       args.model_type)

    # Remove the num_encoder and cat_encoder so we are not passing
    # them as parameters to the model
    best_params.pop("num_encoder")
    best_params.pop("cat_encoder")

    # Use the best hyperparameters to initialize the model
    model = init.get_model(args.model_type, **best_params)

    train_pipeline = make_pipeline(column_transformer,
                                   functional_transformer,
                                   model)

    # Train the model by 5-fold cross-validation
    print_header("TRAINING MODEL", True,)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(train_pipeline,
                             x_train,
                             y_train["status_group"],
                             cv=skf,
                             scoring='accuracy')

    # Print the mean and standard deviation of the cross-validated accuracy
    print(f"Cross-validated accuracy: {np.mean(scores)} Â± {np.std(scores)}")
    print_header("MODEL TRAINED", False)

    # Make predictions on the test data
    print_header("MAKING PREDICTIONS", True)
    train_pipeline.fit(x_train, y_train["status_group"])
    predictions = train_pipeline.predict(x_test)

    # Append id from test data to the predictions
    predictions = np.column_stack((x_test_id, predictions))
    write_predictions(args.test_prediction_output_file, predictions)
    print_header("PREDICTIONS MADE", False)

if __name__ == "__main__":
    main()
