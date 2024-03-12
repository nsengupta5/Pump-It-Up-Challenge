import pandas as pd
import logging
import initializer as init
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
    logging.info(f"Reading the data from {input_file}, {labels_file}, and {test_file}")
    
    x_train = pd.read_csv(input_file)
    y_train = pd.read_csv(labels_file)
    x_test = pd.read_csv(test_file)

    logging.info("Data read successfully")
    return x_train, y_train, x_test

"""
Write the predictions to the output file following the submission format

args:
    output_file: The file to save the test predictions
    predictions: The predictions to save
"""
def write_predictions(output_file, predictions):
    logging.info(f"Writing the predictions to {output_file}")
    predictions.to_csv(output_file, index=False)
    logging.info("Predictions written successfully")

def main():
    logging.basicConfig(level=logging.INFO)
    args = init.get_args()

    x_train, y_train, x_test = read_data(args.train_input_file, 
                                         args.train_labels_file, 
                                         args.test_input_file)

    explore_data(x_train, y_train)
    # x_train, x_test = clean_data(x_train, x_test)

    # column_transformer = init.get_column_transformer(x_train, 
    #                                                  args.categorical_preprocessing,
    #                                                  args.numerical_preprocessing)

    # model = init.get_model(args.model_type)
    # pipeline = make_pipeline(column_transformer, model)

    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    # scores = cross_val_score(pipeline, x_train, y_train["status_group"], cv=skf, scoring='accuracy')
    # print(f"Cross-validated accuracy: {np.mean(scores)} Â± {np.std(scores)}")

if __name__ == "__main__":
    main()
