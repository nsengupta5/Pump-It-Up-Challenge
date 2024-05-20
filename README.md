# Pump it Up: Data Mining the Water Table

This project predicts the functionality of water pumps across Tanzania using Scikit-Learn and Pandas. It follows best practices of the ML pipeline, including data inspection and cleaning, model training and hyperparameter tuning using Optuna. The ML models utilized as part of the proejct are as follows:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Hist Gradient Boosting Classifier
- MLPClassifier

The project achieved an accuracy of 82.36%, ranking in the top 10% of worldwide submissions for the competition. 

## Instructions

- **Data exploration** is conducted in the `data_exploration.py` file
- **Data cleaning** is conducted in the `cleaner.py` file
- **Model training** is conducted in the `part1.py` file
- **Hyperparameter tuning** is conducted in the `hpo.py` file

To run the script, navigate to the root directory and run the following command:
```
python3 part1.py <train-input-file> <train-labels-file> <test-input-file> <
numerical-preprocessing> <categorical-preprocessing> <model-type> <test-
prediction-output-file>
```

where:
- `<train-input-file>, <train-labels-file>, <test-input-file>` are the paths to the .csv data files provided by the competition.
- `<numerical-preprocessing>` represents the type of scaling method for numerical features. Valid values include: None and StandardScaler.
- `<categorical-preprocessing>` represents the type of encoding for categorical features. Valid values include: OneHotEncoder, OrdinalEncoder, and TargetEncoder.
- `<model-type>` represents the model type. Valid values include: LogisticRegression,
RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
and MLPClassifier.
- `<test-prediction-output-file>` consists of the predictions on the test dataset of the competition. This must follow the .csv submission format required by the competition.
