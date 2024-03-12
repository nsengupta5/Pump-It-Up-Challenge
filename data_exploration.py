import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from fuzzywuzzy import process
import json
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.svm import SVR
from sklearn.feature_selection import RFE


SIMILARITY_THRESHOLD = 95

def explore_data(features, labels):
    # explore_labels(labels)
    explore_basic_stats(features)

    cat_features = features.select_dtypes(include=[object])
    # num_features = features.select_dtypes(include=[np.number])
    # explore_categories(cat_features)
    # explore_numerical_data(num_features)
    # explore_feature_importance_numerical(num_features, labels)
    # explore_geographical_data(features, labels)
    explore_categorical_features(cat_features)
    # explore_feature_importance_categories(cat_features, labels)


def explore_basic_stats(features):
    print("------- Exploring Basic Stats -------")
    # Print the basic stats for the numerical features
    print(features.head())
    print(features.info())
    print(features.describe())

def explore_labels(labels):
    print("------- Exploring Value Counts for Labels -------")
    # Count the aggregate number for each label
    label_counts = labels["status_group"].value_counts()
    print(label_counts)
    print()

def explore_high_cardinality_categories(cat_features):
    # print("------- Exploring Categories -------")
    # Set numpy to print the full array
    np.set_printoptions(threshold=np.inf)

    high_cardinality_categories = ["funder", "installer", "subvillage", "ward", "scheme_name"]

    # Print only the names of the first 100 most frequent funders
    funders = cat_features["installer"].value_counts().head(1000).index.to_list()

    # Sort the funders
    funders_clean = [str(f).lower().strip() for f in funders]

    # Identify potential duplicates using fuzzy matching
    clusters = {}
    for funder in funders_clean:
        if funder == 'nan':
            continue

        matches = process.extract(funder, funders_clean, limit=2)
        for match, score in matches:
            if score >= SIMILARITY_THRESHOLD and funder != match:
                clusters.setdefault(match, set()).add(funder)

    clusters_json_ready = {k: list(v) for k, v in clusters.items()}
    print(json.dumps(clusters_json_ready, indent=4))


def explore_categorical_features(cat_features):
    print("------- Exploring Value Counts for Categories -------")
    # Print the value counts for each categorical feature
    for col in cat_features.columns:
        print(f"------- Exploring Value Counts for {col} -------")
        print(cat_features[col].value_counts())
        
        # Print number of categories in each column
        print(f"Number of categories in {col}: {cat_features[col].nunique()}")

        # Print number of missing values in each column
        print(f"Number of missing values in {col}: {cat_features[cat_features[col].isnull()].shape[0]}")
        print()

def explore_numerical_data(num_features):
    print("------- Exploring Numerical Data -------")
    # Print the basic stats for the numerical features
    for col in num_features.columns:
        print(f"------- Exploring Stats for {col} -------")
        df = num_features[col][num_features[col] != 0]
        print(df.describe())
        
        # Print number of zeros in each column
        print(f"Number of zeros in {col}: {num_features[num_features[col] == 0].shape[0]}")
        print()

    num_features = num_features[num_features != 0]
    num_features.hist(bins=50, figsize=(20, 15))
    plt.show()

def explore_geographical_data(features, labels):
    print("------- Exploring Geographical Data -------")
    # Print the basic stats for the numerical features
    df = features[["id", "longitude", "latitude"]]
    df = df[df["longitude"] != 0]
    df = df[df["latitude"] != 0]

    # Join labels on the id
    df = df.join(labels.set_index("id"), on="id")

     # Create a figure and axis for the plot
    _, ax = plt.subplots()

    # Plot each status group with a different color
    for status, color in zip(df['status_group'].unique(), ['yellow', 'purple', 'green']):
        df_status = df[df['status_group'] == status]
        ax.scatter(df_status['longitude'], df_status['latitude'], c=color, label=status, alpha=0.4)

    ax.legend()
    plt.show()

def explore_correlations(features, labels):
    print("------- Exploring Correlations -------")

def explore_feature_importance_categories(cat_features, labels):
    print("------- Exploring Feature Importance for Categories -------")
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select=10, step=1)
    
    cat_features_encoded = OneHotEncoder().fit_transform(cat_features)
    labels_encoded = LabelEncoder().fit_transform(labels["status_group"])


    selector = selector.fit(cat_features_encoded, labels_encoded)
    print(selector.support_)
    print(selector.ranking_)


def explore_feature_importance_numerical(num_features, labels):
    print("------- Exploring Feature Importance for Numerical Data -------")

    select_k_best = SelectKBest(score_func=f_oneway, k=10)

    select_k_best.fit(num_features, labels["status_group"])
    scores = select_k_best.scores_
    print(scores)
