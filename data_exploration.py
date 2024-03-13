import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, chi2_contingency
from fuzzywuzzy import process
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.svm import SVR
from sklearn.feature_selection import RFE

SIMILARITY_THRESHOLD = 95

def explore_data(features, labels):
    print("----------------- EXPLORING DATA -----------------")
    # explore_basic_stats(features, labels)
    # explore_categorical_features(features, labels)
    explore_numerical_features(features, labels)
    print("----------------- DATA EXPLORATION COMPLETE -----------------\n")

def explore_categorical_features(features, labels):
    cat_features = features.select_dtypes(include=[object])
    explore_categorical_stats(cat_features)
    # explore_high_cardinality_categories(cat_features)
    # explore_feature_importance_categories(cat_features, labels)
    # explore_subset_features(cat_features, labels)

def explore_numerical_features(features, labels):
    num_features = features.select_dtypes(include=[np.number])
    explore_numerical_stats(num_features)
    explore_feature_importance_numerical(num_features, labels)
    explore_geographical_data(features, labels)

def explore_basic_stats(features, labels):
    print("------- Exploring Basic Stats -------")
    # Print the basic stats for the numerical features
    print(features.head())
    print(features.info())
    print(features.describe())
    print("------- Percentage of Missing Values -------")
    missing_percent = features.isnull().mean() * 100
    print(missing_percent)
    explore_labels(labels)

def explore_labels(labels):
    print("------- Exploring Value Counts for Labels -------")
    # Count the aggregate number for each label
    label_counts = labels["status_group"].value_counts()
    print(label_counts)
    print()

def explore_high_cardinality_categories(cat_features):
    # print("------- Exploring Categories -------")
    np.set_printoptions(threshold=np.inf)

    high_cardinality_categories = ["funder", "installer"]

    for col in high_cardinality_categories:
        # Print only the names of the first 100 most frequent funders
        df = cat_features[col].value_counts().head(1000).index.to_list()

        # Sort the funders
        funders_clean = [str(f).lower().strip() for f in df]

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
        with open(f"data/spellingData/clusters_{col}.json", "w") as f:
            json.dump(clusters_json_ready, f)

def explore_categorical_stats(cat_features):
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

def explore_numerical_stats(num_features):
    print("------- Exploring Numerical Data -------")
    # Print the basic stats for the numerical features
    for col in num_features.columns:
        print(f"------- Exploring Stats for {col} -------")
        df = num_features[col][num_features[col] != 0]
        print(df.describe())
        
        # Print number of zeros in each column
        print(f"Number of zeros in {col}: {num_features[num_features[col] == 0].shape[0]}")
        print()

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
    plt.title("Scatter Plot of Longitude and Latitude")
    plt.savefig("plots/longitude-vs-latitude.png")

    plt.clf()

    full_train = pd.merge(features, labels, on="id")

    # Plot geographic features with status code
    sns.countplot(x="lga", hue="status_group", data=full_train)
    plt.title("lga vs Status Group")
    plt.savefig("plots/lga-vs-status_group.png")
    plt.clf()

    sns.countplot(x="region", hue="status_group", data=full_train)
    plt.title("region vs Status Group")
    plt.savefig("plots/region-vs-status_group.png")
    plt.clf()

    sns.countplot(x="district_code", hue="status_group", data=full_train)
    plt.title("district_code vs Status Group")
    plt.savefig("plots/district_code-vs-status_group.png")
    plt.clf()

    sns.countplot(x="ward", hue="status_group", data=full_train)
    plt.title("ward vs Status Group")
    plt.savefig("plots/ward-vs-status_group.png")
    plt.clf()

    show_corr(full_train, "lga", "region")
    show_corr(full_train, "ward", "region")
    show_corr(full_train, "ward", "lga")
    show_corr(full_train, "district_code", "region")

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

def explore_subset_features(features, labels):
    print("------- Exploring Subset Features -------")
    df = pd.concat([features, labels], axis=1)

    mappings = {
        "extraction_type_class": ["extraction_type", "extraction_type_group"],
        "management_group": ["management"],
        "quality_group": ["water_quality"],
        "source_class": ["source", "source_type"],
        "waterpoint_type_group": ["waterpoint_type"]
    }

    for general, specific in mappings.items():
        print(f"------- Exploring Subset Features for {general} -------")
        print(df[general].value_counts())
        sns.countplot(x=general, hue="status_group", data=df)
        plt.title(f"{general} vs Status Group")
        plt.savefig(f"plots/{general}-vs-status_group.png")
        plt.clf()
        print()
        for feature in specific:
            print(df[feature].value_counts())
            sns.countplot(x=feature, hue="status_group", data=df)
            plt.title(f"{feature} vs Status Group")
            plt.savefig(f"plots/{feature}-vs-status_group.png")
            plt.clf()
            show_corr(df, feature, general)

        print()

def show_corr(df, feature, general):
    print(f"------- Exploring Correlation for {feature} and {general} -------")
    contingency_table = pd.crosstab(df[feature], df[general])
    chi2, p, dof, _ = chi2_contingency(contingency_table)
    print(f"Chi2: {chi2}, p: {p:.5f}, dof: {dof}")
