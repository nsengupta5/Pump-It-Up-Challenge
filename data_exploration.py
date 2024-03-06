import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def explore_data(features, labels):
    explore_labels(labels)

    explore_basic_stats(features)

    cat_features = features.select_dtypes(include=[object])
    num_features = features.select_dtypes(include=[np.number])
    # explore_categorical_features(cat_features)
    # explore_region_code(features)
    test(features)

def explore_basic_stats(features):
    num_rows, num_cols = features.shape
    print(f"Number of samples: {num_rows}")
    print(f"Number of features: {num_cols}")

def explore_labels(labels):
    print("------- Exploring Value Counts for Labels -------")
    # Count the aggregate number for each label
    label_counts = labels["status_group"].value_counts()
    print(label_counts)
    print()

def explore_categorical_features(cat_features):
    for col in cat_features.columns:
        print(f"------- Exploring Value Counts for {col} -------")
        # Count the aggregate number for each label
        label_counts = cat_features[col].value_counts()
        print(label_counts)
        print()

        # Print the number of nan values
        nan_count = cat_features[col].isna().sum()
        print(f"Number of NaN values: {nan_count}")
        print()



def explore_region_code(features):
    print("------- Exploring Value Counts for region_code -------")
    # Within a region, look at what districts are in that region
    districts_in_regions = features.groupby('region')['district_code'].unique().reset_index()
    print(districts_in_regions)

    tmp = features.groupby('region_code')['district_code'].unique().reset_index()


    region_mappings = features.groupby(['region_code', 'region']).size().reset_index(name='count')
    # sort by region 
    region_mappings = region_mappings.sort_values(by=['region'])
    print(tmp)

    print(region_mappings)

    tmp2 = features.groupby('region_code')['region'].unique().reset_index()
    print(tmp2)

def test(features):
    construction_year = features['construction_year']
    recorded_year = features['date_recorded']

    # Create new dataframe with above two columns
    df = pd.DataFrame({'construction_year': construction_year, 'recorded_year': recorded_year})
    print(df)

    year = features['date_recorded'].str.split('-').str[0]
    print(year)

    construction_year = features['construction_year'][features['construction_year'] != 0]
    year = features['date_recorded'].str.split('-').str[0][features['construction_year'] != 0]

    # plot the data
    plt.scatter(construction_year, year)
    plt.show()


