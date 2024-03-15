import logging
import pandas as pd
import numpy as np
import json

def clean_data(train_data, test_data, threshold=0.95):
    print("----------------- CLEANING DATA -----------------")
    train_data, test_data = remove_irrelevant_columns(train_data, test_data)
    train_data, test_data = remove_single_value_columns(train_data, test_data)
    train_data, test_data = remove_redundant_columns(train_data, test_data)
    train_data, test_data = replace_construction_year_with_decades(train_data, test_data)
    train_data, test_data = replace_zero_longitude(train_data, test_data)
    train_data, test_data = replace_zero_gps_height(train_data, test_data)
    train_data, test_data = replace_zero_population(train_data, test_data)
    train_data, test_data = replace_zero_district_code(train_data, test_data)
    train_data, test_data = fix_formatting_errors(train_data, test_data)
    train_data, test_data = limit_high_cardinality(train_data, test_data, threshold=threshold)
    train_data, test_data = replace_missing_boolean_values(train_data, test_data)
    train_data, test_data = convert_to_datetime(train_data, test_data)
    print("----------------- DATA CLEANED -----------------\n")
    return train_data, test_data

def remove_irrelevant_columns(train_data, test_data):
    logging.info("Removing irrelevant columns")

    irrelevant_columns = [
        "id",
        "wpt_name",
        "region_code",
        "num_private",
        "scheme_name",
        "amount_tsh",
    ]

    train_data = train_data.drop(columns=irrelevant_columns)
    test_data = test_data.drop(columns=irrelevant_columns)
    logging.info("Irrelevant columns removed")
    return train_data, test_data

def remove_single_value_columns(train_data, test_data):
    logging.info("Removing single value columns")

    single_value_columns = ["recorded_by"]

    train_data = train_data.drop(columns=single_value_columns)
    test_data = test_data.drop(columns=single_value_columns)
    logging.info("Single value columns removed")
    return train_data, test_data

def convert_to_datetime(train_data, test_data):
    logging.info("Converting date columns to datetime")
    date_columns = ["date_recorded"]
    for column in date_columns:
        train_data[column] = pd.to_datetime(train_data[column])
        test_data[column] = pd.to_datetime(test_data[column])

        train_data["day_recorded"] = train_data[column].dt.day
        train_data["month_recorded"] = train_data[column].dt.month
        train_data["year_recorded"] = train_data[column].dt.year
        train_data = train_data.drop(columns=column)

        test_data["day_recorded"] = test_data[column].dt.day
        test_data["month_recorded"] = test_data[column].dt.month
        test_data["year_recorded"] = test_data[column].dt.year
        test_data = test_data.drop(columns=column)
    logging.info("Date columns converted to datetime")
    return train_data, test_data

def remove_redundant_columns(train_data, test_data):
    logging.info("Removing redundant columns")

    redundant_columns = [
        "waterpoint_type_group",
        "source",
        "water_quality",
        "scheme_management",
        "extraction_type",
        "payment",
        "quantity",
    ]

    train_data = train_data.drop(columns=redundant_columns)
    test_data = test_data.drop(columns=redundant_columns)
    logging.info("Redundant columns removed")
    return train_data, test_data

def replace_construction_year_with_decades(train_data, test_data):
    logging.info("Replacing construction year with decades")

    def bin_decades(year):
        if year == 0:
            return "Unknown"
        return f"{year // 10 * 10}s"

    train_data["decade"] = train_data["construction_year"].apply(bin_decades)
    test_data["decade"] = test_data["construction_year"].apply(bin_decades)

    train_data = train_data.drop(columns="construction_year")
    test_data = test_data.drop(columns="construction_year")

    logging.info("Construction year replaced with decades")
    return train_data, test_data

def replace_zero_longitude(train_data, test_data):
    logging.info("Replacing zero longitude")

    # Replace zero longitude with the mean longitude of the region
    train_data["longitude"] = train_data["longitude"].replace(0, pd.NA)
    test_data["longitude"] = test_data["longitude"].replace(0, pd.NA)

    train_region_means = train_data.groupby("region")["longitude"].mean()

    def impute_longitude(row):
        if pd.isna(row["longitude"]):
            return train_region_means[row["region"]]  
        return row["longitude"]

    train_data["longitude"] = train_data.apply(impute_longitude, axis=1)
    test_data["longitude"] = test_data.apply(impute_longitude, axis=1)

    logging.info("Zero longitude replaced")
    return train_data, test_data

def replace_zero_gps_height(train_data, test_data):
    logging.info("Replacing zero gps_height")

    train_region_means = train_data.groupby("region")["gps_height"].mean()
    na_means = ["Dodoma", "Kagera", "Mbeya", "Tabora"]

    def impute_gps_height(row):
        if row["gps_height"] == 0:
            if row["region"] in na_means:
                return get_neighbour_mean(row["region"], train_region_means)
            return train_region_means[row["region"]]  
        return row["gps_height"]

    train_data["gps_height"] = train_data.apply(impute_gps_height, axis=1)
    test_data["gps_height"] = test_data.apply(impute_gps_height, axis=1)

    logging.info("Zero gps_height replaced")
    return train_data, test_data

def get_neighbour_mean(region, region_means):
    neighbours = []
    if region == "Dodoma":
        neighbours = ["Iringa", "Manyara", "Singida", 'Morogoro']
    elif region == "Kagera":
        neighbours = ["Kigoma", "Mwanza"]
    elif region == "Mbeya":
        neighbours = ["Iringa", "Singida", "Rukwa"]
    elif region == "Tabora":
        neighbours = ["Shinyanga", "Singida", "Kigoma"]
    else:
        return pd.NA

    # Get the mean of the neighbours
    neighbour_means = region_means[neighbours]
    return neighbour_means.mean()

def replace_zero_population(train_data, test_data):
    logging.info("Replacing zero population")

    train_region_means = train_data.groupby("region")["population"].mean()
    na_means = ["Dodoma", "Kagera", "Mbeya", "Tabora"]

    def impute_population(row):
        if row["population"] == 0:
            if row["region"] in na_means:
                return get_neighbour_mean(row["region"], train_region_means)
            return train_region_means[row["region"]]  
        return row["population"]

    train_data["population"] = train_data.apply(impute_population, axis=1)
    test_data["population"] = test_data.apply(impute_population, axis=1)

    logging.info("Zero population replaced")
    return train_data, test_data

def replace_zero_district_code(train_data, test_data):
    logging.info("Replacing zero district_code")

    train_region_means = train_data.groupby("region")["district_code"].median()

    def impute_district_code(row):
        if row["district_code"] == 0:
            return train_region_means[row["region"]]  
        return row["district_code"]

    train_data["district_code"] = train_data.apply(impute_district_code, axis=1)
    test_data["district_code"] = test_data.apply(impute_district_code, axis=1)

    logging.info("Zero district_code replaced")
    return train_data, test_data

def fix_formatting_errors(train_data, test_data):
    logging.info("Fixing formatting errors")

    incorrect_format_columns = ["funder", "installer"]

    # Load the mapping files
    for col in incorrect_format_columns:
        with open(f"data/spellingData/clusters_{col}.json", "r") as f:
            mapping = json.load(f)

        def replace_values(row, mapping_dict):
            value = str(row).lower().strip()
            for k, v in mapping_dict.items():
                if value in v:
                    return k
            return value

        train_data[col] = train_data[col].apply(replace_values, mapping_dict=mapping)
        test_data[col] = test_data[col].apply(replace_values, mapping_dict=mapping)

    logging.info("Formatting errors fixed")
    return train_data, test_data

def limit_high_cardinality(train_data, test_data, threshold):
    logging.info("Limiting high cardinality")

    high_cardinality_columns = ["funder", "installer", "subvillage", "ward"]

    for col in high_cardinality_columns:

        # Find the top categories that cover the 'threshold' percentage of data points
        top_categories = train_data[col].value_counts(normalize=True).cumsum()
        top_categories = top_categories[top_categories <= threshold].index.tolist()
        
        # Update the train data to only have the top categories and 'Other'
        train_data[col] = np.where(train_data[col].isin(top_categories), train_data[col], 'Other')
        
        # Update the test data using the same top categories identified from the train data
        test_data[col] = np.where(test_data[col].isin(top_categories), test_data[col], 'Other')

        # Replace missing values with 'Unknown'
        train_data[col] = train_data[col].fillna("Unknown")
        test_data[col] = test_data[col].fillna("Unknown")

    logging.info("High cardinality limited")
    return train_data, test_data

def replace_missing_boolean_values(train_data, test_data):
    logging.info("Replacing missing boolean values")
    pd.set_option('future.no_silent_downcasting', True)
    boolean_columns = ["public_meeting", "permit"]

    # True is the most frequent value in the data
    train_data[boolean_columns] = train_data[boolean_columns].fillna(True)
    test_data[boolean_columns] = test_data[boolean_columns].fillna(True)

    logging.info("Missing boolean values replaced")
    return train_data, test_data
