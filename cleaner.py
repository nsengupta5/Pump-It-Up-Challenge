import logging
import pandas as pd

def clean_data(train_data, test_data):
    logging.info("Preprocessing data")
    train_data, test_data = remove_irrelevant_columns(train_data, test_data)
    train_data, test_data = remove_single_value_columns(train_data, test_data)
    train_data, test_data = convert_to_datetime(train_data, test_data)
    logging.info("Data preprocessed")
    return train_data, test_data

def remove_irrelevant_columns(train_data, test_data):
    logging.info("Removing irrelevant columns")
    columns_to_remove = []

    irrelevant_columns = [
        "id",
        "wpt_name",
    ]

    duplicate_columns = [
        "quantity_group",
        "payment_type",
    ]

    columns_to_remove.extend(irrelevant_columns)
    columns_to_remove.extend(duplicate_columns)

    train_data = train_data.drop(columns=columns_to_remove)
    test_data = test_data.drop(columns=columns_to_remove)
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
