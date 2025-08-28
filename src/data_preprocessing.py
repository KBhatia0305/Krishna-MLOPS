import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import streamlit as st
import re
import os
import logging


# ------------------- LOGGER SETUP -------------------
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("preprocessing_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ------------------- DATA LOADING -------------------
def load_data():
    logger.info("Loading train and test data from data/raw/")
    train_df = pd.read_csv("data/raw/train.csv")
    test_df = pd.read_csv("data/raw/test.csv")
    logger.debug(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df


# ------------------- STANDARDIZE VERTICAL -------------------
def standardize_vertical(value):
    if re.search(r"e-?commerce", value, re.IGNORECASE):
        return "Ecommerce"
    return value


def apply_vertical_standardization(train_df, test_df):
    logger.info("Standardizing vertical column")
    train_df["vertical"] = train_df["vertical"].apply(standardize_vertical)
    test_df["vertical"] = test_df["vertical"].apply(standardize_vertical)
    return train_df, test_df


# ------------------- CLEAN INVESTOR NAMES -------------------
def clean_investor_names(name):
    """Clean investor names by removing unwanted characters."""
    name = re.sub(r"\\x[a-fA-F0-9]{2}", "", name)  # remove hex chars
    name = re.sub(r"’s", "", name)                 # remove possessive ’s
    name = name.replace("’", "")                   # remove stray apostrophes
    return name.strip()


def apply_investor_cleaning(train_df, test_df):
    logger.info("Cleaning investor names")
    train_df["investor"] = train_df["investor"].apply(clean_investor_names)
    test_df["investor"] = test_df["investor"].apply(clean_investor_names)
    return train_df, test_df


# ------------------- SAVE DATA -------------------
def save_processed_data(train_df, test_df):
    logger.info("Saving processed data to data/processed/")
    data_path = os.path.join("data", "processed")
    os.makedirs(data_path, exist_ok=True)

    train_path = os.path.join(data_path, "train_processed.csv")
    test_path = os.path.join(data_path, "test_processed.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.debug(f"Processed train shape: {train_df.shape}, saved to {train_path}")
    logger.debug(f"Processed test shape: {test_df.shape}, saved to {test_path}")
    logger.info("Data successfully saved.")


# ------------------- MAIN -------------------
def main():
    logger.info("Starting preprocessing pipeline")
    train_df, test_df = load_data()
    train_df, test_df = apply_vertical_standardization(train_df, test_df)
    train_df, test_df = apply_investor_cleaning(train_df, test_df)
    save_processed_data(train_df, test_df)
    logger.info("✅ Data preprocessing completed")


if __name__ == "__main__":
    main()
