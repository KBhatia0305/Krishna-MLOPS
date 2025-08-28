import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import streamlit as st
import re
import os
import yaml
import logging

st.set_page_config(layout='wide', page_title='Startup Analysis')

# Setup File Logging and Console Logging
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_config():
    logger.info("Loading configuration from params.yaml")
    return yaml.safe_load(open('params.yaml', 'r'))['data_ingestion']['test_size']


def load_data(file_path="data/startup_cleaned12.csv"):
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)


def preprocess_data(df):
    logger.info("Preprocessing data")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    city_replacements = {
        'Bengaluru': 'Bangalore',
        'Gurugram': 'Gurgaon',
        'Ahmedabad': 'Ahemdabad',
        'Ahemadabad': 'Ahemdabad',
        'Bhubneswar': 'Bhubaneswar'
    }

    df['city'] = df['city'].replace(city_replacements)
    df = df.dropna(subset=['investor'])
    return df


def split_data(df, test_size):
    logger.info(f"Splitting data into train and test with test_size={test_size}")
    return train_test_split(df, test_size=test_size, random_state=42)


def save_data(train_data, test_data):
    logger.info("Saving train and test data to data/raw/")
    data_path = os.path.join("data", "raw")
    os.makedirs(data_path, exist_ok=True)
    train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
    test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
    logger.info("Data successfully saved.")


def main():
    test_size = load_config()
    df = load_data()
    df = preprocess_data(df)
    train_data, test_data = split_data(df, test_size)
    save_data(train_data, test_data)


if __name__ == "__main__":
    main()
