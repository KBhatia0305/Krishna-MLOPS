import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors
import os
import yaml
import logging


# ------------------- LOGGER SETUP -------------------
logger = logging.getLogger("recommender_training")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("recommender.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ------------------- LOAD CONFIG -------------------
def load_params(config_path="params.yaml"):
    """Load parameters for recommendation training from YAML config."""
    logger.info(f"Loading parameters from {config_path}")
    params = yaml.safe_load(open(config_path, 'r'))['recommendation_training']
    logger.debug(f"Parameters loaded: {params}")
    return params


# Load and combine train and test datasets
def load_data(train_path, test_path):  
    logger.info(f"Loading datasets: {train_path}, {test_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    df = pd.concat([train_df, test_df], ignore_index=True)
    logger.debug(f"Combined dataset shape: {df.shape}")
    return df


# Prepare investor dataframe and startup-investor matrix
def prepare_investor_matrix(df):
    logger.info("Preparing investor dataframe and startup-investor matrix")
    investor_df = df.assign(investor=df['investor'].str.split(',')).explode('investor')
    investor_df['investor'] = investor_df['investor'].str.strip()

    startup_investor_matrix = investor_df.pivot_table(
        index='startup',
        columns='investor',
        values='amount',
        fill_value=0
    )

    logger.debug(f"Investor dataframe shape: {investor_df.shape}")
    logger.debug(f"Startup-investor matrix shape: {startup_investor_matrix.shape}")
    return investor_df, startup_investor_matrix


# Train KNN recommender model
def train_knn(startup_investor_matrix, params):
    logger.info("Training KNN recommender model")
    model_knn = NearestNeighbors(metric=params['metric'], algorithm=params['algorithm'])
    model_knn.fit(startup_investor_matrix)
    logger.debug("KNN model training completed")
    return model_knn


# Save trained model and data to disk
def save_model(model_knn, startup_investor_matrix, investor_df, model_path):
    logger.info(f"Saving recommender model to {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump((model_knn, startup_investor_matrix, investor_df), f)
    logger.info(f"✅ Recommender model saved to {model_path}")


# ------------------- MAIN -------------------
def build_recommender(
    train_path="data/processed/train_processed.csv",
    test_path="data/processed/test_processed.csv",
    model_path="models/investor_recommender.pkl"
):
    logger.info("Starting recommender system build process")
    params = load_params()
    df = load_data(train_path, test_path)
    investor_df, startup_investor_matrix = prepare_investor_matrix(df)
    model_knn = train_knn(startup_investor_matrix, params)
    save_model(model_knn, startup_investor_matrix, investor_df, model_path)
    logger.info("Recommender system build process completed")


if __name__ == "__main__":
    build_recommender()
