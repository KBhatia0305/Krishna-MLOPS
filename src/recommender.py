import pickle


# ------------------- LOAD MODEL -------------------
def load_recommender(model_path="models/investor_recommender.pkl"):
    with open(model_path, "rb") as f:
        model_knn, startup_investor_matrix, investor_df = pickle.load(f)
    return model_knn, startup_investor_matrix, investor_df


# ------------------- CHECK STARTUP -------------------
def validate_startup(startup_name, startup_investor_matrix):
    """Check if startup exists in the dataset."""
    if startup_name not in startup_investor_matrix.index:
        return False
    return True


# ------------------- GET NEIGHBORS -------------------
def get_neighbors(model_knn, startup_investor_matrix, startup_name, n_recommendations):
    """Find nearest neighbor startups for a given startup."""
    startup_index = startup_investor_matrix.index.get_loc(startup_name)
    startup_data = startup_investor_matrix.iloc[startup_index, :].values.reshape(1, -1)

    distances, indices = model_knn.kneighbors(
        startup_data, n_neighbors=n_recommendations + 1
    )

    # Exclude the startup itself
    recommended_indices = indices.flatten()[1:]
    return startup_investor_matrix.index[recommended_indices]


# ------------------- GET INVESTORS -------------------
def get_investors(investor_df, startup_name, recommended_startups):
    """Get investors for recommended startups, excluding already known ones."""
    recommended_investors = set(
        investor_df[investor_df['startup'].isin(recommended_startups)]['investor'].unique()
    )

    known_investors = set(
        investor_df[investor_df['startup'] == startup_name]['investor'].unique()
    )

    return list(recommended_investors - known_investors)


# ------------------- MAIN RECOMMENDER -------------------
def recommend_investors(startup_name, model_path="models/investor_recommender.pkl", n_recommendations=5):
    model_knn, startup_investor_matrix, investor_df = load_recommender(model_path)

    # Validate startup
    if not validate_startup(startup_name, startup_investor_matrix):
        return ["Startup not found in the dataset."]

    # Find nearest startups
    recommended_startups = get_neighbors(
        model_knn, startup_investor_matrix, startup_name, n_recommendations
    )

    # Get recommended investors
    final_recommendations = get_investors(
        investor_df, startup_name, recommended_startups
    )

    return final_recommendations[:n_recommendations]


# ------------------- USAGE EXAMPLE -------------------
if __name__ == "__main__":
    recs = recommend_investors("Ola", n_recommendations=5)
    print("✅ Recommended investors:", recs)
