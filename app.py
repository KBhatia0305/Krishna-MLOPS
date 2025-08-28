# app.py
import streamlit as st
from src.recommender import recommend_investors
import pickle

def display_recommendations(recommendations):
    st.markdown("<h1 style='text-align: center;text-decoration: underline;'>Recommended Investors</h1>", unsafe_allow_html=True)
    if recommendations:
        st.markdown("<h2 style='font-size: 24px;'>Here are the top recommended investors:</h2>", unsafe_allow_html=True)
        st.markdown(
            "<ul style='list-style-type: disc; padding-left: 20px;'>"
            + "".join([f"<li style='font-size: 1.7em; color: white;'>{investor}</li>" for investor in recommendations])
            + "</ul>",
            unsafe_allow_html=True
        )
    else:
        st.write("No recommendations found for this startup.")

def display_recommendation_paragraph():
    st.markdown("""
        <div style="padding: 17px; border-radius: 10px; margin-top: 20px;">
            <p style="font-size: 1.4em; color: white;">
                These investor recommendations are tailored based on an in-depth analysis of historical funding patterns 
                and sectoral preferences. By leveraging machine learning and collaborative filtering, we have identified 
                investors who align closely with the strategic needs and growth potential of your startup. Each suggested 
                investor has demonstrated a commitment to ventures in similar industries, making them well-suited to 
                support your vision. Connecting with the right investors is a critical step toward transforming ideas into 
                impact, and we hope this curated list brings you closer to partnerships that drive innovation and success.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Streamlit UI
with open("models/investor_recommender.pkl", "rb") as f:
    model_knn, startup_investor_matrix, investor_df = pickle.load(f)

st.title("Startup Investor Recommendation System")

# Sidebar for startup selection
st.sidebar.header("Investor Recommendations")

# Get unique startup names from the matrix index
startup_list = sorted(startup_investor_matrix.index.tolist())

startup_name = st.sidebar.selectbox("Select Startup", startup_list)

if st.sidebar.button("Recommend Investors"):
    recommendations = recommend_investors(startup_name)  
    display_recommendations(recommendations)
    display_recommendation_paragraph()
