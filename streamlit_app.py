import streamlit as st
import pandas as pd
import numpy as np
from surprise import SVD
from surprise import Dataset, Reader
from sklearn.metrics.pairwise import cosine_similarity

st.title('🎥 Movie Recommender App')
st.write('App is starting...')

# Cache data loading
@st.cache_data
def load_movies_df():
    return pd.read_csv('movies_df.csv')

@st.cache_data
def load_ratings_df():
    return pd.read_csv('ratings_df.csv')

# Cache model training
@st.cache_resource
def train_model(ratings_df):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[['userId','movieId','rating']], reader)
    trainset = data.build_full_trainset()
    model = SVD(n_factors=50, random_state=42)
    model.fit(trainset)
    return model

# Cache similarity matrix
@st.cache_resource
def compute_similarity(movies_df):
    genres_df = movies_df['genres'].str.get_dummies('|')
    return cosine_similarity(genres_df)

# Load data & model
st.write('Loading data...')
movies_df = load_movies_df()
ratings_df = load_ratings_df()
model = train_model(ratings_df)
similarity_matrix = compute_similarity(movies_df)

st.success('✅ Ready to recommend movies!')

# Movie search
selected_title = st.selectbox(
    "Pick a movie you like:",
    movies_df['title'].tolist()
)

# Recommend similar movies
if st.button('Get Similar Movies'):
    idx = movies_df[movies_df['title']==selected_title].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    top_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    recs = movies_df.iloc[[i for i,s in top_scores]]['title'].tolist()
    st.write('📜 Similar movies:', recs)

# User rating interface
st.markdown('---')
st.header('Rate a movie and get personalized recommendations')
user_rating_movie = st.selectbox('Pick a movie to rate:', movies_df['title'].tolist())
user_rating = st.slider('Your Rating (0.5-5.0)', min_value=0.5, max_value=5.0, step=0.5)

if st.button('Get Personalized Recommendations'):
    # Predict for all movies
    all_ids = movies_df['movieId'].tolist()
    preds = [model.predict(uid=99999, iid=mid).est for mid in all_ids]  # Dummy new user
    top_movies_idx = np.argsort(preds)[::-1][:5]
    top_movies = movies_df.iloc[top_movies_idx]['title'].tolist()
    st.success('Your Personalized Recs:')
    st.write(top_movies)
