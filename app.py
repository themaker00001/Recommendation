import streamlit as st
import pandas as pd
import numpy as np
import ast
import faiss
import os
import requests
from sentence_transformers import SentenceTransformer
import difflib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")

if not TMDB_API_KEY:
    st.error("❌ TMDB API key not found. Please set TMDB_API_KEY in your .env file.")
    st.stop()

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("data/tmdb_5000_movies.csv")
    df['genre_text'] = df['genres'].apply(lambda x: " ".join([d['name'] for d in ast.literal_eval(x)]))
    df['overview'] = df['overview'].fillna("")
    df['text_for_embedding'] = df['title'] + " " + df['genre_text'] + " " + df['overview']
    df = df[df['genre_text'].str.strip().astype(bool)].reset_index(drop=True)
    return df

df = load_data()

# Build FAISS index
@st.cache_resource
def build_model_and_index(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts.tolist(), normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return model, index, embeddings

model, index, embeddings = build_model_and_index(df['text_for_embedding'])

# Poster fetch from TMDB
def fetch_poster(title, api_key=TMDB_API_KEY):
    url = f"https://api.themoviedb.org/3/search/movie"
    params = {"api_key": api_key, "query": title}
    try:
        res = requests.get(url, params=params).json()
        if res['results']:
            poster_path = res['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass
    return None

# Recommendation: Hybrid
def recommend_by_text(query_text, top_k=10):
    query_vec = model.encode([query_text], normalize_embeddings=True)
    D, I = index.search(query_vec, top_k * 2)
    return df.iloc[I[0]]

# Recommendation: Fuzzy by Title
def recommend_by_title(input_title, top_k=10):
    titles = df['title'].tolist()
    best_match = difflib.get_close_matches(input_title, titles, n=1, cutoff=0.5)
    if not best_match:
        return None, None
    match = best_match[0]
    idx = df[df['title'] == match].index[0]
    query_vec = embeddings[idx].reshape(1, -1)
    D, I = index.search(query_vec, top_k + 5)
    I = [i for i in I[0] if i != idx]
    return match, df.iloc[I]

# Filter logic
def filter_results(df_result, min_rating, min_votes, release_range):
    df_result = df_result[
        (df_result['vote_average'] >= min_rating) &
        (df_result['vote_count'] >= min_votes)
    ]
    df_result = df_result[
        df_result['release_date'].str[:4].fillna("0").astype(int).between(*release_range)
    ]
    return df_result

# Sidebar Controls
st.sidebar.title("🎛 Recommender Settings")
search_type = st.sidebar.radio("Search by:", ["Movie Title", "Genre/Keywords"])
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 10.0, 6.0)
release_range = st.sidebar.slider("Release Year", 1950, 2025, (2000, 2022))
min_votes = st.sidebar.slider("Minimum Vote Count", 0, 5000, 100)
top_k = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

# Main UI
st.title("🎬 Movie Recommender System")

if search_type == "Movie Title":
    movie_input = st.text_input("Enter Movie Title (e.g. Inception):")
    if movie_input:
        with st.spinner("Finding similar movies..."):
            match, result_df = recommend_by_title(movie_input, top_k=top_k)
            if result_df is not None:
                filtered = filter_results(result_df, min_rating, min_votes, release_range).head(top_k)
                st.subheader(f"🎯 Top {len(filtered)} Movies Similar to: *{match}*")
                for _, row in filtered.iterrows():
                    cols = st.columns([1, 4])
                    with cols[0]:
                        img_url = fetch_poster(row['title'])
                        if img_url:
                            st.image(img_url, width=120)
                    with cols[1]:
                        st.markdown(f"### {row['title']} ({row['release_date'][:4]})")
                        st.markdown(f"⭐ {row['vote_average']} | 📊 {row['vote_count']} votes")
                        st.markdown(f"🗂️ *{row['genre_text']}*")
                        st.markdown(f"_📝 {row['overview'][:300]}..._")
                        st.markdown("---")
            else:
                st.error("❌ Movie not found. Try another title.")

elif search_type == "Genre/Keywords":
    query_input = st.text_input("Enter Genre or Keywords (e.g. Action Sci-Fi love):")
    if query_input:
        with st.spinner("Finding similar movies..."):
            result_df = recommend_by_text(query_input, top_k=top_k)
            filtered = filter_results(result_df, min_rating, min_votes, release_range).head(top_k)
            st.subheader(f"🔎 Top {len(filtered)} Movies Matching: *{query_input}*")
            for _, row in filtered.iterrows():
                cols = st.columns([1, 4])
                with cols[0]:
                    img_url = fetch_poster(row['title'])
                    if img_url:
                        st.image(img_url, width=120)
                with cols[1]:
                    st.markdown(f"### {row['title']} ({row['release_date'][:4]})")
                    st.markdown(f"⭐ {row['vote_average']} | 📊 {row['vote_count']} votes")
                    st.markdown(f"🗂️ *{row['genre_text']}*")
                    st.markdown(f"_📝 {row['overview'][:300]}..._")
                    st.markdown("---")
