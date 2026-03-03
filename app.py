import pandas as pd
import numpy as np
import sqlite3
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix


# CONFIG

st.set_page_config(page_title="CineInsight Engine", layout="wide")


# LOAD DATA

@st.cache_data
def load_data():
    df = pd.read_csv("movie_ratings_dataset.csv")
    
    df.drop_duplicates(inplace=True)
    df['overview'] = df['overview'].fillna('')
    
    movies_df = df.drop_duplicates(subset=['title']).reset_index(drop=True)
    
    genre_counts = (
        df['genres']
        .str.split('|')
        .explode()
        .value_counts()
        .head(10)
    )
    
    return df, movies_df, genre_counts

try:
    df, movies_df, genre_counts = load_data()
except FileNotFoundError:
    st.error("Dataset not found.")
    st.stop()


# DATABASE

@st.cache_resource
def init_db(data):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    data.to_sql("movie_ratings", conn, index=False, if_exists="replace")
    return conn

conn = init_db(df)

def run_cohort_analysis():
    query = """
    SELECT userId,
           COUNT(movieId) as rating_count,
           AVG(rating) as avg_rating
    FROM movie_ratings
    GROUP BY userId
    HAVING COUNT(movieId) >= 15
    ORDER BY rating_count DESC
    """
    return pd.read_sql(query, conn)


# CONTENT MODEL

@st.cache_resource
def train_content_model(data):
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = tfidf.fit_transform(data["overview"])
    similarity = linear_kernel(matrix, matrix)
    
    indices = pd.Series(data.index, index=data["title"]).drop_duplicates()
    return similarity, indices

cosine_sim, title_index = train_content_model(movies_df)

def get_content_recs(title, top_n=5):
    if title not in title_index:
        return None
    
    idx = title_index[title]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    movie_indices = [i[0] for i in scores]
    return movies_df.iloc[movie_indices]


@st.cache_resource
def train_collab_model(data):
    user_movie = data.pivot_table(
        index="userId",
        columns="title",
        values="rating"
    ).fillna(0)

    sparse_matrix = csr_matrix(user_movie.values)
    
    svd = TruncatedSVD(n_components=12, random_state=42)
    reduced = svd.fit_transform(sparse_matrix.T)
    
    return reduced, user_movie.columns

svd_matrix, movie_titles = train_collab_model(df)

def get_collab_recs(title, top_n=5):
    if title not in movie_titles:
        return []
    
    idx = list(movie_titles).index(title)
    sim_scores = cosine_similarity(
        [svd_matrix[idx]],
        svd_matrix
    )[0]
    
    top_indices = np.argsort(sim_scores)[::-1][1:top_n+1]
    return movie_titles[top_indices]


st.title("🎥 Movie Recommendation Dashboard")

col1, col2 = st.columns(2)
col1.metric("Average Rating", f"{df['rating'].mean():.2f}")
col2.metric("Unique Movies", df['movieId'].nunique())

st.subheader("Top Genres")
st.bar_chart(genre_counts)

st.divider()

selected_movie = st.selectbox(
    "Select a Movie",
    movies_df["title"]
)

if st.button("Generate Recommendations"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Content-Based")
        recs = get_content_recs(selected_movie)
        if recs is not None:
            for _, row in recs.iterrows():
                st.write(f"**{row['title']}**")
        else:
            st.warning("Movie not found.")

    with col2:
        st.subheader("Collaborative (SVD)")
        recs = get_collab_recs(selected_movie)
        for movie in recs:
            st.write(f"**{movie}**")
