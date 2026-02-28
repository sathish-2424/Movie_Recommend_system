import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD




@st.cache_data
def load_data():
    df = pd.read_excel("movie_ratings_dataset.xlsx")
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['userId','movieId','rating','title'], inplace=True)
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    df["rating"] = df["rating"].astype(float)
    return df


df = load_data()



st.title("🎬 Movie Recommendation System")
st.write("Built with Python, Scikit-learn, Pandas & Streamlit")




movies = df[['title', 'genres', 'overview', 'director']].drop_duplicates()
movies.fillna("", inplace=True)

movies["combined_features"] = (
    movies["genres"] + " " +
    movies["overview"] + " " +
    movies["director"]
)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies["combined_features"])
cosine_sim = cosine_similarity(tfidf_matrix)


def recommend_content(movie_title, top_n=5):
    if movie_title not in movies['title'].values:
        return []

    idx = movies[movies['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores,
                               key=lambda x: x[1],
                               reverse=True)

    similar_indices = [i[0] for i in similarity_scores[1:top_n+1]]
    return movies.iloc[similar_indices]['title'].tolist()




def recommend_by_keyword(keyword, top_n=5):
    filtered = df[
        df['genres'].str.contains(keyword, case=False, na=False) |
        df['overview'].str.contains(keyword, case=False, na=False)
    ]

    movie_scores = filtered.groupby('title').agg({
        'rating': 'mean',
        'popularity': 'mean',
        'vote_count': 'mean'
    }).reset_index()

    movie_scores['score'] = (
        movie_scores['rating'] * 0.5 +
        movie_scores['popularity'] * 0.3 +
        movie_scores['vote_count'] * 0.2
    )

    movie_scores = movie_scores.sort_values(by='score',
                                            ascending=False)

    return movie_scores['title'].head(top_n).tolist()




option = st.sidebar.selectbox(
    "Choose Recommendation Type",
    ("Content-Based (Movie Name)", "Keyword / Genre Based")
)




if option == "Content-Based (Movie Name)":
    movie_name = st.selectbox("Select Movie", movies['title'].unique())

    if st.button("Recommend"):
        recommendations = recommend_content(movie_name)

        st.subheader("Recommended Movies:")
        for movie in recommendations:
            st.write("👉", movie)

elif option == "Keyword / Genre Based":
    keyword = st.text_input("Enter Genre or Keyword (Love, Action, Horror...)")

    if st.button("Recommend"):
        recommendations = recommend_by_keyword(keyword)

        st.subheader("Recommended Movies:")
        for movie in recommendations:
            st.write("👉", movie)
