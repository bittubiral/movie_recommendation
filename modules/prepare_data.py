"""
prepare_data.py
----------------
Builds the artifacts the Streamlit app needs:
  - movie_data.csv   -> one row per movie: title, genres, tags, avg_rating, rating_count
  - tfidf_matrix.pkl -> TF-IDF matrix whose ROW ORDER matches movie_data.csv exactly

Run this once (or whenever the CSVs change):
    python prepare_data.py

NOTE ON A BUG IN THE ORIGINAL movie.py
---------------------------------------
The original app loaded `movies.csv` directly and indexed straight into
`tfidf_matrix` using positions from `movies.csv`. But the notebook actually
built `tfidf_matrix` from `movie_rating`, which is `movies` merged with
`ratings` AND `tags` -- a much longer, duplicated-per-rating dataframe whose
row order does NOT match `movies.csv`. So `tfidf_matrix[idx]` in the app was
pulling the vector for the wrong movie almost every time.

This script fixes that by building one de-duplicated row per movie and
saving the TF-IDF matrix in that exact same order, so `movie_data.csv` row i
always corresponds to `tfidf_matrix[i]`.
"""

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

print("Loading CSVs...")
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
tags = pd.read_csv("tags.csv")

# --- Rating stats per movie (computed on the FULL ratings table, not a merge) ---
rating_stats = (
    ratings.groupby("movieId")["rating"]
    .agg(avg_rating="mean", rating_count="count")
    .reset_index()
)

# --- Tags per movie: combine every tag into one string ---
tags_grouped = (
    tags.dropna(subset=["tag"])
    .groupby("movieId")["tag"]
    .apply(lambda t: " ".join(t.astype(str)))
    .reset_index()
    .rename(columns={"tag": "tags"})
)

# --- One row per movie, everything joined on movieId ---
movie_data = movies.merge(rating_stats, on="movieId", how="left")
movie_data = movie_data.merge(tags_grouped, on="movieId", how="left")
movie_data["tags"] = movie_data["tags"].fillna("")
movie_data["avg_rating"] = movie_data["avg_rating"].fillna(0).round(2)
movie_data["rating_count"] = movie_data["rating_count"].fillna(0).astype(int)

# genres like "Adventure|Animation|Comedy" -> "Adventure Animation Comedy"
movie_data["genres_clean"] = movie_data["genres"].str.replace("|", " ", regex=False)
movie_data["genres_clean"] = movie_data["genres_clean"].replace(
    "(no genres listed)", ""
)

# Content field used for similarity: genres + tags (weight genres x2 so
# genre match matters at least as much as free-text tags)
movie_data["content"] = (
    movie_data["genres_clean"] + " " + movie_data["genres_clean"] + " " + movie_data["tags"]
)

movie_data = movie_data.reset_index(drop=True)

print(f"Building TF-IDF matrix over {len(movie_data)} movies...")
tfidf = TfidfVectorizer(stop_words="english", min_df=1)
tfidf_matrix = tfidf.fit_transform(movie_data["content"])

# --- Save artifacts ---
movie_data.to_csv("movie_data.csv", index=False)
with open("tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Saved movie_data.csv, tfidf_matrix.pkl, tfidf_vectorizer.pkl")
print(movie_data[["title", "genres", "avg_rating", "rating_count"]].head())
