# 🎬 Movie Recommendation System

A content-based movie recommender built on the MovieLens dataset, with a Streamlit front end. Pick a movie you like and get similar titles based on genre and user-submitted tags, plus browsable Top Rated and Most Popular lists.

## Project structure

```
your-project/
├── movies.csv           # MovieLens: movieId, title, genres
├── ratings.csv          # MovieLens: userId, movieId, rating, timestamp
├── tags.csv             # MovieLens: userId, movieId, tag, timestamp
├── links.csv            # MovieLens: movieId, imdbId, tmdbId
├── prepare_data.py      # builds movie_data.csv + tfidf_matrix.pkl
├── movie.py             # the Streamlit app
├── movie_data.csv        (generated)
├── tfidf_matrix.pkl       (generated)
└── requirements.txt
```

`movies.csv`, `ratings.csv`, `tags.csv`, and `links.csv` are the raw MovieLens files — keep your existing copies in the project folder.

## Setup

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Build the data artifacts. Run this once, and again any time `movies.csv`, `ratings.csv`, or `tags.csv` change:
   ```bash
   python prepare_data.py
   ```
   This creates `movie_data.csv` (one row per movie with genres, aggregated tags, average rating, and rating count) and `tfidf_matrix.pkl` (the TF-IDF similarity matrix), with rows aligned so row *i* in both files always refers to the same movie.

4. Run the app:
   ```bash
   streamlit run movie.py
   ```
   Streamlit will open the app in your browser, typically at `http://localhost:8501`.

## Features

- **Get Recommendations tab** — pick a movie from a searchable dropdown and get similar titles, each shown with genre, similarity score, average rating, and number of ratings.
- **Adjustable result count** — slider to control how many recommendations come back (3–20).
- **Genre filter** — narrow recommendations, top-rated, and popular lists to specific genres.
- **Minimum ratings filter** — hide movies with too few ratings to be reliable.
- **Top Rated tab** — highest average-rated movies, with a configurable minimum ratings threshold so obscure one-review movies don't dominate.
- **Most Popular tab** — movies with the most ratings overall.

## How it works

`prepare_data.py` merges `movies.csv` with rating statistics (mean rating, rating count per movie) and aggregated tags per movie, then builds a TF-IDF matrix over each movie's genres + tags. `movie.py` loads that prebuilt matrix and, for a selected movie, ranks all other movies by cosine similarity.

## Note on a bug in the original version

The original script built `tfidf_matrix` from a dataframe that merged `movies`, `ratings`, and `tags` together (one row per *rating*, not per *movie*), but then indexed into it using row positions from `movies.csv` alone. Since the two didn't share the same row order, the app was often pulling the similarity vector for the wrong movie. `prepare_data.py` fixes this by building one de-duplicated row per movie and generating the TF-IDF matrix in that exact same order.

## Possible next steps

- Add posters via the TMDB API (`links.csv` already has `tmdbId`)
- Wire in the SVD collaborative-filtering model from the notebook for "recommend for this user" personalization
- Add a "surprise me" random pick button
- Deploy to Streamlit Community Cloud for a shareable link
