import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# ----------------------------------------------------------------------
# Load data (cached so it only happens once per session)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movie_data.csv")
    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    return movies, tfidf_matrix


movies, tfidf_matrix = load_data()
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

all_genres = sorted(
    {g for genre_str in movies["genres"].dropna() for g in genre_str.split("|") if g != "(no genres listed)"}
)

# ----------------------------------------------------------------------
# Recommendation logic
# ----------------------------------------------------------------------
def recommend(title, n=5, genre_filter=None, min_rating_count=0):
    if title not in indices:
        return pd.DataFrame()

    idx = indices[title]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]

    candidates = movies.copy()
    candidates["similarity"] = sim_scores
    candidates = candidates[candidates.index != idx]  # exclude the movie itself

    if genre_filter:
        candidates = candidates[
            candidates["genres"].apply(lambda g: any(genre in g.split("|") for genre in genre_filter))
        ]
    if min_rating_count:
        candidates = candidates[candidates["rating_count"] >= min_rating_count]

    candidates = candidates.sort_values("similarity", ascending=False).head(n)
    return candidates[["title", "genres", "avg_rating", "rating_count", "similarity"]]


def top_rated(n=10, genre_filter=None, min_rating_count=20):
    df = movies[movies["rating_count"] >= min_rating_count].copy()
    if genre_filter:
        df = df[df["genres"].apply(lambda g: any(genre in g.split("|") for genre in genre_filter))]
    return df.sort_values("avg_rating", ascending=False).head(n)[
        ["title", "genres", "avg_rating", "rating_count"]
    ]


def most_popular(n=10, genre_filter=None):
    df = movies.copy()
    if genre_filter:
        df = df[df["genres"].apply(lambda g: any(genre in g.split("|") for genre in genre_filter))]
    return df.sort_values("rating_count", ascending=False).head(n)[
        ["title", "genres", "avg_rating", "rating_count"]
    ]


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🎬 Movie Recommendation System")

with st.sidebar:
    st.header("Options")
    num_recs = st.slider("Number of recommendations", min_value=3, max_value=20, value=5)
    genre_filter = st.multiselect("Filter by genre", all_genres)
    min_rating_count = st.slider(
        "Minimum number of ratings a movie must have", min_value=0, max_value=200, value=0, step=5
    )
    st.caption("Filters apply to recommendations and to the Top Rated / Popular tabs.")

tab1, tab2, tab3 = st.tabs(["🔍 Get Recommendations", "⭐ Top Rated", "🔥 Most Popular"])

with tab1:
    movie_name = st.selectbox(
        "Pick a movie you like:",
        options=sorted(movies["title"].unique()),
        index=None,
        placeholder="Start typing a title...",
    )

    if movie_name:
        source = movies.loc[indices[movie_name]]
        st.caption(f"Because you liked **{movie_name}** ({source['genres']})")

        results = recommend(
            movie_name, n=num_recs, genre_filter=genre_filter, min_rating_count=min_rating_count
        )

        if results.empty:
            st.warning("No recommendations matched your filters. Try loosening them.")
        else:
            for _, row in results.iterrows():
                cols = st.columns([4, 2, 1, 1])
                cols[0].markdown(f"**{row['title']}**  \n{row['genres'].replace('|', ', ')}")
                cols[1].markdown(f"Match: {row['similarity']*100:.0f}%")
                cols[2].markdown(f"⭐ {row['avg_rating']}")
                cols[3].markdown(f"{int(row['rating_count'])} ratings")
                st.divider()

with tab2:
    st.subheader("Top rated movies")
    n_top = st.slider("How many to show", 5, 30, 10, key="top_n")
    top_df = top_rated(n=n_top, genre_filter=genre_filter, min_rating_count=max(min_rating_count, 20))
    st.dataframe(
        top_df.rename(
            columns={
                "title": "Title",
                "genres": "Genres",
                "avg_rating": "Avg Rating",
                "rating_count": "# Ratings",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

with tab3:
    st.subheader("Most rated (most popular) movies")
    n_pop = st.slider("How many to show", 5, 30, 10, key="pop_n")
    pop_df = most_popular(n=n_pop, genre_filter=genre_filter)
    st.dataframe(
        pop_df.rename(
            columns={
                "title": "Title",
                "genres": "Genres",
                "avg_rating": "Avg Rating",
                "rating_count": "# Ratings",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )