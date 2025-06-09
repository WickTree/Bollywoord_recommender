import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎓 Final Year Project - Bollywood Movie Recommendation Dashboard")
st.markdown("An intelligent system that recommends movies using **semantic overview matching**, **genre**, **director**, and **cast similarity**.")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("IMDB-Movie-Dataset(2023-1951)[2].csv")
    df.dropna(subset=["overview", "genre", "director", "cast"], inplace=True)
    df["Star1"] = df["cast"].apply(lambda x: x.split(",")[0].strip() if isinstance(x, str) else "")
    df["Star2"] = df["cast"].apply(lambda x: x.split(",")[1].strip() if isinstance(x, str) and len(x.split(",")) > 1 else "")
    df["Combined_Info"] = (
        df["overview"].fillna("") + " " +
        df["genre"].fillna("") + " " +
        df["Star1"] + " " +
        df["Star2"] + " " +
        df["director"].fillna("")
    )
    return df.reset_index(drop=True)

df = load_data()
movies = df["movie_name"].tolist()

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

@st.cache_data
def compute_embeddings():
    return model.encode(df["overview"].tolist(), convert_to_tensor=True)

overview_embeddings = compute_embeddings()

# Initialize session state weights
if 'original_weights' not in st.session_state:
    st.session_state.original_weights = {
        'genre': 0.3,
        'director': 0.2,
        'cast': 0.1,
        'overview': 0.4
    }

if 'weights' not in st.session_state:
    st.session_state.weights = st.session_state.original_weights.copy()

def reset_weights():
    st.session_state.weights = st.session_state.original_weights.copy()

# Sidebar: sliders and reset
st.sidebar.header("⚙️ Adjust Similarity Weights")
st.sidebar.button("🔄 Reset to Default", on_click=reset_weights)

input_weights = {}
for key in ['genre', 'director', 'cast', 'overview']:
    input_weights[key] = st.sidebar.slider(
        f"{key.capitalize()} Weight (Before Normalization)",
        0.0, 1.0, float(st.session_state.weights[key]), step=0.05
    )

# Normalize weights
total = sum(input_weights.values()) or 1  # Avoid divide by zero
normalized_weights = {k: round(v/total, 3) for k, v in input_weights.items()}
st.session_state.weights = normalized_weights.copy()

st.sidebar.markdown("### 🔍 Normalized Weights")
for key, val in normalized_weights.items():
    st.sidebar.markdown(f"- **{key.capitalize()}**: {val:.3f}")

# Movie selection
st.sidebar.header("🎞️ Quick Select")
selected_movies_button = [movie for movie in movies[:100] if st.sidebar.button(movie, key=movie)]
selected_movies_dropdown = st.multiselect("📽️ Select Movies for Recommendation", options=movies)
selected_movies = list(set(selected_movies_button + selected_movies_dropdown))

# Recommendation logic
def get_recommendations(selected_movies, top_n=5):
    if not selected_movies:
        return []

    indices = [movies.index(movie) for movie in selected_movies if movie in movies]
    avg_embedding = np.mean([overview_embeddings[idx] for idx in indices], axis=0)
    similarity_scores = util.cos_sim(avg_embedding, overview_embeddings)[0].cpu().numpy()

    input_rows = df.iloc[indices]
    results = []

    for i, row in df.iterrows():
        if i in indices:
            continue

        genre_match = any(row["genre"] == r["genre"] for _, r in input_rows.iterrows())
        director_match = any(row["director"] == r["director"] for _, r in input_rows.iterrows())
        cast_match = any(
            row["Star1"] in [r["Star1"], r["Star2"]] or
            row["Star2"] in [r["Star1"], r["Star2"]]
            for _, r in input_rows.iterrows()
        )

        genre_score = 1 if genre_match else 0
        director_score = 1 if director_match else 0
        cast_score = 1 if cast_match else 0
        overview_score = similarity_scores[i]

        score = (
            genre_score * normalized_weights['genre'] +
            director_score * normalized_weights['director'] +
            cast_score * normalized_weights['cast'] +
            overview_score * normalized_weights['overview']
        )

        results.append({
            'title': row['movie_name'],
            'weighted_score': score,
            'overview_score': overview_score,
            'genre_score': genre_score,
            'director_score': director_score,
            'cast_score': cast_score
        })

    return sorted(results, key=lambda x: x['weighted_score'], reverse=True)[:top_n]

# Display results
def display_recommendations(recommendations):
    if not recommendations:
        st.warning("No recommendations to show.")
        return

    st.markdown("## 🎯 Top Movie Recommendations")
    genre_total = director_total = cast_total = 0
    overview_scores = []
    weighted_scores = []

    for rec in recommendations:
        overview_scores.append(rec['overview_score'])
        weighted_scores.append(rec['weighted_score'])

        genre_total += rec['genre_score']
        director_total += rec['director_score']
        cast_total += rec['cast_score']

        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.subheader(f"🎬 {rec['title']}")
                reasons = []
                if rec['genre_score']: reasons.append("Genre")
                if rec['director_score']: reasons.append("Director")
                if rec['cast_score']: reasons.append("Cast")
                if not reasons: reasons.append("Overview")
                st.markdown("Reason: **" + ", ".join(reasons) + "**")
                st.progress(float(rec['weighted_score']), text=f"Score: {rec['weighted_score']*100:.2f}%")

            with col2:
                with st.expander("🧠 Contributions", expanded=False):
                    labels = ["Genre", "Director", "Cast", "Overview"]
                    values = [
                        normalized_weights['genre'] if rec['genre_score'] else 0,
                        normalized_weights['director'] if rec['director_score'] else 0,
                        normalized_weights['cast'] if rec['cast_score'] else 0,
                        normalized_weights['overview'] * rec['overview_score']
                    ]
                    values += values[:1]
                    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
                    angles += angles[:1]

                    fig, ax = plt.subplots(figsize=(2.5, 2.5), subplot_kw=dict(polar=True))
                    ax.plot(angles, values, color='teal', linewidth=2)
                    ax.fill(angles, values, color='teal', alpha=0.25)
                    ax.set_yticklabels([])
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(labels, fontsize=8)
                    st.pyplot(fig, clear_figure=True)

    st.markdown("## 📊 Summary Metrics")
    st.write({
        "🎭 Genre Matches": genre_total,
        "🎬 Director Matches": director_total,
        "🌟 Cast Matches": cast_total,
        "📈 Avg. Overview Similarity": f"{np.mean(overview_scores)*100:.2f}%",
        "🏆 Avg. Weighted Similarity": f"{np.mean(weighted_scores)*100:.2f}%"
    })

# Main logic
if selected_movies:
    recs = get_recommendations(selected_movies)
    display_recommendations(recs)
else:
    st.info("Please select at least one movie to get recommendations.")