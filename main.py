# Movie Recommendation System - FastAPI backend
import os
import pickle
import difflib
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/original"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None
TITLE_TO_IDX: Optional[Dict[str, int]] = None

# TMDB genre ids. Kept here so mood/personalization can use current TMDB data.
GENRE_IDS = {
    "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35,
    "Crime": 80, "Documentary": 99, "Drama": 18, "Family": 10751,
    "Fantasy": 14, "History": 36, "Horror": 27, "Music": 10402,
    "Mystery": 9648, "Romance": 10749, "Science Fiction": 878,
    "Thriller": 53, "War": 10752, "Western": 37, "TV Movie": 10770,
}

MOOD_GENRES = {
    "😄 Happy": [35, 10751, 16, 10402],
    "❤️ Romantic": [10749, 35, 18],
    "😢 Emotional": [18, 10749, 10752],
    "😱 Thriller": [53, 9648, 80],
    "👻 Horror": [27, 53, 9648],
    "😂 Comedy": [35, 16, 10751],
    "🔥 Energetic": [28, 12, 878],
    "🧠 Mind-Bending": [878, 9648, 53],
    "🌌 Sci-Fi": [878, 12, 28],
    "😌 Relaxing": [35, 16, 10751, 10402],
}

app = FastAPI(title="CineMatch - Personalized Movie Discovery API", version="4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = Field(default_factory=list)
    homepage: Optional[str] = None
    imdb_url: Optional[str] = None
    tmdb_url: Optional[str] = None
    watch_url: Optional[str] = None
    watch_providers: List[str] = Field(default_factory=list)


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    similar_recommendations: List[TMDBMovieCard]
    genre_recommendations: List[TMDBMovieCard]


def _norm_title(t: str) -> str:
    return " ".join(str(t).strip().lower().split())


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(params)
    q["api_key"] = TMDB_API_KEY
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"TMDB request error: {type(e).__name__} | {repr(e)}")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"TMDB error {r.status_code}: {r.text}")
    return r.json()


async def tmdb_cards_from_results(results: List[dict], limit: int = 20) -> List[TMDBMovieCard]:
    out = []
    for m in (results or [])[:limit]:
        if not m.get("id"):
            continue
        out.append(TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or m.get("name") or "",
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date") or m.get("first_air_date"),
            vote_average=m.get("vote_average"),
        ))
    return out


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(
        f"/movie/{movie_id}",
        {"language": "en-US", "append_to_response": "external_ids,watch/providers"},
    )
    external = data.get("external_ids") or {}
    providers = (data.get("watch/providers") or {}).get("results") or {}
    region = providers.get("IN") or providers.get("US") or {}
    provider_names = []
    for key in ("flatrate", "free", "ads", "rent", "buy"):
        for p in region.get(key, []) or []:
            name = p.get("provider_name")
            if name and name not in provider_names:
                provider_names.append(name)

    imdb_id = external.get("imdb_id")
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
        homepage=data.get("homepage") or None,
        imdb_url=f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else None,
        tmdb_url=f"https://www.themoviedb.org/movie/{movie_id}",
        watch_url=region.get("link"),
        watch_providers=provider_names,
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    return await tmdb_get("/search/movie", {
        "query": query, "include_adult": "false", "language": "en-US", "page": page,
    })


async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


# ---------------- TF-IDF ----------------
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    title_to_idx = {}
    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception as e:
        raise RuntimeError("indices.pkl must be dict or pandas Series-like") from e


def get_local_idx_by_title(title: str) -> int:
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])

    # Fuzzy matching fixes harmless variations such as Spider-Man / Spider Man.
    keys = list(TITLE_TO_IDX.keys())
    matches = difflib.get_close_matches(key, keys, n=1, cutoff=0.78)
    if matches:
        return int(TITLE_TO_IDX[matches[0]])
    raise HTTPException(status_code=404, detail=f"Title not found in local dataset: '{title}'")


def tfidf_recommend_titles(query_title: str, top_n: int = 10) -> List[Tuple[str, float]]:
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")
    idx = get_local_idx_by_title(query_title)
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    order = np.argsort(-scores)
    out = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]), title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"), vote_average=m.get("vote_average"),
        )
    except Exception:
        return None


async def tmdb_recommendations(movie_id: int, limit: int = 12) -> List[TMDBMovieCard]:
    data = await tmdb_get(f"/movie/{movie_id}/recommendations", {"language": "en-US", "page": 1})
    return await tmdb_cards_from_results(data.get("results", []), limit=limit)


async def tmdb_similar(movie_id: int, limit: int = 12) -> List[TMDBMovieCard]:
    data = await tmdb_get(f"/movie/{movie_id}/similar", {"language": "en-US", "page": 1})
    return await tmdb_cards_from_results(data.get("results", []), limit=limit)


async def discover_movies(genre_ids: List[int], limit: int = 18, seed: Optional[int] = None) -> List[TMDBMovieCard]:
    params = {
        "with_genres": "|".join(str(x) for x in genre_ids),
        "language": "en-US",
        "sort_by": "popularity.desc",
        "page": 1,
        "vote_count.gte": 100,
    }
    if seed is not None:
        params["page"] = max(1, min(5, seed))
    data = await tmdb_get("/discover/movie", params)
    return await tmdb_cards_from_results(data.get("results", []), limit=limit)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)
    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)
    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)
    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")
    yield


app.router.lifespan_context = lifespan


@app.get("/health")
def health():
    return {"status": "ok", "version": "4.0", "features": [
        "voice-search", "personalization", "mood-recommendations", "surprise-me",
        "hybrid-recommendations", "official-links"
    ]}


@app.get("/home", response_model=List[TMDBMovieCard])
async def home(category: str = Query("popular"), limit: int = Query(24, ge=1, le=50)):
    if category == "trending":
        data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
        return await tmdb_cards_from_results(data.get("results", []), limit)
    if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
        raise HTTPException(status_code=400, detail="Invalid category")
    data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
    return await tmdb_cards_from_results(data.get("results", []), limit)


@app.get("/tmdb/search")
async def tmdb_search(query: str = Query(..., min_length=1), page: int = Query(1, ge=1, le=10)):
    return await tmdb_search_movies(query=query, page=page)


@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)


@app.get("/movie/{tmdb_id}/trailer")
async def movie_trailer(tmdb_id: int):
    videos = await tmdb_get(f"/movie/{tmdb_id}/videos", {"language": "en-US"})
    for video in videos.get("results", []):
        if video.get("site") == "YouTube" and video.get("type") == "Trailer":
            return {"youtube_url": f"https://www.youtube.com/watch?v={video['key']}"}
    return {"youtube_url": None}


@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(tmdb_id: int = Query(...), limit: int = Query(18, ge=1, le=50)):
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []
    cards = await discover_movies([details.genres[0]["id"]], limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]


@app.get("/recommend/tfidf")
async def recommend_tfidf(title: str = Query(..., min_length=1), top_n: int = Query(10, ge=1, le=50)):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


@app.get("/recommend/similar", response_model=List[TMDBMovieCard])
async def recommend_similar(tmdb_id: int = Query(...), limit: int = Query(12, ge=1, le=30)):
    """Hybrid fallback: TMDB recommendations -> TMDB similar."""
    cards = await tmdb_recommendations(tmdb_id, limit)
    if not cards:
        cards = await tmdb_similar(tmdb_id, limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]


@app.get("/recommend/mood", response_model=List[TMDBMovieCard])
async def recommend_mood(mood: str = Query(...), limit: int = Query(18, ge=1, le=50)):
    if mood not in MOOD_GENRES:
        raise HTTPException(status_code=400, detail=f"Unsupported mood. Choose from: {', '.join(MOOD_GENRES)}")
    cards = await discover_movies(MOOD_GENRES[mood], limit)
    return cards


@app.get("/recommend/personalized", response_model=List[TMDBMovieCard])
async def recommend_personalized(
    genre_ids: str = Query(..., description="Comma-separated TMDB genre ids"),
    limit: int = Query(18, ge=1, le=50),
):
    try:
        ids = [int(x.strip()) for x in genre_ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="genre_ids must be comma-separated integers")
    ids = list(dict.fromkeys(ids))[:5]
    if not ids:
        return await home("popular", limit)
    return await discover_movies(ids, limit)


@app.get("/random", response_model=TMDBMovieCard)
async def random_movie():
    # Random page + random item from that page gives a lightweight Surprise Me feature.
    import random
    page = random.randint(1, 5)
    data = await tmdb_get("/discover/movie", {
        "language": "en-US", "sort_by": "popularity.desc", "page": page,
        "vote_count.gte": 100,
    })
    cards = await tmdb_cards_from_results(data.get("results", []), 20)
    if not cards:
        raise HTTPException(status_code=404, detail="No random movie available")
    return random.choice(cards)


@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(status_code=404, detail=f"No TMDB movie found for query: {query}")
    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    tfidf_items: List[TFIDFRecItem] = []
    try:
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []
    for title, score in recs:
        card = await attach_tmdb_card_by_title(title)
        if card:
            tfidf_items.append(TFIDFRecItem(title=title, score=score, tmdb=card))

    # The important fix: if local TF-IDF has no matching movie, never leave Similar empty.
    similar = []
    if tfidf_items:
        similar = [x.tmdb for x in tfidf_items if x.tmdb]
    if not similar:
        similar = await tmdb_recommendations(tmdb_id, tfidf_top_n)
    if not similar:
        similar = await tmdb_similar(tmdb_id, tfidf_top_n)

    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_recs = await discover_movies([details.genres[0]["id"]], genre_limit)
        genre_recs = [c for c in genre_recs if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query, movie_details=details,
        tfidf_recommendations=tfidf_items,
        similar_recommendations=similar,
        genre_recommendations=genre_recs,
    )
