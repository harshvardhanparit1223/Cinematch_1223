import io
import html
import requests
import streamlit as st

try:
    from streamlit_mic_recorder import speech_to_text
    VOICE_AVAILABLE = True
except Exception:
    VOICE_AVAILABLE = False

API_BASE = "(https://cinematch-1223-5.onrender.com)"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

st.markdown("""
<style>
.stApp{background-color:#141414;color:white}
[data-testid="stSidebar"]{background:#000;color:white}
h1,h2,h3,h4,h5,h6{color:white !important}
.card{background:#1f1f1f;border-radius:15px;padding:15px;border:1px solid #2a2a2a;box-shadow:0 5px 15px rgba(0,0,0,.4)}
.stButton>button{border:none;border-radius:8px;font-weight:bold}
.movie-title{color:white;font-weight:600;text-align:center;margin-top:10px;min-height:42px}
.small-muted{color:#b3b3b3}
.feature-pill{display:inline-block;background:#242424;border:1px solid #3b3b3b;border-radius:20px;padding:5px 10px;margin:3px;font-size:13px}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION / USER PROFILE ----------------
def init_state():
    defaults = {
        "view": "home",
        "selected_tmdb_id": None,
        "search_text": "",
        "history": [],
        "liked": {},
        "disliked": set(),
        "genre_scores": {},
        "tracked_ids": set(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def goto_home():
    st.session_state.view = "home"
    st.session_state.selected_tmdb_id = None
    st.query_params["view"] = "home"
    if "id" in st.query_params:
        del st.query_params["id"]
    st.rerun()


def goto_details(tmdb_id: int):
    st.session_state.view = "details"
    st.session_state.selected_tmdb_id = int(tmdb_id)
    st.query_params["view"] = "details"
    st.query_params["id"] = str(int(tmdb_id))
    st.rerun()


qp_view = st.query_params.get("view")
qp_id = st.query_params.get("id")
if qp_view in ("home", "details"):
    st.session_state.view = qp_view
if qp_id:
    try:
        st.session_state.selected_tmdb_id = int(qp_id)
        st.session_state.view = "details"
    except ValueError:
        pass


# ---------------- API HELPERS ----------------
@st.cache_data(ttl=30)
def api_get_json(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=25)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}: {r.text[:300]}"
        return r.json(), None
    except Exception as e:
        return None, f"Request failed: {e}"


def safe_text(value):
    return html.escape(str(value or ""))


def track_movie(details):
    """Record a movie once per session and learn its genres."""
    if not details:
        return
    mid = int(details.get("tmdb_id"))
    if mid in st.session_state.tracked_ids:
        return
    st.session_state.tracked_ids.add(mid)
    st.session_state.history.insert(0, {
        "tmdb_id": mid,
        "title": details.get("title", "Untitled"),
        "poster_url": details.get("poster_url"),
    })
    st.session_state.history = st.session_state.history[:20]
    for g in details.get("genres", []) or []:
        gid = int(g.get("id"))
        st.session_state.genre_scores[gid] = st.session_state.genre_scores.get(gid, 0) + 1


def poster_grid(cards, cols=6, key_prefix="grid"):
    if not cards:
        st.info("No movies to show.")
        return
    for row_start in range(0, len(cards), cols):
        row = cards[row_start:row_start + cols]
        colset = st.columns(cols)
        for c, m in enumerate(row):
            with colset[c]:
                tmdb_id = m.get("tmdb_id")
                title = m.get("title", "Untitled")
                poster = m.get("poster_url")
                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.write("🖼️ No poster")
                if st.button("Open", key=f"{key_prefix}_{row_start}_{c}_{tmdb_id}") and tmdb_id:
                    goto_details(tmdb_id)
                st.markdown(f"<div class='movie-title'>{safe_text(title)}</div>", unsafe_allow_html=True)


def to_cards_from_tfidf_items(items):
    cards = []
    for x in items or []:
        tmdb = x.get("tmdb") or {}
        if tmdb.get("tmdb_id"):
            cards.append({
                "tmdb_id": tmdb["tmdb_id"],
                "title": tmdb.get("title") or x.get("title") or "Untitled",
                "poster_url": tmdb.get("poster_url"),
            })
    return cards


def parse_tmdb_search_to_cards(data, keyword: str, limit: int = 24):
    raw_items = []
    if isinstance(data, dict) and "results" in data:
        for m in data.get("results") or []:
            title = (m.get("title") or "").strip()
            tmdb_id = m.get("id")
            if title and tmdb_id:
                raw_items.append({
                    "tmdb_id": int(tmdb_id), "title": title,
                    "poster_url": f"{TMDB_IMG}{m.get('poster_path')}" if m.get("poster_path") else None,
                    "release_date": m.get("release_date", ""),
                })
    elif isinstance(data, list):
        for m in data:
            tmdb_id = m.get("tmdb_id") or m.get("id")
            title = (m.get("title") or "").strip()
            if title and tmdb_id:
                raw_items.append({
                    "tmdb_id": int(tmdb_id), "title": title,
                    "poster_url": m.get("poster_url"), "release_date": m.get("release_date", ""),
                })
    if not raw_items:
        return [], []
    key = keyword.strip().lower()
    matched = [x for x in raw_items if key in x["title"].lower()]
    final_list = matched if matched else raw_items
    suggestions = []
    for x in final_list[:10]:
        year = (x.get("release_date") or "")[:4]
        suggestions.append((f"{x['title']} ({year})" if year else x["title"], x["tmdb_id"]))
    return suggestions, final_list[:limit]


def top_profile_genres():
    scores = st.session_state.genre_scores
    return [gid for gid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]]


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    if st.button("🏠 Home", use_container_width=True):
        goto_home()

    st.markdown("---")
    st.markdown("### 🧭 Discover")
    home_category = st.selectbox(
        "Home Feed", ["trending", "popular", "top_rated", "now_playing", "upcoming"], index=0
    )
    grid_cols = st.slider("Grid columns", 4, 8, 6)

    st.markdown("### 🎭 Mood Match")
    mood = st.selectbox("What are you in the mood for?", [
        "😄 Happy", "❤️ Romantic", "😢 Emotional", "😱 Thriller", "👻 Horror",
        "😂 Comedy", "🔥 Energetic", "🧠 Mind-Bending", "🌌 Sci-Fi", "😌 Relaxing"
    ])
    if st.button("🎭 Find Movies for My Mood", use_container_width=True):
        cards, err = api_get_json("/recommend/mood", params={"mood": mood, "limit": 18})
        if err:
            st.error(err)
        else:
            st.session_state.mood_cards = cards or []
            st.session_state.view = "home"
            st.rerun()

    if st.button("🎲 Surprise Me", use_container_width=True):
        movie, err = api_get_json("/random")
        if err:
            st.error(err)
        elif movie:
            goto_details(movie["tmdb_id"])

    st.markdown("---")
    st.markdown("### 👤 Your Profile")
    st.caption(f"Movies explored: {len(st.session_state.history)}")
    st.caption(f"Liked: {len(st.session_state.liked)}")
    preferred = top_profile_genres()
    if preferred:
        st.success("Your taste profile is learning from your interactions.")
    else:
        st.info("Explore a few movies and like your favourites to build your taste profile.")
    if st.button("🧹 Reset My Profile", use_container_width=True):
        for k in ["history", "liked", "disliked", "genre_scores", "tracked_ids"]:
            st.session_state[k] = [] if k in ["history"] else set() if k in ["disliked", "tracked_ids"] else {}
        st.rerun()

    st.markdown("---")
    st.markdown("### 🕐 Recently Viewed")
    for item in st.session_state.history[:6]:
        if st.button(item["title"], key=f"history_{item['tmdb_id']}", use_container_width=True):
            goto_details(item["tmdb_id"])

# ---------------- HEADER ----------------
st.title("🎬 CineMatch")
st.markdown(
    "<div class='small-muted'>Personalized movie discovery with voice search, mood matching, hybrid recommendations and Surprise Me.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<span class='feature-pill'>🎙️ Voice Search</span><span class='feature-pill'>🎭 Mood Match</span>"
    "<span class='feature-pill'>👤 Personalization</span><span class='feature-pill'>🎲 Surprise Me</span>",
    unsafe_allow_html=True,
)
st.divider()

# ---------------- HOME ----------------
if st.session_state.view == "home":
    st.markdown("### 🔎 Find a Movie")
    search_col, voice_col = st.columns([5, 1])
    with search_col:
        typed = st.text_input(
            "Search by movie title",
            value=st.session_state.search_text,
            placeholder="Try: Spider-Man, Avatar, Interstellar...",
            label_visibility="collapsed",
        )
        st.session_state.search_text = typed
    with voice_col:
        if VOICE_AVAILABLE:
            voice_text = speech_to_text(
                language="en-US", start_prompt="🎙️ Speak", stop_prompt="⏹️ Stop", just_once=True,
                key="voice_search",
            )
            if voice_text:
                st.session_state.search_text = voice_text.strip()
                st.rerun()
        else:
            st.caption("Voice feature needs the project requirements installed.")

    if st.session_state.search_text.strip():
        query = st.session_state.search_text.strip()
        if len(query) < 2:
            st.caption("Type at least 2 characters.")
        else:
            data, err = api_get_json("/tmdb/search", params={"query": query})
            if err:
                st.error(f"Search failed: {err}")
            else:
                suggestions, cards = parse_tmdb_search_to_cards(data, query, limit=24)
                if suggestions:
                    labels = ["-- Select a movie --"] + [s[0] for s in suggestions]
                    selected = st.selectbox("Suggestions", labels, index=0)
                    if selected != "-- Select a movie --":
                        label_to_id = {s[0]: s[1] for s in suggestions}
                        goto_details(label_to_id[selected])
                else:
                    st.info("No suggestions found. Try another keyword.")
                st.markdown("### Search Results")
                poster_grid(cards, cols=grid_cols, key_prefix="search_results")
        st.stop()

    # Mood results are kept until the user changes the search.
    if st.session_state.get("mood_cards"):
        st.markdown(f"### 🎭 Movies for {safe_text(mood)}")
        poster_grid(st.session_state.mood_cards, cols=grid_cols, key_prefix="mood_results")
        st.divider()

    # Personalized feed
    profile_genres = top_profile_genres()
    if profile_genres:
        st.markdown("### ✨ Recommended For You")
        personalized, perr = api_get_json(
            "/recommend/personalized",
            params={"genre_ids": ",".join(map(str, profile_genres)), "limit": 18},
        )
        if not perr and personalized:
            poster_grid(personalized, cols=grid_cols, key_prefix="personalized")
        st.divider()

    # Home feed
    st.markdown(f"### 🏠 {home_category.replace('_', ' ').title()}")
    home_cards, err = api_get_json("/home", params={"category": home_category, "limit": 24})
    if err or not home_cards:
        st.error(f"Home feed failed: {err or 'Unknown error'}")
    else:
        poster_grid(home_cards, cols=grid_cols, key_prefix="home_feed")

# ---------------- DETAILS ----------------
elif st.session_state.view == "details":
    tmdb_id = st.session_state.selected_tmdb_id
    if not tmdb_id:
        st.warning("No movie selected.")
        st.stop()

    top_a, top_b = st.columns([3, 1])
    with top_a:
        st.markdown("### 📄 Movie Details")
    with top_b:
        if st.button("← Back to Home", use_container_width=True):
            goto_home()

    data, err = api_get_json(f"/movie/id/{tmdb_id}")
    if err or not data:
        st.error(f"Could not load details: {err or 'Unknown error'}")
        st.stop()

    track_movie(data)
    left, right = st.columns([1, 2.4], gap="large")
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if data.get("poster_url"):
            st.image(data["poster_url"], use_container_width=True)
        else:
            st.write("🖼️ No poster")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"## {safe_text(data.get('title', ''))}")
        release = data.get("release_date") or "-"
        genres = ", ".join(g.get("name", "") for g in data.get("genres", []) or []) or "-"
        st.markdown(f"<div class='small-muted'>Release: {safe_text(release)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>Genres: {safe_text(genres)}</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Overview")
        st.write(data.get("overview") or "No overview available.")

        # Like / dislike builds the personalization profile.
        like_col, dislike_col = st.columns(2)
        with like_col:
            if st.button("❤️ Like this movie", use_container_width=True):
                st.session_state.liked[tmdb_id] = data.get("title", "")
                st.session_state.disliked.discard(tmdb_id)
                for g in data.get("genres", []) or []:
                    gid = int(g["id"])
                    st.session_state.genre_scores[gid] = st.session_state.genre_scores.get(gid, 0) + 3
                st.success("Added to your taste profile!")
        with dislike_col:
            if st.button("👎 Not for me", use_container_width=True):
                st.session_state.disliked.add(tmdb_id)
                st.session_state.liked.pop(tmdb_id, None)
                st.info("Got it. We will avoid treating this movie as a preference.")

        # Legal / official access links only.
        st.markdown("### 🔗 Official Movie Links")
        links = []
        if data.get("homepage"):
            links.append(("🎬 Official Website", data["homepage"]))
        if data.get("imdb_url"):
            links.append(("⭐ IMDb", data["imdb_url"]))
        if data.get("tmdb_url"):
            links.append(("📚 TMDB", data["tmdb_url"]))
        if data.get("watch_url"):
            links.append(("▶️ Where to Watch", data["watch_url"]))
        if links:
            for label, url in links:
                st.markdown(f"[{label}]({url})")
            if data.get("watch_providers"):
                st.caption("Available providers: " + ", ".join(data["watch_providers"]))
        else:
            st.caption("No official external link was returned for this movie.")

        trailer, terr = api_get_json(f"/movie/{tmdb_id}/trailer")
        if not terr and trailer and trailer.get("youtube_url"):
            st.markdown("### 🎬 Official Trailer")
            st.video(trailer["youtube_url"])
        st.markdown("</div>", unsafe_allow_html=True)

    if data.get("backdrop_url"):
        st.markdown("#### Backdrop")
        st.image(data["backdrop_url"], use_container_width=True)

    st.divider()
    title = (data.get("title") or "").strip()

    # Recommendation bundle: local TF-IDF when possible + TMDB fallback.
    if title:
        bundle, err2 = api_get_json(
            "/movie/search",
            params={"query": title, "tfidf_top_n": 12, "genre_limit": 12},
        )
        if not err2 and bundle:
            similar_cards = bundle.get("similar_recommendations") or to_cards_from_tfidf_items(bundle.get("tfidf_recommendations"))
            st.markdown("### 🧠 Similar Content")
            st.caption("Local TF-IDF is used when the movie exists in the trained dataset; TMDB recommendations/similar titles fill the gap for newer movies.")
            poster_grid(similar_cards, cols=grid_cols, key_prefix="details_similar")

            st.markdown("### 🎭 More Like This")
            poster_grid(bundle.get("genre_recommendations", []), cols=grid_cols, key_prefix="details_genre")
        else:
            st.warning("Recommendation service is temporarily unavailable.")

    # Personalization CTA
    if top_profile_genres():
        st.divider()
        st.markdown("### ✨ Because of Your Taste")
        st.caption("Your recommendations are updated from movies you explore and like.")
        recs, perr = api_get_json(
            "/recommend/personalized",
            params={"genre_ids": ",".join(map(str, top_profile_genres())), "limit": 12},
        )
        if not perr:
            poster_grid(recs, cols=grid_cols, key_prefix="details_personalized")
