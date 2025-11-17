# AI News Orchestrator — Full Cloud-Safe Version
# Streamlit Cloud Optimized — OpenAI + NewsAPI Enabled

import os
import re
import json
import urllib.parse
from datetime import datetime, date
from io import StringIO
from collections import Counter

import streamlit as st
import requests
import feedparser
import plotly.graph_objects as go
from dotenv import load_dotenv

# --------------------
# Load API Keys
# --------------------
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --------------------
# Safe OpenAI Client
# --------------------
openai_client = None
try:
    if OPENAI_API_KEY:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception:
    openai_client = None

# --------------------
# Helpers
# --------------------
def clean_html(text):
    return re.sub(r"<[^>]+>", " ", str(text or "")).strip()

def short(text, w=200):
    text = str(text or "")
    return text if len(text) <= w else text[:w] + "..."

def extract_domain(url):
    try:
        net = urllib.parse.urlparse(url).netloc
        return net.replace("www.", "").strip()
    except Exception:
        return ""

# --------------------
# NewsAPI Fetch
# --------------------
@st.cache_data(ttl=300)
def fetch_newsapi(topic, max_items=20):
    if not NEWS_API_KEY:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "apiKey": NEWS_API_KEY,
        "q": topic,
        "pageSize": max_items,
        "language": "en",
        "sortBy": "publishedAt",
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
    except Exception:
        return []

    out = []
    for a in data.get("articles", []):
        out.append({
            "title": a.get("title", ""),
            "summary": clean_html(a.get("description") or a.get("content", "")),
            "url": a.get("url", ""),
            "published": a.get("publishedAt", ""),
        })
    return out

# --------------------
# Google RSS Fallback
# --------------------
@st.cache_data(ttl=300)
def fetch_google_rss(query, max_items=20):
    rss = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        feed = feedparser.parse(rss)
    except Exception:
        return []

    out = []
    seen = set()

    for e in feed.entries[: max_items * 3]:
        title = e.get("title", "")
        link = e.get("link", "")

        # Extract real URL
        if "url=" in link:
            m = re.search(r"url=([^&]+)", link)
            if m:
                link = urllib.parse.unquote(m.group(1))

        if (title, link) in seen:
            continue
        seen.add((title, link))

        out.append({
            "title": title,
            "summary": clean_html(e.get("summary", "")),
            "url": link,
            "published": e.get("published", ""),
        })

        if len(out) >= max_items:
            break

    return out

# --------------------
# Timeline extraction
# --------------------
def extract_events(article):
    events = []
    pub = article.get("published", "")
    try:
        dt = datetime.fromisoformat(pub.replace("Z", "+00:00")).date()
    except Exception:
        dt = None

    events.append(
        {
            "date": dt.isoformat() if dt else "",
            "event": short(article.get("title", ""), 200),
            "sources": [article.get("url", "")],
        }
    )
    return events

# --------------------
# OpenAI relevance scoring
# --------------------
def relevance_score(article, topic):
    if not openai_client:
        return 0.5

    prompt = (
    f"Rate relevance 0-1 of this article to '{topic}'.\n"
    f"Title: {article['title']}\n"
    f"Summary: {article['summary']}"
)


    try:
        res = openai_client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=20,
        )
        txt = res.output_text
        m = re.search(r"([01](?:\.\d+)?)", txt)
        return float(m.group(1)) if m else 0.5
    except Exception:
        return 0.5

# --------------------
# Build Timeline Pipeline
# --------------------
def build_timeline(topic, max_articles=20):
    data = []

    # Fetch NewsAPI
    primary = fetch_newsapi(topic, max_articles)
    data.extend(primary)

    # Fallback RSS
    if len(data) < max_articles:
        fallback = fetch_google_rss(topic, max_articles)
        for f in fallback:
            if len(data) >= max_articles:
                break
            if f["url"] not in [d["url"] for d in data]:
                data.append(f)

    # Relevance scoring
    scored = []
    for a in data:
        s = relevance_score(a, topic)
        scored.append((s, a))

    scored.sort(reverse=True, key=lambda x: x[0])
    entries = [a for s, a in scored]

    # Timeline
    timeline = []
    for a in entries:
        timeline.extend(extract_events(a))

    return entries, timeline

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="AI News Orchestrator", layout="wide")
st.title("🧭 AI News Orchestrator — Cloud Safe Edition")

query = st.text_input("Search Topic", "Chandrayaan-3")
max_items = st.slider("Max Articles", 5, 50, 20)

if st.button("Build Timeline"):
    if len(query.strip()) < 3:
        st.error("Enter a valid topic.")
        st.stop()

    with st.spinner("Fetching, analyzing, ranking..."):
        entries, timeline = build_timeline(query, max_items)

    st.success(f"Fetched {len(entries)} articles. Timeline generated.")

    # Summary
    st.header("Summary")
    combined = " ".join([e["title"] + " " + e["summary"] for e in entries])
    st.write(short(combined, 1200))

    # Timeline list
    st.header("Timeline")
    for i, item in enumerate(timeline, 1):
        st.subheader(f"{i}. {item['date']}")
        st.write(item["event"])
        for s in item["sources"]:
            st.write(f"- [{extract_domain(s)}]({s})")
        st.markdown("---")

    # Downloads
    st.header("Export")
    j = json.dumps(timeline, ensure_ascii=False, indent=2)
    st.download_button("Download JSON", j, "timeline.json", "application/json")

# End of File
