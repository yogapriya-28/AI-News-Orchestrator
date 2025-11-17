# news_orchestrator_pro.py
# Clean, cloud-safe, error-free version

import os
import re
import json
from io import StringIO
import csv
import urllib.parse
from datetime import datetime
from collections import Counter

import streamlit as st
import requests
import feedparser
from dotenv import load_dotenv

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

openai_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        openai_client = None


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def clean_html(s):
    return re.sub(r"<[^>]+>", " ", str(s or "")).strip()

def short(s, w=200):
    s = str(s or "")
    return s if len(s) <= w else s[:w] + "..."

def extract_domain(url):
    try:
        host = urllib.parse.urlparse(url).netloc.lower().replace("www.", "")
        return host.split(":")[0]
    except:
        return ""


# -------------------------------------------------
# DOMAIN CREDIBILITY
# -------------------------------------------------
DOMAIN_WEIGHTS = {
    "reuters":3, "bbc":3, "nytimes":3, "washingtonpost":3,
    "thehindu":2, "timesofindia":2, "economictimes":2, "mint":2,
    "ndtv":1.5, "indianexpress":2, "hindustantimes":2, "bloomberg":3
}

def domain_weight(domain):
    domain = (domain or "").lower()
    for k, v in DOMAIN_WEIGHTS.items():
        if k in domain:
            return v
    return 0.8

def authenticity_score(domains):
    if not domains:
        return 0.0
    weights = [domain_weight(d) for d in domains]
    avg = sum(weights) / len(weights)
    return round((avg/3) * 100, 1)

def credibility_rows(domains):
    rows = []
    counts = Counter(domains)
    for d, c in counts.most_common():
        w = domain_weight(d)
        grade = (
            "A+" if w >= 3 else
            "A" if w >= 2.5 else
            "B" if w >= 1.5 else
            "C"
        )
        rows.append({"domain": d, "count": c, "weight": w, "grade": grade})
    return rows


# -------------------------------------------------
# NEWS FETCHERS
# -------------------------------------------------
@st.cache_data(ttl=300)
def fetch_newsapi(topic, max_articles=20):
    if not NEWS_API_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "apiKey": NEWS_API_KEY,
        "q": topic,
        "language": "en",
        "pageSize": max_articles,
        "sortBy": "publishedAt",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
    except:
        return []

    out = []
    for a in data.get("articles", [])[:max_articles]:
        out.append({
            "title": a.get("title", ""),
            "summary": clean_html(a.get("description") or a.get("content", "")),
            "url": a.get("url", ""),
            "published": a.get("publishedAt", ""),
            "source": extract_domain(a.get("url", "")),
        })
    return out


@st.cache_data(ttl=300)
def fetch_google_rss(query, max_items=20, country="IN"):
    enc = urllib.parse.quote(query)
    rss = f"https://news.google.com/rss/search?q={enc}&hl=en-{country}&gl={country}&ceid={country}:en"

    try:
        feed = feedparser.parse(rss)
    except:
        return []

    out, seen = [], set()
    for e in feed.entries[:max_items * 3]:
        title = e.get("title", "")
        link = e.get("link", "")

        if link and "url=" in link:
            m = re.search(r"url=([^&]+)", link)
            if m:
                link = urllib.parse.unquote(m.group(1))

        link = link.replace("amp;", "")

        if (title, link) in seen:
            continue
        seen.add((title, link))

        out.append({
            "title": title,
            "summary": clean_html(e.get("summary") or e.get("description", "")),
            "url": link,
            "published": e.get("published", ""),
            "source": extract_domain(link),
        })

        if len(out) >= max_items:
            break

    return out


# -------------------------------------------------
# OPENAI MODULES
# -------------------------------------------------
def relevance_openai(item, topic):
    if not openai_client:
        return 0.5

    text = (item.get("title", "") + " " + item.get("summary", ""))[:3000]
    prompt = f"Rate relevance 0–1 to '{topic}'. Only return a number.\n{text}"

    try:
        resp = openai_client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=10,
        )
        m = re.search(r"([01](?:\.\d+)?)", resp.output_text)
        if m:
            return float(m.group(1))
    except:
        pass

    return 0.5


def translate_openai(text, lang):
    if not openai_client:
        return f"[OpenAI key missing] Cannot translate to {lang}"

    prompt = f"Translate to {lang}:\n{text}"

    try:
        resp = openai_client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=300,
        )
        return resp.output_text.strip()
    except:
        return f"Translation failed ({lang})"


# -------------------------------------------------
# TIMELINE
# -------------------------------------------------
def extract_events(entries):
    timeline = []
    for e in entries:
        pub = e.get("published", "")
        try:
            d = datetime.fromisoformat(pub.replace("Z", "+00:00")).date().isoformat()
        except:
            d = "Unknown"

        timeline.append({
            "date": d,
            "event": e["title"],
            "summary": e["summary"],
            "url": e["url"],
        })
    return timeline


def render_vertical_timeline(timeline):
    st.subheader("📌 Vertical Timeline")

    for i, item in enumerate(timeline, 1):
        st.markdown(
            f"""
            ### {i}. {item['date']}
            **[{item['event']}]({item['url']})**  
            <span style='font-size:14px;color:#555;'>{short(item['summary'], 220)}</span>
            <hr>
            """,
            unsafe_allow_html=True
        )


# -------------------------------------------------
# PIPELINE
# -------------------------------------------------
def build_pipeline(topic, max_articles=20, country="IN", use_openai=True):
    entries = []

    if NEWS_API_KEY:
        entries = fetch_newsapi(topic, max_articles)

    if len(entries) < max_articles:
        fallback = fetch_google_rss(topic, max_items=max_articles, country=country)
        urls = {e["url"] for e in entries}
        for f in fallback:
            if f["url"] not in urls:
                entries.append(f)
            if len(entries) >= max_articles:
                break

    scored = []
    for e in entries:
        score = relevance_openai(e, topic) if use_openai else 0.5
        scored.append((score, e))

    scored.sort(reverse=True, key=lambda x: x[0])
    sorted_entries = [e for _, e in scored]

    timeline = extract_events(sorted_entries)
    domains = [extract_domain(e["url"]) for e in sorted_entries]
    summary = " ".join(e["summary"] for e in sorted_entries)

    return {
        "timeline": timeline,
        "domains": domains,
        "summary": summary,
    }


# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.set_page_config(page_title="AI News Orchestrator PRO", layout="wide")
st.title("🧭 AI News Orchestrator — PRO")

with st.sidebar:
    st.header("🌐 Region & Options")

    region = st.selectbox("Choose Region", ["India", "Global"])
    topic = st.text_input("Topic", "Chandrayaan-3")
    max_articles = st.slider("Max Articles", 5, 50, 20)

    lang_multi = st.multiselect(
        "Translate summary into languages",
        ["Tamil", "Hindi", "Spanish", "French", "German", "Arabic", "Russian"]
    )

    use_openai = st.checkbox(
        "Use OpenAI Ranking & Translation",
        value=bool(openai_client)
    )

country_code = "IN" if region == "India" else "US"

if st.button("🔎 Build Timeline"):
    with st.spinner("Fetching articles..."):
        data = build_pipeline(
            topic,
            max_articles=max_articles,
            country=country_code,
            use_openai=use_openai,
        )

    timeline = data["timeline"]
    domains = data["domains"]

    st.success(f"Generated timeline with {len(timeline)} events.")

    st.subheader("📝 Summary")
    st.write(data["summary"])

    if lang_multi:
        st.subheader("🌍 Translated Summaries")
        for lang in lang_multi:
            st.write(f"### {lang}")
            st.write(translate_openai(data["summary"], lang))
            st.write("---")

    render_vertical_timeline(timeline)

    st.subheader("✅ Domain Credibility Score")
    score = authenticity_score(domains)
    st.metric("Authenticity Score (%)", score)
    st.table(credibility_rows(domains))

    st.subheader("⬇ Export Timeline")

    csv_buf = StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(["date", "title", "summary", "url"])
    for it in timeline:
        writer.writerow([it["date"], it["event"], it["summary"], it["url"]])

    st.download_button(
        "Download CSV",
        csv_buf.getvalue(),
        "timeline.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download JSON",
        json.dumps(timeline, indent=2),
        "timeline.json",
        mime="application/json"
    )
