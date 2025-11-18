import os, re, json, csv, time, urllib.parse
from io import StringIO
from datetime import datetime, date
from math import ceil
from collections import OrderedDict

import streamlit as st
import requests
import feedparser
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load .env (optional for local)
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Config ---
DEFAULT_HEADERS = {"User-Agent": "AI-News-Orchestrator-Cloud/1.0"}
MAX_TOTAL_FETCH = 200        # max articles we'll attempt to fetch (sane for cloud)
NEWSAPI_PAGE_SIZE = 50      # NewsAPI supports up to 50 (depends on plan); we'll try 50 then page
RSS_PER_QUERY = 40
YEARS_BACK = 10               # how many past years to add as query variations

# -------------------------
# Helpers
# -------------------------
def safe_get(url, params=None, timeout=10):
    try:
        return requests.get(url, params=params, timeout=timeout, headers=DEFAULT_HEADERS)
    except Exception:
        return None

def clean_html(s):
    if not s: return ""
    return re.sub(r"<[^>]+>", " ", str(s)).strip()

def extract_domain(url):
    try:
        host = urllib.parse.urlparse(url).netloc.lower().replace("www.","")
        return host.split(":")[0]
    except Exception:
        return ""

def parse_date_iso(s):
    if not s: return None
    s = str(s)
    try:
        # try strict ISO first
        return datetime.fromisoformat(s.replace("Z","+00:00")).date()
    except Exception:
        pass
    # email-style
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(s)
        return dt.date()
    except Exception:
        pass
    # year heuristics
    m = re.search(r"\b(20\d{2}|19\d{2})\b", s)
    if m:
        try:
            return date(int(m.group(1)), 1, 1)
        except:
            return None
    return None

# -------------------------
# Query variations (for historical coverage)
# -------------------------
def build_query_variations(topic):
    q = topic.strip()
    parts = [q, f"{q} timeline", f"{q} history", f"{q} archive", f"{q} latest"]
    # add trailing year keywords for past years (helps RSS)
    this_year = datetime.utcnow().year
    for y in range(this_year, this_year - YEARS_BACK, -1):
        parts.append(f"{q} {y}")
    seen = []
    for p in parts:
        if p and p not in seen:
            seen.append(p)
    return seen

# -------------------------
# Fetch from NewsAPI using pagination
# -------------------------
@st.cache_data(ttl=300)
def fetch_newsapi_paginated(topic, max_items=100):
    if not NEWS_API_KEY:
        return []
    out = []
    page = 1
    page_size = min(NEWSAPI_PAGE_SIZE, max_items)
    # NewsAPI may limit rate/total; this is best-effort
    while len(out) < max_items:
        params = {
            "apiKey": NEWS_API_KEY,
            "qInTitle": topic,
            "language": "en",
            "pageSize": page_size,
            "page": page,
            "sortBy": "publishedAt"
        }
        r = safe_get("https://newsapi.org/v2/everything", params=params, timeout=12)
        if not r:
            break
        try:
            data = r.json()
        except Exception:
            break
        articles = data.get("articles") or []
        if not articles:
            break
        for a in articles:
            out.append({
                "title": a.get("title") or "",
                "summary": clean_html(a.get("description") or a.get("content") or ""),
                "url": a.get("url") or "",
                "published": a.get("publishedAt") or "",
                "source": extract_domain(a.get("url") or "")
            })
            if len(out) >= max_items:
                break
        # stop if fewer articles returned than page_size (end)
        if len(articles) < page_size:
            break
        page += 1
        # safety
        if page > 10:
            break
    return out

# -------------------------
# Google News RSS fallback for queries
# -------------------------
@st.cache_data(ttl=300)
def fetch_google_rss_for_query(query, limit=20, country="IN"):
    q = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-{country}&gl={country}&ceid={country}:en"
    try:
        feed = feedparser.parse(url)
    except Exception:
        return []
    out = []
    seen = set()
    for e in (feed.entries or [])[:limit*3]:
        title = e.get("title","")
        link = e.get("link","") or ""
        # fix google redirect
        if "url=" in link:
            m = re.search(r"[?&]url=([^&]+)", link)
            if m:
                link = urllib.parse.unquote(m.group(1))
        link = link.replace("amp;","")
        if (title, link) in seen:
            continue
        seen.add((title, link))
        out.append({
            "title": title,
            "summary": clean_html(e.get("summary") or e.get("description") or ""),
            "url": link,
            "published": e.get("published") or e.get("pubDate") or "",
            "source": extract_domain(link)
        })
        if len(out) >= limit:
            break
    return out

# -------------------------
# Orchestrator: fetch expanded
# -------------------------
def gather_articles(topic, max_items=120, region="IN"):
    max_items = min(max_items, MAX_TOTAL_FETCH)
    entries = []
    seen_urls = set()

    # 1) NewsAPI primary: try paginated fetch (title-limited first for relevance)
    n_from_api = min(max_items, 120)
    newsapi_items = fetch_newsapi_paginated(topic, n_from_api)
    for a in newsapi_items:
        if a["url"] and a["url"] not in seen_urls:
            entries.append(a); seen_urls.add(a["url"])
        if len(entries) >= max_items:
            break

    # 2) Expand via query variations and RSS to get older / additional content
    if len(entries) < max_items:
        queries = build_query_variations(topic)
        # iterate queries older-first (years produce older results)
        for q in queries:
            if len(entries) >= max_items: break
            rss_items = fetch_google_rss_for_query(q, limit=RSS_PER_QUERY, country=region)
            for r in rss_items:
                if r["url"] and r["url"] not in seen_urls:
                    entries.append(r); seen_urls.add(r["url"])
                if len(entries) >= max_items: break

    # 3) If still short, try NewsAPI with general q (not limited to title)
    if len(entries) < max_items and NEWS_API_KEY:
        more = fetch_newsapi_paginated(topic, max_items - len(entries))
        for a in more:
            if a["url"] and a["url"] not in seen_urls:
                entries.append(a); seen_urls.add(a["url"])
            if len(entries) >= max_items: break

    return entries

# -------------------------
# Timeline building (oldest -> newest)
# -------------------------
def build_timeline_from_entries(entries):
    items = []
    for e in entries:
        d = parse_date_iso(e.get("published") or "")
        items.append({
            "date_obj": d,
            "date": d.isoformat() if d else "",
            "event": e.get("title",""),
            "summary": e.get("summary",""),
            "url": e.get("url",""),
            "source": e.get("source","")
        })
    # sort: put items with real dates first, ascending. Unknown dates after.
    items_sorted = sorted(items, key=lambda it: (it["date_obj"] is None, it["date_obj"] or date(9999,1,1)))
    # remove date_obj before returning
    for it in items_sorted:
        it.pop("date_obj", None)
    return items_sorted

# -------------------------
# Vertical timeline chart (Plotly)
# -------------------------
def vertical_timeline_plotly(timeline, title="Timeline"):
    if not timeline:
        return None
    n = len(timeline)
    y = list(range(n))[::-1]
    dates = [t["date"] or "Unknown" for t in timeline]
    short_events = [re.sub(r"\s+"," ", t["event"])[:80] for t in timeline]

    fig = go.Figure()
    # central line
    fig.add_trace(go.Scatter(x=[0]*n, y=y, mode="lines", line=dict(color="#e6e6e6", width=4), hoverinfo="skip"))
    # markers
    fig.add_trace(go.Scatter(
        x=[0]*n, y=y, mode="markers+text",
        marker=dict(size=28, color="#0057ff", line=dict(width=2, color="#222")),
        text=[f"{i+1}" for i in range(n)],
        textposition="middle center",
        hoverinfo="skip"
    ))
    # date + short event on right
    fig.add_trace(go.Scatter(
        x=[0.5]*n, y=y, mode="text",
        text=[f"<b>{dates[i]}</b><br>{short_events[i]}" for i in range(n)],
        textposition="middle left",
        textfont=dict(size=12),
        hoverinfo="skip"
    ))
    # invisible hover layer with full text & sources
    hover_texts = []
    for i in range(n):
        src = timeline[i].get("source") or ""
        hover_texts.append(f"<b>{i+1}. {timeline[i].get('event')}</b><br><b>Date:</b> {dates[i]}<br><b>Source:</b> {src}<br><br>{timeline[i].get('summary','')[:800]}")
    fig.add_trace(go.Scatter(
        x=[0.5]*n, y=y, mode="markers",
        marker=dict(size=28, color="rgba(0,0,0,0)"),
        hoverinfo="text",
        hovertext=hover_texts,
        showlegend=False
    ))

    fig.update_layout(
        title=title,
        height=max(600, n * 50),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=60, r=60, t=80, b=60),
        plot_bgcolor="white",
        showlegend=False
    )
    return fig

# -------------------------
# Translation helper 
# -------------------------
def translate_openai_safe(text, lang):
    if not OPENAI_KEY:
        return "[OpenAI key missing]"
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
            json={
                "model":"gpt-4o-mini",
                "messages":[{"role":"user","content":f"Translate the following into {lang} (concise):\n\n{text}"}],
                "max_tokens": 800
            },
            timeout=25
        )
        data = r.json()
        return data.get("choices",[{}])[0].get("message",{}).get("content","[translation failed]")
    except Exception as e:
        return f"[translation error] {e}"

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI News Orchestrator", layout="wide")
st.markdown("<h1>🧭 AI News Orchestrator </h1>", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    topic = st.text_input("Topic / Event", value="Chandrayaan-3")
    region = st.selectbox("Region", ["India","World"])
    max_items = st.slider("Max total articles to aggregate", 10, 200, 100)
    
    translate_langs = st.multiselect("Translate combined summary to", ["Tamil","Hindi","French","Spanish"])
    run_button = st.button("🔎 Build news")



# Action
if run_button:
    region_code = "IN" if region=="India" else "US"
    started = datetime.utcnow()
    with st.spinner("Gathering articles (this may take a few seconds)..."):
        # 1) gather articles (NewsAPI paginated + RSS variations)
        raw = gather_articles(topic, max_items=max_items, region=region_code)
        # 2) build sorted timeline (oldest -> newest)
        timeline = build_timeline_from_entries(raw)
        # 3) combined summary (simple concatenation; can plug in summarizer)
        combined_summary = " ".join([t.get("summary","") for t in timeline if t.get("summary")])
        # 4) credibility overview (domain count & simple weight)
        domains = [t.get("source") for t in raw if t.get("source")]
        domain_counts = {}
        for d in domains:
            domain_counts[d] = domain_counts.get(d, 0) + 1

    st.subheader("📝 Combined summary (extract) ")
    if combined_summary.strip():
        st.write(combined_summary)
    else:
        st.write("No summary text available from fetched items.")

    if translate_langs and combined_summary.strip():
        st.subheader("🌍 Translations")
        for lg in translate_langs:
            with st.expander(f"Translation → {lg}"):
                st.write(translate_openai_safe(combined_summary, lg))

    st.subheader("📌 Timeline (chronological — oldest → newest)")
    # small list display
    for idx, item in enumerate(timeline, start=1):
        st.markdown(f"### {idx}. {item.get('date') or 'Unknown date'}")
        st.write(f"**[{item.get('event')}]({item.get('url')})**")
        if item.get("source"):
            st.caption(f"Source: {item.get('source')}")
        if item.get("summary"):
            st.write(item.get("summary"))
        st.markdown("---")

    st.subheader("📊 Vertical Timeline (interactive)")
    fig = vertical_timeline_plotly(timeline, title=f"Timeline — {topic}")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔍 Source overview")
    if domain_counts:
        for d, n in sorted(domain_counts.items(), key=lambda x: -x[1])[:40]:
            st.write(f"• `{d}` — {n} articles")
    else:
        st.write("No source domains found.")

    st.subheader("⬇ Exports")
    csv_buf = StringIO()
    w = csv.writer(csv_buf)
    w.writerow(["index","date","event","summary","url","source"])
    for i, t in enumerate(timeline, start=1):
        w.writerow([i, t.get("date",""), t.get("event",""), t.get("summary",""), t.get("url",""), t.get("source","")])
    st.download_button("Download CSV", data=csv_buf.getvalue(), file_name="timeline.csv", mime="text/csv")
    st.download_button("Download JSON", data=json.dumps(timeline, ensure_ascii=False, indent=2), file_name="timeline.json", mime="application/json")
# End of file
