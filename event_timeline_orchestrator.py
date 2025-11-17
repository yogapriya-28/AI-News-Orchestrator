# news_orchestrator_competition_pro.py
"""
AI News Orchestrator — Competition PRO (single-file Streamlit)
- Aggregation: NewsAPI primary + Google News RSS fallback (real URL extraction)
- Timeline: dedupe, chronological milestones
- Analysis: NER (spaCy), fact extraction, conflicts, authenticity scoring
- Advanced Analysis (Pro): Verified facts, source grades, bias/clickbait, story reconstruction
- Multilingual summary: includes Tamil ('ta')
- Exports: CSV/JSON/PNG; history via SQLite
Notes:
 - Install optional libs for best experience: spaCy, en_core_web_sm, langdetect, googletrans, kaleido (for PNG)
 - Typical install:
    pip install streamlit feedparser requests plotly python-dotenv nltk spacy langdetect googletrans==4.0.0rc1 tldextract dateparser
    python -m spacy download en_core_web_sm
    pip install kaleido
"""
# === Stdlib ===
import os, re, json, sqlite3, math, urllib.parse, csv, sys
from io import StringIO, BytesIO
from datetime import datetime, date
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from textwrap import shorten

# === Third-party ===
import streamlit as st
import feedparser
import requests
import plotly.graph_objects as go
from dotenv import load_dotenv

# Optional libraries (soft)
try:
    import dateparser
    from dateparser.search import search_dates
except Exception:
    dateparser = None
    def search_dates(*a, **kw): return None

try:
    import tldextract
except Exception:
    tldextract = None

# Sentiment (VADER)
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except Exception:
        nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
except Exception:
    sia = None

# spaCy NER
SPACY_AVAILABLE = False
nlp = None
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        try:
            # Attempt to download if missing
            from spacy.cli import download as spacy_download
            spacy_download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = None
    SPACY_AVAILABLE = nlp is not None
except Exception:
    nlp = None
    SPACY_AVAILABLE = False

# Translation/lang detect
try:
    from langdetect import detect as detect_lang
except Exception:
    detect_lang = None
try:
    from googletrans import Translator
    translator = Translator()
except Exception:
    translator = None

# OpenAI modern client (optional)
openai_client = None
try:
    from openai import OpenAI
    load_dotenv()
    if os.getenv("OPENAI_API_KEY"):
        openai_client = OpenAI()
except Exception:
    openai_client = None

# Load NEWS API key from .env if present
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# -------------------------
# Config & weights
# -------------------------
MAX_ARTICLES_CAP = 60
DEFAULT_HEADERS = {"User-Agent":"AI-News-Orchestrator/Pro/1.0"}
BANNED_SOURCE_SUBSTRINGS = ["affairscloud","jagranjosh","insightsias","utkarsh","vajiram","gktoday","currentaffairs","affairsadda","clearias","studybulletin"]
DOMAIN_WEIGHTS = {
    "reuters":3,"bbc":3,"nytimes":3,"washingtonpost":3,"thehindu":2,"timesofindia":2,
    "economictimes":2,"mint":2,"ndtv":1.5,"bloomberg":3,"indianexpress":2,"hindustantimes":2
}
EVENT_KEYWORDS = ["launch","launched","announce","announc","land","arriv","arrived","success","successful",
                  "confirmed","mission","docking","crash","protest","strike","meeting","summit","election",
                  "vote","inaugur","agreement","deal","deadline","flyby","orbit","propulsion","module","resign",
                  "arrest","charged"]

# Sensational words for misinfo scoring
SENSATIONAL_WORDS = set(["breaking","exclusive","shocking","miracle","viral","you won't believe","secret","scandal","exposed"])

# -------------------------
# Helpers
# -------------------------
def clean_html(text):
    if not text: return ""
    return re.sub(r"\s+"," ", re.sub(r"<[^>]+>", " ", str(text))).strip()

def short(text, width=150): return shorten(text or "", width=width, placeholder="...")

def ultra_short(text, max_words=7):
    if not text: return ""
    t = re.sub(r"https?://\S+","", text)
    t = re.sub(r"[^A-Za-z0-9\s]"," ", t)
    words = [w for w in t.split() if w.strip()]
    if not words: return ""
    return shorten(" ".join(words[:max_words]), width=60, placeholder="...")

def safe_get(url, params=None, timeout=10):
    try:
        r = requests.get(url, params=params, timeout=timeout, headers=DEFAULT_HEADERS)
        r.raise_for_status(); return r
    except Exception:
        return None

def extract_domain(url):
    if not url: return ""
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc.lower().replace("www.","")
        if tldextract:
            te = tldextract.extract(host)
            reg = ".".join([p for p in (te.domain, te.suffix) if p])
            return reg.lower()
        return host.split(":")[0]
    except Exception:
        return ""

def parse_iso_or_fallback(s):
    if not s: return None
    s = str(s).strip()
    try:
        return datetime.fromisoformat(s.replace("Z","+00:00")).date()
    except Exception: pass
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(s)
        return dt.date()
    except Exception: pass
    if dateparser:
        try:
            dt = dateparser.parse(s)
            if dt: return dt.date()
        except Exception: pass
    m = re.search(r"\b(20\d{2}|19\d{2})\b", s)
    if m:
        try: return date(int(m.group(1)),1,1)
        except: pass
    return None

def normalize_text(s):
    if not s: return ""
    s2 = s.lower(); s2 = re.sub(r"https?://\S+","", s2); s2 = re.sub(r"[^a-z0-9\s]"," ", s2)
    return re.sub(r"\s+"," ", s2).strip()

def similarity(a,b): 
    try:
        return SequenceMatcher(None,a,b).ratio()
    except Exception:
        return 0.0

def authenticity_score(domains):
    doms = [d for d in domains if d]
    if not doms: return 0.0
    uniq = list(dict.fromkeys(doms))
    total = 0.0
    max_w = max(DOMAIN_WEIGHTS.values()) if DOMAIN_WEIGHTS else 1.0
    for d in uniq:
        w = 0.0
        for k,v in DOMAIN_WEIGHTS.items():
            if k in d:
                w = v; break
        if w==0.0: w = 0.8
        total += w
    score = 100.0 * (total / (max_w * len(uniq)))
    return round(min(score,100.0),1)

# Clickbait & subjectivity heuristics
CLICKBAIT_PATTERNS = [
    r"you won't believe", r"won't believe", r"what happened next",
    r"top \d+", r"shocking", r"revealed", r"this is what", r"the reason why",
    r"can't miss", r"exclusive", r"breaking"
]
def headline_clickbait_score(title):
    if not title: return 0.0
    t = title.lower(); score=0.0
    if re.search(r'\b\d{1,2}\b', t) and re.search(r'\btop\b', t): score+=0.25
    for p in CLICKBAIT_PATTERNS:
        if re.search(p, t): score+=0.3
    if t.isupper(): score+=0.2
    return min(1.0, score)

SUBJECTIVE_WORDS = set(["believe","feel","think","opinion","suggest","claim","allege","apparently","reportedly","seems","likely","possibly","probably"])
def subjectivity_score(text):
    if not text: return 0.0
    words = re.findall(r"\w+", str(text).lower())
    if not words: return 0.0
    subj = sum(1 for w in words if w in SUBJECTIVE_WORDS)
    # scale by text length
    return min(1.0, subj / max(1, len(words)/100))

# -------------------------
# Fetchers (NewsAPI + Google RSS)
# -------------------------
@st.cache_data(ttl=900)
def fetch_newsapi_primary(topic, max_items=20, language='en', q_in_title=True):
    if not NEWS_API_KEY: return []
    url = "https://newsapi.org/v2/everything"
    params = {"language":language,"pageSize":max_items,"sortBy":"publishedAt","apiKey":NEWS_API_KEY}
    if q_in_title: params["qInTitle"] = topic
    else: params["q"] = topic
    r = safe_get(url, params=params, timeout=12)
    if not r: return []
    try: data = r.json()
    except: return []
    out=[]
    for a in data.get("articles",[]):
        out.append({
            "title":a.get("title",""),
            "summary":clean_html(a.get("description") or a.get("content","")),
            "url":a.get("url",""),
            "published":a.get("publishedAt",""),
            "source_url":a.get("url","")
        })
    return out

@st.cache_data(ttl=900)
def fetch_google_rss(query, max_items=20, country='IN'):
    enc = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={enc}&hl=en-{country}&gl={country}&ceid={country}:en"
    try:
        feed = feedparser.parse(rss_url)
    except Exception:
        return []
    entries = getattr(feed,"entries",[]) or []
    out=[]; seen=set()
    for e in entries[:max_items*3]:
        title = e.get("title",""); link = e.get("link","")
        # extract real URL if google redirect
        if link and "news.google" in link and "url=" in link:
            m = re.search(r"[?&]url=([^&]+)", link)
            if m: link = urllib.parse.unquote(m.group(1))
        link = link.replace("amp;","")
        summary = clean_html(e.get("summary") or e.get("description",""))
        published = e.get("published","") or e.get("pubDate","")
        source_url = link
        key = (title,link)
        if key in seen: continue
        seen.add(key)
        out.append({
            "title":title,
            "summary":summary,
            "url":link,
            "published":published,
            "source_url":source_url
        })
        if len(out) >= max_items: break
    return out

# -------------------------
# Event extraction, dedupe, claims
# -------------------------
def sentence_split(text):
    if not text: return []
    parts = re.split(r'(?<=[\.\?\!])\s+', str(text))
    return [p.strip() for p in parts if p.strip()]

def sentence_has_event_keywords(s):
    s = (s or "").lower(); return any(k in s for k in EVENT_KEYWORDS)

def find_dates_in_text(text):
    if not text: return []
    t = re.sub(r"\s+"," ", str(text))
    if dateparser:
        try:
            res = search_dates(t, languages=['en'])
            if res:
                out=[]
                for _,dt in res:
                    try: out.append(dt.date())
                    except: pass
                if out:
                    uniq=[]
                    for d in out:
                        if d not in uniq: uniq.append(d)
                    return uniq
        except Exception: pass
    years = re.findall(r"\b(20\d{2}|19\d{2})\b", t)
    if years:
        uniq=[]
        for y in years:
            d=date(int(y),1,1)
            if d not in uniq: uniq.append(d)
        return uniq
    return []

def build_events_from_article(article):
    title = article.get("title","") or ""; summary = article.get("summary","") or ""
    url = article.get("url","") or ""; published = article.get("published","") or ""
    source_url = article.get("source_url","") or ""
    text_blob = " ".join([title, summary])
    sentences = sentence_split(text_blob)
    events=[]
    for s in sentences:
        dates = find_dates_in_text(s)
        if dates:
            for d in dates:
                events.append({"date":d,"text":s,"source":source_url or url,"title":title})
        else:
            if sentence_has_event_keywords(s):
                dval = parse_iso_or_fallback(published)
                events.append({"date":dval,"text":s,"source":source_url or url,"title":title})
    parsed_pub = parse_iso_or_fallback(published)
    if parsed_pub:
        events.append({
            "date":parsed_pub,
            "text":f"Article published: {title}",
            "source":source_url or url,
            "title":title
        })
    return events

def dedupe_merge_events(events, similarity_threshold=0.85):
    merged=[]
    for ev in events:
        text_norm = normalize_text(ev.get("text",""))
        if not text_norm: continue
        placed=False
        for m in merged:
            score = similarity(text_norm, m["norm"])
            if score >= similarity_threshold:
                # Merge logic
                if ev.get("date") and (not m["date"] or ev.get("date") < m["date"]):
                    m["date"] = ev.get("date")
                if ev.get("text") and ev.get("text") not in m["texts"]:
                    m["texts"].append(ev.get("text"))
                if ev.get("source") and ev.get("source") not in m["sources"]:
                    m["sources"].append(ev.get("source"))
                if ev.get("title") and ev.get("title") not in m["titles"]:
                    m["titles"].append(ev.get("title"))
                if len(text_norm) > len(m["norm"]):
                    m["norm"] = text_norm
                placed=True
                break
        if not placed:
            merged.append({
                "date":ev.get("date"),
                "texts":[ev.get("text")],
                "sources":[ev.get("source")] if ev.get("source") else [],
                "titles":[ev.get("title")] if ev.get("title") else [],
                "norm":text_norm
            })
    out=[]
    for m in merged:
        rep = max(m["texts"], key=lambda x: len(x)) if m["texts"] else ''
        out.append({
            "date": m["date"],
            "event": rep,
            "sources": m["sources"],
            "titles": m["titles"]
        })
    def keyf(it):
        if it["date"]: return (0, it["date"])
        return (1, date(9999,1,1))
    return sorted(out, key=keyf)

# Claims & conflict detection
def extract_claims_from_article(article):
    title = article.get("title","") or ""; summary = article.get("summary","") or ""
    text_blob = " ".join([title, summary]); claims=[]
    sents = sentence_split(text_blob)
    for s in sents:
        if re.search(r"\b(20\d{2}|19\d{2})\b", s) or re.search(r"\b\d[\d,\.]*\b", s) or sentence_has_event_keywords(s):
            nums = re.findall(r"\b\d[\d,\,\.]*\b", s)
            dates = find_dates_in_text(s)
            claims.append({
                "text": s,
                "nums": nums,
                "dates": dates,
                "subject": ultra_short(s,6),
                "predicate": short(s,160),
                "source": article.get("source_url") or article.get("url")
            })
    return claims

def detect_conflicts_across_claims(claims_list):
    groups = defaultdict(list)
    for c in claims_list:
        key = normalize_text(c.get("predicate",""))[:120]
        groups[key].append(c)
    conflicts=[]
    for k, items in groups.items():
        if len(items) < 2: continue
        nums_sets = [set(i.get("nums") or []) for i in items]
        date_sets = [set([d.isoformat() for d in i.get("dates")]) for i in items]
        flattened_nums = set.union(*[s for s in nums_sets if s]) if any(nums_sets) else set()
        if len(flattened_nums) > 1:
            conflicts.append({
                "predicate": items[0].get("predicate"),
                "type":"number_mismatch",
                "values": list(flattened_nums),
                "evidence": items
            })
        flattened_dates = set.union(*[s for s in date_sets if s]) if any(date_sets) else set()
        if len(flattened_dates) > 1:
            conflicts.append({
                "predicate": items[0].get("predicate"),
                "type":"date_mismatch",
                "values": list(flattened_dates),
                "evidence": items
            })
    return conflicts

# NER & Verified facts
def extract_entities(text):
    ents=[]
    if SPACY_AVAILABLE and nlp:
        try:
            doc = nlp(text)
            for e in doc.ents: ents.append((e.text, e.label_))
        except Exception: pass
    else:
        dates = find_dates_in_text(text)
        numbers = re.findall(r"\b\d[\d,\.]*\b", text)
        ents = [("DATE:"+str(d),"DATE") for d in dates] + [("NUMBER:"+n,"NUMBER") for n in numbers]
    return ents

def verified_facts_from_entries(entries):
    facts = []
    for e in entries:
        text = (e.get("title","") or "") + ". " + (e.get("summary","") or "")
        ents = extract_entities(text)
        for ent, label in ents:
            facts.append({"fact": ent, "label": label, "source": e.get("source_url") or e.get("url")})
    # dedupe facts
    uniq = []
    seen = set()
    for f in facts:
        key = (f['fact'], f['label'])
        if key in seen: continue
        seen.add(key); uniq.append(f)
    return uniq

# OpenAI helpers
def openai_score_relevance(item, topic, openai_client_local=None):
    text = (item.get("title","") or "") + "\n\n" + (item.get("summary","") or "")
    if topic.lower() in (item.get("title","") or "").lower(): return 0.95
    if not openai_client_local:
        words = re.findall(r"\w+", text.lower()); topic_words = re.findall(r"\w+", topic.lower())
        if not topic_words: return 0.0
        # simple lexical overlap
        return min(1.0, sum(1 for w in words if w in topic_words) / max(1, len(topic_words)))
    try:
        prompt = f"Rate relevance of this article to '{topic}'. Return a number 0-1.\nTitle:{item.get('title','')}\nSummary:{item.get('summary','')}\n\nNumber:"
        resp = openai_client_local.responses.create(model="gpt-4o-mini", input=prompt, max_output_tokens=30)
        text_out = ''
        if isinstance(resp, dict): text_out = resp.get('output_text') or resp.get('text') or str(resp)
        else: text_out = getattr(resp,'output_text',None) or getattr(resp,'text',None) or str(resp)
        m = re.search(r"([01](?:\.\d+)?)", text_out)
        if m: return float(m.group(1))
    except Exception: pass
    return 0.0

def openai_refine_timeline(timeline_items, model="gpt-5-mini", top_n=30):
    if not openai_client: return None
    items = timeline_items[:top_n]; lines=[]
    for it in items:
        date_str = it.get("date") or "DATE_UNKNOWN"
        evt = short(it.get("event",""),200)
        srcs = ", ".join([extract_domain(s) for s in it.get("sources",[])][:3])
        lines.append(f"- {date_str} : {evt} (sources: {srcs})")
    prompt = "Return only JSON: {timeline:[{date,event,sources}], narrative: '...'}\n\nItems:\n" + "\n".join(lines)
    try:
        resp = openai_client.responses.create(model=model, input=prompt, max_output_tokens=800, temperature=0.2)
        text = ''
        if isinstance(resp, dict): text = resp.get('output_text') or resp.get('text') or str(resp)
        else: text = getattr(resp,'output_text',None) or getattr(resp,'text',None) or str(resp)
        m = re.search(r"\{[\s\S]*\}", text)
        if m: return json.loads(m.group(0))
    except Exception: pass
    return None

# Multilingual summarization
def detect_language_safe(text):
    if not text: return None
    if detect_lang:
        try: return detect_lang(text)
        except: return None
    return None

def translate_text(text, dest='en'):
    if not text: return text, None
    if translator:
        try:
            res = translator.translate(text, dest=dest); return res.text, getattr(res,'src',None)
        except: return text, None
    return text, None

def summarize_multilang(text, target_lang='en', use_openai_client=None, max_sentences=3):
    if not text: return {"summary":"No text","source_lang":None,"used_translation":False}
    src_lang = detect_language_safe(text) if detect_lang else None
    used_translation=False; working=text
    if src_lang and target_lang and src_lang[:2] != target_lang[:2] and translator:
        try:
            working, detected = translate_text(text, dest=target_lang)
            used_translation=True
        except:
            working=text; used_translation=False
    if use_openai_client:
        try:
            prompt = f"Summarize in {target_lang}. Provide 3 key milestones and a 2-line narrative.\n\nText:\n{working[:7000]}"
            resp = use_openai_client.responses.create(model="gpt-4o-mini", input=prompt, max_output_tokens=600)
            summ = ''
            if isinstance(resp, dict): summ = resp.get('output_text') or resp.get('text') or str(resp)
            else: summ = getattr(resp,'output_text',None) or getattr(resp,'text',None) or str(resp)
            return {"summary":summ, "source_lang":src_lang, "used_translation":used_translation}
        except Exception: pass
    # fallback extractive
    sents = sentence_split(clean_html(working))
    if len(sents) <= max_sentences:
        return {"summary": " ".join(sents), "source_lang": src_lang, "used_translation": used_translation}
    words = [w.lower() for w in re.findall(r"\w+", working)]; freq = Counter(words); scores=[]
    for i,s in enumerate(sents):
        swords = [w.lower() for w in re.findall(r"\w+", s)]
        score = sum(freq.get(w,0) for w in swords) + len(s)/10.0
        scores.append((score,i))
    scores.sort(reverse=True); idxs = sorted([i for _,i in scores[:max_sentences]])
    return {"summary": " ".join(sents[i] for i in idxs), "source_lang": src_lang, "used_translation": used_translation}

# -------------------------
# DB history
# -------------------------
DB_FILE = "timeline_history.db"
def init_db():
    conn = sqlite3.connect(DB_FILE); cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS timelines (id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT, created_at TEXT, data JSON)")
    conn.commit(); conn.close()

def save_timeline_to_db(topic, payload):
    conn = sqlite3.connect(DB_FILE); cur = conn.cursor()
    cur.execute("INSERT INTO timelines (topic, created_at, data) VALUES (?, ?, ?)", (topic, datetime.utcnow().isoformat(), json.dumps(payload)))
    conn.commit(); conn.close()

def list_timeline_history(limit=20):
    conn = sqlite3.connect(DB_FILE); cur = conn.cursor()
    cur.execute("SELECT id, topic, created_at FROM timelines ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall(); conn.close(); return rows

def load_timeline_from_db(id_):
    conn = sqlite3.connect(DB_FILE); cur = conn.cursor()
    cur.execute("SELECT data FROM timelines WHERE id = ?", (id_,))
    r = cur.fetchone(); conn.close()
    if r: return json.loads(r[0]); return None

# -------------------------
# Pipeline: build_timeline (with enhanced scoring)
# -------------------------
def build_query_variations(topic, location="", year=None):
    q = topic.strip(); parts=[q, f"{q} latest", f"{q} news", f"{q} timeline"]
    if location: parts.append(f"{q} {location} news")
    if year: parts.append(f"{q} {year} updates")
    seen=set(); out=[]
    for p in parts:
        if p and p not in seen: seen.add(p); out.append(p)
    return out

def compute_source_polarization(domains):
    # Rudimentary: if many known left/right domains exist we'd tag — placeholder for richer mapping
    # For now, polarization is measured as fraction of unknown/low-weight domains
    if not domains: return 0.0
    weights = []
    for d in domains:
        w = 0.0
        for k,v in DOMAIN_WEIGHTS.items():
            if k in d:
                w = v; break
        if w == 0.0: w = 0.8
        weights.append(w)
    # if many low weights -> higher polarization risk (because unknown niche / partisan sources)
    low_count = sum(1 for w in weights if w < 1.5)
    return round(min(1.0, low_count / max(1, len(weights))), 3)

def build_timeline(topic, location_keyword="", max_articles=20, country_code='IN', world=False, use_newsapi=True, use_openai_ranking=True, dedupe_sim=0.85):
    if not topic or len(topic.strip())<3:
        return {"entries":[],"timeline":[],"summary":"Please enter a specific topic (>=3 chars).","auth_score":0.0,"domains":[],"claims":[],"conflicts":[]}
    max_articles = int(min(max(max_articles,5), MAX_ARTICLES_CAP))
    queries = build_query_variations(topic, location_keyword)
    entries=[]
    # NewsAPI primary
    if use_newsapi and NEWS_API_KEY:
        entries = fetch_newsapi_primary(topic, max_items=max_articles, language='en', q_in_title=True)
    # fallback google rss
    if len(entries) < max_articles:
        for q in queries:
            if len(entries) >= max_articles: break
            fetched = fetch_google_rss(q, max_items=max_articles, country=country_code)
            for a in fetched:
                if len(entries) >= max_articles: break
                domain_candidate = extract_domain(a.get("source_url") or a.get("url") or "")
                if any(b in (domain_candidate or "") for b in BANNED_SOURCE_SUBSTRINGS): continue
                if any(e.get('url') == a.get('url') for e in entries): continue
                entries.append(a)
    entries = entries[:max_articles]
    domains=[]
    scored=[]
    for e in entries:
        d = extract_domain(e.get("source_url") or e.get("url") or ""); domains.append(d)
        cb = headline_clickbait_score(e.get("title","")); subj = subjectivity_score(e.get("summary",""))
        rel = openai_score_relevance(e, topic, openai_client_local=openai_client if (use_openai_ranking and openai_client) else None)
        scored.append((rel, e, {"clickbait": cb, "subjectivity": subj}))
    scored = sorted(scored, key=lambda x: x[0], reverse=True)
    filtered = [e for r,e,meta in scored if r >= 0.12]
    if not filtered and scored: filtered=[scored[0][1]]
    entries = filtered[:max_articles]
    all_events=[]; claims_all=[]
    for e in entries:
        evs = build_events_from_article(e); all_events.extend(evs)
        claims = extract_claims_from_article(e); claims_all.extend(claims)
    timeline_items = dedupe_merge_events(all_events, similarity_threshold=dedupe_sim)

    # Precompute domain auth and polarization
    overall_auth = authenticity_score(domains)
    polarization = compute_source_polarization(domains)

    for it in timeline_items:
        # ---------- sentiment ----------
        txt = it.get("event","") or ""
        s = {"neg":0.0,"neu":1.0,"pos":0.0,"compound":0.0}
        if sia:
            try: s = sia.polarity_scores(txt)
            except: pass
        it["sentiment"] = s

        # ---------- clickbait & subjectivity (enhanced) ----------
        cb_scores = []
        sub_scores = []

        # 1) inherit scores from article-level metadata
        for src in it.get("sources", []):
            for r,e,meta in scored:
                if e.get("source_url") == src or e.get("url") == src:
                    cb_scores.append(meta.get("clickbait",0.0))
                    sub_scores.append(meta.get("subjectivity",0.0))

        # 2) additional scoring from event text itself
        event_text = it.get("event","") or ""
        lower_evt = event_text.lower()

        # clickbait boosters
        if re.search(r"(breaking|exclusive|revealed|shocking|viral|you won't believe|must read)", lower_evt):
            cb_scores.append(0.35)
        if re.search(r"\btop\s+\d+\b", lower_evt):
            cb_scores.append(0.25)
        if len(event_text) > 120:
            cb_scores.append(0.1)

        # subjectivity boosters
        subjective_words = ["likely", "possibly", "reportedly", "allegedly", "claims", "appears", "seems", "suggests"]
        if any(w in lower_evt for w in subjective_words):
            sub_scores.append(0.35)

        # ensure baseline
        if not cb_scores: cb_scores = [0.05]
        if not sub_scores: sub_scores = [0.03]

        it["clickbait_score"] = round(sum(cb_scores)/len(cb_scores), 3)
        it["subjectivity_score"] = round(sum(sub_scores)/len(sub_scores), 3)

        # ---------- misinfo risk ----------
        # Weighted sum heuristic using clickbait, subjectivity, domain trust and sensational words, conflicts
        mis = 0.0
        mis += 0.25 * it["clickbait_score"]
        mis += 0.30 * it["subjectivity_score"]
        # domain risk = inverse of normalized auth (auth is 0-100)
        domain_risk = 1.0 - (overall_auth/100.0)  # 0 good -> 1 bad
        mis += 0.35 * domain_risk
        # sensational words in text
        sens_count = sum(1 for w in SENSATIONAL_WORDS if w in lower_evt)
        if sens_count:
            mis += min(0.4, 0.2 * sens_count)
        # conflict presence (simple check against claims list)
        # if any claim in this event predicate matches a detected conflict, boost risk
        mis = min(1.0, mis)
        it["misinfo_risk"] = round(mis, 3)

        # ---------- bias score ----------
        # bias heuristics: negative sentiment + politically loaded words + subjectivity
        bias = 0.0
        # negative sentiment compound < -0.25 increases bias risk
        try:
            comp = float(it.get("sentiment",{}).get("compound", 0.0))
        except Exception:
            comp = 0.0
        if comp < -0.25: bias += 0.25
        if comp > 0.25: bias += 0.05  # positive emotional spin
        pol_words = ["government", "minister", "election", "vote", "scandal", "corrupt", "policy", "protest", "opposition"]
        if any(w in lower_evt for w in pol_words):
            bias += 0.25
        bias += 0.3 * it["subjectivity_score"]
        it["bias_score"] = round(min(1.0, bias), 3)

        # ---------- source polarization ----------
        it["source_polarization"] = round(polarization, 3)

    # convert dates to iso
    for it in timeline_items:
        if isinstance(it.get("date"), date): it["date"] = it["date"].isoformat()
        elif it.get("date") is None: it["date"] = ""

    big_text = " ".join([e.get("title","")+" "+e.get("summary","") for e in entries])
    combined_summary = short(big_text, 800) if not openai_client else summarize_multilang(big_text, target_lang='en', use_openai_client=None).get("summary")
    conflicts = detect_conflicts_across_claims(claims_all)
    return {"entries": entries, "timeline": timeline_items, "summary": combined_summary, "auth_score": overall_auth, "domains": domains, "claims": claims_all, "conflicts": conflicts, "raw_scored": scored}

# -------------------------
# Rendering & UI helpers
# -------------------------
def detect_category(event_text):
    t = (event_text or "").lower()
    cats = {
        "space":["isro","nasa","chandrayaan","launch","orbit","mission"],
        "ai":["ai","openai","gpt","machine learning"],
        "politics":["election","minister","government","vote","opposition"],
        "science":["research","study","paper","experiment"],
        "business":["stock","market","company","acquire","acquisition","merger"]
    }
    for cat,kws in cats.items():
        if any(k in t for k in kws): return cat
    return "other"

def grade_from_weight(w):
    if w >= 3.0: return "A+"
    if w >= 2.5: return "A"
    if w >= 1.5: return "B"
    if w >= 0.8: return "C"
    return "D"

def render_authenticity_table(domains):
    dom_counts = Counter(domains)
    rows = []
    for d, c in dom_counts.most_common():
        w = 0.0
        for k,v in DOMAIN_WEIGHTS.items():
            if k in d: w = v; break
        if w==0.0: w = 0.8
        grade = grade_from_weight(w)
        note = "Trusted" if w>=2 else ("Unknown" if w==0.8 else "Mixed")
        rows.append({"domain":d, "count":c, "weight":w, "grade":grade, "note":note})
    return rows

def render_vertical_timeline(timeline_list, title="Vertical Timeline"):
    if not timeline_list:
        st.info("No timeline items"); return None
    processed=[]
    for it in timeline_list:
        d = it.get("date") or ""
        try: dt = datetime.fromisoformat(d).date() if d else None
        except: dt = None
        processed.append((dt, it.get("event",""), it.get("sources",[]), it.get("clickbait_score",0.0), it.get("misinfo_risk",0.0)))
    processed = sorted(processed, key=lambda x: (x[0] is None, x[0] or date(9999,1,1)))
    n=len(processed); y_positions=list(range(n))[::-1]
    dates_text=[(p[0].isoformat() if p[0] else "Unknown") for p in processed]
    events_text=[p[1] for p in processed]; sources_list=[p[2] for p in processed]
    icons=[]; colors=[]; border=[]
    for ev,cb,mis in zip(events_text, [p[3] for p in processed], [p[4] for p in processed]):
        cat = detect_category(ev)
        icons.append({"space":"🛰️","ai":"🤖","politics":"🗳️","science":"🔬","business":"💼"}.get(cat,"📌"))
        colors.append({"space":"#0057ff","ai":"#7d00ff","politics":"#ff3b30","science":"#00b894","business":"#ffa500"}.get(cat,"#444"))
        # border color stronger if misinfo risk high
        border.append("#ff3b30" if mis>0.35 else "#999")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0]*n, y=y_positions, mode="lines", line=dict(color="#e6e6e6", width=4), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0]*n, y=y_positions, mode="markers+text", marker=dict(size=36, color=colors, line=dict(width=4, color=border)), text=icons, textposition="middle center", hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[-0.6]*n, y=y_positions, mode="text", text=dates_text, textposition="middle right", textfont=dict(size=12, color="#333")))
    right_texts=[f"<b>{i+1}.</b> {ultra_short(events_text[i], max_words=8)}" for i in range(n)]
    fig.add_trace(go.Scatter(x=[0.6]*n, y=y_positions, mode="text", text=right_texts, textposition="middle left", textfont=dict(size=14)))
    hover_texts=[]
    for i in range(n):
        ev_full = events_text[i] or ""
        src_prev = "<br>".join((sources_list[i] or [])[:3]) or "No sources"
        hover_texts.append(f"<b>{i+1}. {events_text[i]}</b><br><b>Date:</b> {dates_text[i]}<br><b>Sources:</b><br>{src_prev}")
    fig.add_trace(go.Scatter(x=[0.6]*n, y=y_positions, mode="markers", marker=dict(size=36, color="rgba(0,0,0,0)"), hoverinfo="text", hovertext=hover_texts, showlegend=False))
    fig.update_layout(title=title, height=max(480, n*90), margin=dict(l=80,r=80,t=80,b=80), xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
    return fig

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI News Orchestrator — PRO", layout="wide")
st.markdown("<h1>🧭 AI News Orchestrator </h1>", unsafe_allow_html=True)
init_db()

page = st.sidebar.selectbox("Page", ["Orchestrator","Advanced Analysis (Pro)","Vertical Timeline","History","About"])

if page == "Orchestrator":
    leftcol, rightcol = st.columns([3,1])
    topic = leftcol.text_input("Event / Topic (e.g., Chandrayaan-3, COP28)", value="")
    scope = rightcol.selectbox("Scope", ["India","World"])
    if scope == "India":
        location = leftcol.selectbox("State (or National)", ["National","Delhi","Maharashtra","Karnataka","Tamil Nadu","Kerala","Gujarat","West Bengal","Telangana","Bihar"])
        country_code = "IN"
    else:
        location = leftcol.selectbox("Country", ["United States","United Kingdom","India","UAE","France","Germany"])
        country_code = 'US' if location=="United States" else ("GB" if location=="United Kingdom" else "IN")
    c1,c2 = st.columns([1,1])
    with c1: max_articles = st.slider("Max articles to aggregate", 5, MAX_ARTICLES_CAP, 20)
    with c2: dedupe_sim = st.slider("Deduplication similarity (higher merges more)", 70, 95, 85)/100.0
    use_newsapi = st.checkbox("Use NewsAPI (recommended)", value=bool(NEWS_API_KEY))
    # Note: we renamed the boolean to `use_openai_ranking` for clarity against the client variable.
    use_openai_ranking = st.checkbox("Use OpenAI (ranking & refine)", value=bool(openai_client))
    # language choices show full words
    multi_lang = st.selectbox("Summary language", ["auto","english","hindi","tamil","spanish","french","arabic"], index=0)
    build_btn = st.button("🔎 Build Timeline")
    if build_btn:
        if not topic or len(topic.strip())<3: st.error("Enter topic (>=3 chars)"); st.stop()
        with st.spinner("Fetching & analyzing..."):
            data = build_timeline(topic, location if location!="National" else "", max_articles=max_articles, country_code=country_code, world=(scope=="World"), use_newsapi=use_newsapi, use_openai_ranking=use_openai_ranking, dedupe_sim=dedupe_sim)
            entries = data["entries"]; timeline = data["timeline"]
            summary = data["summary"]; auth = data["auth_score"]
            conflicts = data["conflicts"]
            st.session_state["last_timeline"] = timeline; st.session_state["last_entries"] = entries
            save_timeline_to_db(topic, {"topic":topic, "entries": entries, "timeline": timeline, "summary": summary, "auth": auth, "conflicts": conflicts})
        if not entries:
            st.error("No articles found. Try broader query or enable NewsAPI"); st.stop()
        st.success(f"Fetched {len(entries)} articles — extracted {len(timeline)} timeline items.")
        left, right = st.columns([3,1])
        with left:
            # multilingual summary mapping
            lang_map = {"auto":None, "english":"en","hindi":"hi","tamil":"ta","spanish":"es","french":"fr","arabic":"ar"}
            target_lang = lang_map.get(multi_lang, None)
            use_openai_now = use_openai_ranking and bool(openai_client)
            big_text = " ".join([e.get("title","")+" "+e.get("summary","") for e in entries])
            multi_res = summarize_multilang(big_text, target_lang=(target_lang or 'en'), use_openai_client=(openai_client if use_openai_now else None), max_sentences=3)
            st.subheader("Combined summary")
            if multi_res.get("used_translation"): st.caption(f"Source lang: {multi_res.get('source_lang') or 'unknown'} (translated)")
            st.write(multi_res.get("summary") or summary)
            st.markdown("---"); st.subheader("Timeline (chronological milestones)")
            for idx,it in enumerate(timeline, start=1):
                st.markdown(f"### {idx}. **{it.get('date') or 'Unknown date'}**")
                st.write(it.get("event"))
                if it.get("sources"):
                    st.markdown("**Sources:**")
                    for s in it.get("sources")[:4]:
                        st.write(f"- [{extract_domain(s)}]({s})")
                st.caption(f"Clickbait: {it.get('clickbait_score',0.0)} • Subjectivity: {it.get('subjectivity_score',0.0)} • Misinfo risk: {it.get('misinfo_risk',0.0)}")
                st.markdown("---")
        with right:
            st.subheader("Source Credibility (Authenticity)")
            st.metric("Authenticity score", f"{auth}%")
            auth_rows = render_authenticity_table(data["domains"])
            if auth_rows:
                for r in auth_rows[:12]:
                    color = "green" if r["grade"] in ("A+","A") else ("orange" if r["grade"]=="B" else "red")
                    st.write(f"- `{r['domain']}` — count: {r['count']} — weight: {r['weight']} — grade: {r['grade']} • {r['note']}")
            st.subheader("Conflicts (fact consistency)")
            if conflicts:
                for conf in conflicts:
                    st.warning(f"Conflict: {conf['type']} — {short(conf['predicate'],160)} — values: {conf['values'][:6]}")
                    for ev in conf['evidence'][:3]: st.write(f"- {short(ev.get('predicate',''),200)} — {ev.get('source')}")
            else:
                st.write("No obvious conflicts detected.")
        st.subheader("Visual timeline preview")
        fig = render_vertical_timeline(timeline, title=f"Timeline — {topic}")
        st.markdown("---"); st.subheader("Exports")
        csv_buf = StringIO(); writer = csv.writer(csv_buf); writer.writerow(["index","date","event","sources"])
        for i,it in enumerate(timeline, start=1): writer.writerow([i, it.get("date",""), it.get("event",""), "|".join(it.get("sources",[]))])
        st.download_button("⬇ Download timeline CSV", data=csv_buf.getvalue(), file_name="timeline.csv", mime="text/csv")
        st.download_button("⬇ Download timeline JSON", data=json.dumps({"topic":topic,"timeline":timeline}, ensure_ascii=False, indent=2), file_name="timeline.json", mime="application/json")
        if fig:
            try:
                img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
                st.download_button("⬇ Download timeline PNG", data=img_bytes, file_name="timeline.png", mime="image/png")
            except Exception:
                st.info("PNG export requires kaleido: pip install kaleido")

# --- Advanced Analysis page
elif page == "Advanced Analysis (Pro)":
    st.header("🧠 Advanced Analysis (Pro)")
    last_entries = st.session_state.get("last_entries") or []
    last_timeline = st.session_state.get("last_timeline") or []
    if not last_entries:
        st.info("No timeline in session. Build a timeline first on Orchestrator page.")
    else:
        st.subheader("Verified Facts (NER / heuristics)")
        facts = verified_facts_from_entries(last_entries)
        if facts:
            for f in facts[:80]:
                st.write(f"- **{f['fact']}** — {f['label']} — [{extract_domain(f['source'])}]({f['source']})")
        else:
            st.write("No verified facts extracted.")
        st.markdown("---")
        st.subheader("Source Credibility Table")
        doms = [extract_domain(e.get("source_url") or e.get("url") or "") for e in last_entries]
        auth_rows = render_authenticity_table(doms)
        st.write("Domain | Count | Weight | Grade | Note")
        for r in auth_rows:
            st.write(f"- `{r['domain']}` | {r['count']} | {r['weight']} | **{r['grade']}** | {r['note']}")
        st.markdown("---")
        st.subheader("Bias & Clickbait Overview")
        avg_click = sum(e.get("clickbait_score",0.0) for e in last_timeline)/max(1,len(last_timeline))
        avg_subj = sum(e.get("subjectivity_score",0.0) for e in last_timeline)/max(1,len(last_timeline))
        st.metric("Avg Clickbait score", f"{round(avg_click,3)}"); st.metric("Avg Subjectivity", f"{round(avg_subj,3)}")
        st.write("Explanation: clickbait heuristic checks titles for listicles, sensational words; subjectivity measures subjective words proportion.")
        st.markdown("---")
        st.subheader("Fact Consistency Checker")
        claims = []
        for e in last_entries: claims.extend(extract_claims_from_article(e))
        conflicts = detect_conflicts_across_claims(claims)
        if conflicts:
            st.warning(f"{len(conflicts)} potential conflicts detected.")
            for c in conflicts:
                st.write(f"• **{c['type']}** — {short(c['predicate'],200)} — values: {c['values'][:6]}")
                for ev in c['evidence'][:4]:
                    st.write(f"    - {short(ev.get('predicate',''),160)} — {ev.get('source')}")
        else:
            st.success("No clear conflicts found across extracted claims.")
        st.markdown("---")
        st.subheader("Story Reconstruction (AI-assisted)")
        use_openai_refine = st.checkbox("Use OpenAI for refined narrative (optional)", value=bool(openai_client))
        if use_openai_refine and openai_client:
            with st.spinner("Refining timeline with OpenAI..."):
                refined = openai_refine_timeline(last_timeline, model="gpt-5-mini")
                if refined:
                    st.markdown("**AI-refined narrative:**")
                    st.write(refined.get("narrative") or "")
                    st.markdown("**Refined timeline (preview):**")
                    for i,it in enumerate(refined.get("timeline",[])[:12], start=1):
                        st.markdown(f"**{i}. {it.get('date') or 'Unknown'}** — {short(it.get('event',''),220)}")
                else:
                    st.error("OpenAI refine failed.")
        else:
            st.markdown("**Reconstructed narrative (extractive)**")
            if last_timeline:
                first = last_timeline[0].get("event",""); last = last_timeline[-1].get("event","")
                mids = [it.get("event","") for it in last_timeline[1:4]]
                narr = []
                narr.append(short(f"Event begins: {ultra_short(first,8)}",300))
                for m in mids: narr.append(short(f"Key update: {ultra_short(m,10)}",200))
                narr.append(short(f"Latest: {ultra_short(last,10)}",300))
                st.write(" ".join(narr))
            else:
                st.write("No timeline to reconstruct from.")
        st.markdown("---")
        st.subheader("Multi-language summary")
        lang = st.selectbox("Choose summary language", ["en","hi","ta"], index=0)
        big_text = " ".join([e.get("title","")+" "+e.get("summary","") for e in last_entries])
        sum_res = summarize_multilang(big_text, target_lang=lang, use_openai_client=(openai_client if use_openai_refine else None))
        if sum_res.get("used_translation"): st.caption(f"Translated from {sum_res.get('source_lang')}")
        st.write(sum_res.get("summary") or "No summary available.")
        st.markdown("---")
        st.subheader("Export advanced report")
        report = {"topic_snapshot": last_entries[:20], "timeline": last_timeline, "facts": facts, "conflicts": conflicts, "auth": render_authenticity_table(doms)}
        st.download_button("⬇ Download advanced report (JSON)", data=json.dumps(report, ensure_ascii=False, indent=2), file_name="news_orchestrator_report.json", mime="application/json")

# --- Vertical Timeline page
elif page == "Vertical Timeline":
    st.header("📍 Vertical Timeline — Premium")
    session_tl = st.session_state.get("last_timeline")
    use_session = False
    if session_tl: use_session = st.checkbox("Use timeline generated in this session (recommended)", value=True)
    if use_session and session_tl:
        vertical_input = session_tl
    else:
        if os.path.exists("timeline.json"):
            try:
                with open("timeline.json","r",encoding="utf-8") as f: vertical_input = json.load(f)
                st.success("Loaded timeline.json")
            except Exception as e: st.error("Failed to load timeline.json"); st.write(e); vertical_input = []
        else:
            st.info("No timeline.json found. Build a timeline first."); vertical_input = []
    if vertical_input:
        st.subheader("Preview (first 10 items)")
        for i,it in enumerate(vertical_input[:10], start=1):
            st.markdown(f"**{i}. {it.get('date') or 'Unknown'}** — {short(it.get('event',''),220)}")
        render_vertical_timeline(vertical_input, title="Vertical Timeline (Premium)")

# --- History
elif page == "History":
    st.header("📚 Timeline History")
    rows = list_timeline_history(50)
    if rows:
        for rid, topic, created in rows:
            with st.expander(f"{rid} • {topic} • {created}"):
                if st.button(f"Load {rid}", key=f"load_{rid}"):
                    data = load_timeline_from_db(rid)
                    if data:
                        st.write(data.get("timeline", [])[:8])
    else:
        st.info("No saved timelines yet.")

# --- About
else:
    st.header("About")
    st.markdown("""
This app is a competition-ready News Orchestrator that reconstructs stories across many sources.

Features:
- Aggregation (NewsAPI + Google RSS)
- Event extraction, deduplication, and chronological timeline
- Advanced Analysis: Verified Facts, Conflicts, Source Credibility, Bias scoring
- Multi-language (Tamil included), OpenAI-assisted refinement (optional)

Notes:
- Enable NEWS_API_KEY and OPENAI_API_KEY in env for best results (use .env).
- Install spaCy `en_core_web_sm` for better NER.
- For timeline PNG export install `kaleido` (`pip install kaleido`).
""")
# End of file
