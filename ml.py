import html
import io
import json
import os
import random
import re
import time

import fitz  # PyMuPDF: page count + summary PDF
import pandas as pd
import streamlit as st
from google import genai

BAD = ("image", "tts", "live", "audio", "native", "robotics", "computer", "embedding", "omni")
LEVEL = {"high": ("#ef4444", "🔥"), "medium": ("#f59e0b", "⭐"), "low": ("#10b981", "📌")}
PAL = ["#ef4444", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"]
GRAD = ["#6366f1,#8b5cf6", "#ec4899,#f43f5e", "#f59e0b,#f97316", "#10b981,#14b8a6", "#3b82f6,#06b6d4"]


# ---------------- Gemini helpers ----------------
@st.cache_data(show_spinner=False)
def list_models(key):
    try:
        c = genai.Client(api_key=key)
        names = [m.name.replace("models/", "") for m in c.models.list()
                 if "generateContent" in (m.supported_actions or [])]
        names = [n for n in names if "flash" in n and not any(b in n for b in BAD)]
        return sorted(names, key=lambda n: ("lite" not in n, n)) or ["gemini-2.5-flash"]
    except Exception:
        return ["gemini-2.5-flash"]


def generate(client, models, contents, config):
    """Try models in order; retry on busy (503); skip to next model on quota (429) / not found."""
    last = None
    for m in models[:4]:
        for attempt in range(3):
            try:
                return client.models.generate_content(model=m, contents=contents, config=config), m
            except Exception as ex:
                last, s = ex, str(ex)
                if any(x in s for x in ("API key", "PERMISSION_DENIED", "API_KEY_INVALID")):
                    raise
                if any(x in s for x in ("503", "UNAVAILABLE", "500")) and attempt < 2:
                    time.sleep(4 * (attempt + 1))
                    continue
                break
    raise last


PROMPT = """You are an expert study assistant. Read the ENTIRE attached PDF including all images, diagrams, charts,
tables and any scanned/handwritten pages (use vision). The PDF has about {pages} pages.
Write ALL text values in {lang}. Make a chapter-wise summary of about {words} words (~{pct}% of the original).
Return ONLY valid JSON (no markdown fences) with exactly this shape:
{{"title":"","overview":"3-4 sentences",
"chapters":[{{"title":"","points":[{{"text":"","key":true}}]}}],
"topics":[{{"name":"","level":"High|Medium|Low","why":"","pages":""}}],
"images":[{{"page":1,"description":"what the figure shows and why it matters"}}],
"definitions":[{{"term":"","meaning":""}}],"keywords":[""],
"questions":["likely exam question"],
"cheatsheet":["formula / rule / fact to remember"],
"flashcards":[{{"q":"","a":""}}],
"quiz":[{{"q":"","options":["A","B","C","D"],"answer":0,"why":"short explanation"}}],
"mindmap":{{"root":"","branches":[{{"name":"","children":[""]}}]}}}}
Rules: "key":true for definitions, formulas, rules, likely exam points. 6-12 topics ranked by importance.
Describe every important image/diagram. 8-15 exam questions, 12-20 flashcards, 8-12 quiz MCQs
("answer" is the 0-based index of the correct option), 10-20 cheatsheet items, mind map with 5-9 branches."""


def analyse(key, models, data, pct, lang):
    client = genai.Client(api_key=key)
    pages = len(fitz.open(stream=data, filetype="pdf"))
    words = int(min(12000, max(300, pages * 350 * pct / 100)))
    f = client.files.upload(file=io.BytesIO(data), config={"mime_type": "application/pdf"})
    try:
        while f.state and f.state.name == "PROCESSING":
            time.sleep(2)
            f = client.files.get(name=f.name)
        resp, used = generate(client, models, [f, PROMPT.format(pages=pages, words=words, pct=pct, lang=lang)],
                              {"response_mime_type": "application/json", "temperature": 0.3})
    finally:
        try:
            client.files.delete(name=f.name)
        except Exception:
            pass
    raw = re.sub(r"^```(?:json)?|```$", "", (resp.text or "").strip(), flags=re.M).strip()
    out = json.loads(raw)
    out["_pages"], out["_model"] = pages, used
    return out


def ask(key, models, d, history, q):
    ctx = json.dumps({k: d.get(k) for k in ("title", "overview", "chapters", "definitions", "cheatsheet")})[:60000]
    chat = "\n".join(f"{r}: {t}" for r, t in history[-6:])
    p = (f"You are a friendly tutor. Answer using ONLY the study material below; if it is not covered, say so briefly. "
         f"Keep answers clear and short.\n\nMATERIAL:\n{ctx}\n\nCHAT SO FAR:\n{chat}\n\nSTUDENT: {q}")
    r, _ = generate(genai.Client(api_key=key), models, [p], {"temperature": 0.4})
    return r.text


# ---------------- Exports ----------------
def e(x):
    return html.escape(str(x))


def report_html(d):
    h = [f"<h1>{e(d.get('title', 'Summary'))}</h1><p>{e(d.get('overview', ''))}</p><h2>Important topics</h2><ul>"]
    h += [f"<li><b>[{e(t.get('level'))}] {e(t.get('name'))}</b> (pages {e(t.get('pages', ''))}) - {e(t.get('why'))}</li>" for t in d.get("topics", [])]
    m = d.get("mindmap", {})
    h.append(f"</ul><h2>Mind map: {e(m.get('root', ''))}</h2><ul>")
    h += [f"<li><b>{e(b.get('name'))}</b><ul>" + "".join(f"<li>{e(c)}</li>" for c in b.get("children", [])) + "</ul></li>" for b in m.get("branches", [])]
    h.append("</ul>")
    for ch in d.get("chapters", []):
        h.append(f"<h2>{e(ch.get('title'))}</h2><ul>")
        h += [f"<li><b>* {e(p.get('text'))}</b></li>" if p.get("key") else f"<li>{e(p.get('text'))}</li>" for p in ch.get("points", [])]
        h.append("</ul>")
    h.append("<h2>Cheat sheet</h2><ul>" + "".join(f"<li>{e(x)}</li>" for x in d.get("cheatsheet", [])) + "</ul>")
    h.append("<h2>Definitions</h2><ul>" + "".join(f"<li><b>{e(x.get('term'))}</b>: {e(x.get('meaning'))}</li>" for x in d.get("definitions", [])) + "</ul>")
    h.append("<h2>Exam questions</h2><ol>" + "".join(f"<li>{e(q)}</li>" for q in d.get("questions", [])) + "</ol>")
    return "".join(h)


def to_pdf(body):
    try:
        buf = io.BytesIO()
        w = fitz.DocumentWriter(buf)
        st_ = fitz.Story(html="<style>h1{color:#4f46e5}h2{color:#7c3aed}body{font-size:10pt}</style>" + body)
        box = fitz.paper_rect("a4")
        more = 1
        while more:
            dev = w.begin_page(box)
            more, _ = st_.place(box + (40, 40, -40, -40))
            st_.draw(dev)
            w.end_page()
        w.close()
        return buf.getvalue()
    except Exception:
        return None


def mindmap_dot(m):
    q = lambda s: str(s).replace('"', "'")[:70]
    d = ['digraph G{rankdir=LR;bgcolor="transparent";node[shape=box,style="rounded,filled",fontname="Helvetica",'
         'fontsize=12,color="#cbd5e1",fillcolor="#f8fafc",fontcolor="#0f172a"];edge[color="#94a3b8"];',
         f'r[label="{q(m.get("root", "Topic"))}",fillcolor="#4f46e5",fontcolor="white",fontsize=16];']
    for i, b in enumerate(m.get("branches", [])):
        c = PAL[i % len(PAL)]
        d.append(f'b{i}[label="{q(b.get("name"))}",fillcolor="{c}",fontcolor="white"];r->b{i}[color="{c}"];')
        for j, ch in enumerate(b.get("children", [])):
            d.append(f'c{i}_{j}[label="{q(ch)}",color="{c}"];b{i}->c{i}_{j}[color="{c}"];')
    return "".join(d) + "}"


# ---------------- UI ----------------
st.set_page_config(page_title="StudyBuddy AI", page_icon="🎓", layout="wide")
st.markdown("""
<style>
@keyframes flow{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
.hero{padding:2.2rem;border-radius:22px;color:#fff;margin-bottom:1.2rem;background:linear-gradient(120deg,#4f46e5,#9333ea,#ec4899,#f59e0b);
background-size:300% 300%;animation:flow 12s ease infinite;box-shadow:0 12px 32px rgba(147,51,234,.4)}
.hero h1{margin:0;font-size:2.6rem}.hero p{margin:.5rem 0 0;font-size:1.1rem;opacity:.96}
.stat{padding:1rem;border-radius:16px;color:#fff;text-align:center;box-shadow:0 6px 16px rgba(0,0,0,.18)}
.stat b{font-size:2rem;display:block}
.ch{padding:.7rem 1.2rem;margin:.7rem 0;border-radius:14px;background:rgba(127,127,127,.08);border-left:7px solid}
.ch h4{margin:.2rem 0 .4rem}.ch li{margin:.25rem 0}
.topic{padding:.8rem 1rem;margin:.5rem 0;border-radius:14px;background:rgba(127,127,127,.08);border-left:7px solid}
.fc{padding:3rem 2rem;border-radius:22px;color:#fff;text-align:center;font-size:1.5rem;font-weight:600;min-height:200px;
display:flex;align-items:center;justify-content:center;box-shadow:0 10px 26px rgba(0,0,0,.25)}
.chip{display:inline-block;padding:.3rem .9rem;margin:.2rem;border-radius:999px;font-size:.95rem;color:#fff}
.stTabs [data-baseweb="tab"]{font-size:1.05rem;font-weight:600}
</style>
<div class="hero"><h1>🎓 StudyBuddy AI</h1>
<p>Upload a big PDF (text, images or scanned) → short chapter-wise summary • important topics • mind map • flashcards • quiz • ask-the-PDF chat</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    try:
        dk = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        dk = ""
    key = st.text_input("Gemini API key", value=dk, type="password", help="Free: aistudio.google.com/apikey")
    all_models = list_models(key) if key else []
    model = st.selectbox("Model", all_models) if key else None
    pct = st.slider("Summary size (% of original)", 3, 30, 10, help="10% → 500 pages become ~50 pages")
    lang = st.selectbox("Summary language", ["English", "Hindi (हिन्दी)", "Hinglish"])
    st.caption("Busy or quota error? The app retries and automatically tries other models.")

if not key:
    st.info("👈 Paste your free Gemini API key in the sidebar (get it at aistudio.google.com/apikey).")
    st.stop()

up = st.file_uploader("📂 Drop your PDF here", type="pdf")
if up is None:
    st.stop()

if st.button("🚀 Summarize my PDF", type="primary", use_container_width=True):
    order = [model] + [m for m in all_models if m != model]
    try:
        with st.status("🤖 AI is reading your PDF (text + images)...", expanded=True) as status:
            st.write("📤 Uploading and reading all pages...")
            data = analyse(key, order, up.getvalue(), pct, lang)
            status.update(label=f"✅ Done using {data['_model']}", state="complete", expanded=False)
        st.session_state.update(d={"file": up.name, "data": data}, rid=random.randint(1, 10**6),
                                chat=[], fc_i=0, fc_flip=False, models=order)
    except json.JSONDecodeError:
        st.error("The AI reply was cut off. Lower the summary size a little and try again.")
    except Exception as ex:
        st.error(f"Could not finish: {ex}\n\nTip: wait a minute, or pick another model in the sidebar.")

s = st.session_state.get("d")
if not s or s["file"] != up.name:
    st.stop()
d, rid = s["data"], st.session_state["rid"]
fcs, quiz = d.get("flashcards", []), d.get("quiz", [])

stats = [(d["_pages"], "Original pages"), (len(d.get("chapters", [])), "Chapters"), (len(d.get("topics", [])), "Key topics"),
         (len(fcs), "Flashcards"), (len(quiz), "Quiz questions")]
for c, g, (v, l) in zip(st.columns(5), GRAD, stats):
    c.markdown(f"<div class='stat' style='background:linear-gradient(135deg,{g})'><b>{v}</b>{l}</div>", unsafe_allow_html=True)
st.write("")

body = report_html(d)
pdf = to_pdf(body)
c1, c2, c3 = st.columns(3)
if pdf:
    c1.download_button("⬇️ Summary PDF", pdf, file_name="summary.pdf", use_container_width=True, type="primary")
c2.download_button("⬇️ Summary HTML", f"<html><meta charset='utf-8'><body style='font-family:Arial;max-width:800px;margin:auto'>{body}</body></html>",
                   file_name="summary.html", use_container_width=True)
c3.download_button("⬇️ Flashcards CSV (Anki)", pd.DataFrame(fcs).to_csv(index=False), file_name="flashcards.csv", use_container_width=True)
if "Hindi" in lang:
    st.caption("ℹ️ For Hindi, use the HTML download (the PDF export may not show Hindi letters).")

T = st.tabs(["📝 Summary", "🔥 Topics", "🗺️ Mind Map", "🃏 Flashcards", "🧪 Quiz", "💬 Ask PDF", "📋 Cheat Sheet", "🖼️ Images", "📖 Terms", "❓ Exam Qs"])

with T[0]:
    st.info(d.get("overview", ""))
    st.caption("⭐ = key / exam-important point")
    for i, ch in enumerate(d.get("chapters", [])):
        lis = "".join(f"<li>{'⭐ <b>' + e(p.get('text')) + '</b>' if p.get('key') else e(p.get('text'))}</li>" for p in ch.get("points", []))
        st.markdown(f"<div class='ch' style='border-color:{PAL[i % 8]}'><h4>📘 {e(ch.get('title'))}</h4><ul>{lis}</ul></div>", unsafe_allow_html=True)

with T[1]:
    for t in d.get("topics", []):
        col, ic = LEVEL.get(str(t.get("level", "")).lower(), LEVEL["medium"])
        st.markdown(f"<div class='topic' style='border-color:{col}'><b>{ic} {e(t.get('name'))}</b> "
                    f"<small>({e(t.get('level'))} • pages {e(t.get('pages', '-'))})</small><br>{e(t.get('why'))}</div>", unsafe_allow_html=True)

with T[2]:
    if d.get("mindmap"):
        st.graphviz_chart(mindmap_dot(d["mindmap"]), use_container_width=True)

with T[3]:
    if not fcs:
        st.info("No flashcards returned.")
    else:
        box = st.container()
        a, b, c, x = st.columns(4)
        if a.button("⬅️ Previous", use_container_width=True):
            st.session_state.update(fc_i=st.session_state["fc_i"] - 1, fc_flip=False)
        if b.button("🔄 Flip card", use_container_width=True):
            st.session_state["fc_flip"] = not st.session_state["fc_flip"]
        if c.button("Next ➡️", use_container_width=True):
            st.session_state.update(fc_i=st.session_state["fc_i"] + 1, fc_flip=False)
        if x.button("🔀 Shuffle", use_container_width=True):
            random.shuffle(fcs)
            st.session_state.update(fc_i=0, fc_flip=False)
        i, flip = st.session_state["fc_i"] % len(fcs), st.session_state["fc_flip"]
        g = "#10b981,#14b8a6" if flip else "#6366f1,#ec4899"
        box.markdown(f"<div class='fc' style='background:linear-gradient(135deg,{g})'>{'💡 ' + e(fcs[i].get('a')) if flip else '❓ ' + e(fcs[i].get('q'))}</div>", unsafe_allow_html=True)
        box.caption(f"Card {i + 1} of {len(fcs)} • press Flip to see the answer")

with T[4]:
    for i, q in enumerate(quiz):
        st.markdown(f"**Q{i + 1}. {q.get('q')}**")
        st.radio("Choose", q.get("options", []), index=None, key=f"qz{rid}_{i}", label_visibility="collapsed")
    if quiz and st.button("✅ Check my score", type="primary"):
        score = 0
        for i, q in enumerate(quiz):
            opts, ans = q.get("options", []), q.get("answer", 0)
            right = opts[ans] if isinstance(ans, int) and 0 <= ans < len(opts) else None
            ok = st.session_state.get(f"qz{rid}_{i}") == right
            score += ok
            (st.success if ok else st.error)(f"Q{i + 1}: {'Correct ✅' if ok else 'Wrong ❌'} — Answer: {right}. {q.get('why', '')}")
        st.balloons() if score >= len(quiz) * 0.7 else None
        st.subheader(f"🏆 Score: {score} / {len(quiz)}")

with T[5]:
    st.caption("Ask anything about your PDF. Answers come from the generated study material.")
    for role, txt in st.session_state["chat"]:
        st.chat_message(role).write(txt)
    if q := st.chat_input("e.g. Explain the first chapter in simple words"):
        st.chat_message("user").write(q)
        with st.spinner("Thinking..."):
            try:
                ans = ask(key, st.session_state["models"], d, st.session_state["chat"], q)
            except Exception as ex:
                ans = f"Sorry, could not answer right now ({ex}). Try again in a minute."
        st.chat_message("assistant").write(ans)
        st.session_state["chat"] += [("user", q), ("assistant", ans)]

with T[6]:
    for x in d.get("cheatsheet", []):
        st.markdown(f"- 📌 {x}")
    st.markdown("".join(f"<span class='chip' style='background:{PAL[i % 8]}'>{e(k)}</span>" for i, k in enumerate(d.get("keywords", []))), unsafe_allow_html=True)

with T[7]:
    for im in d.get("images", []) or [None]:
        st.info("No important images found.") if im is None else st.markdown(f"**🖼️ Page {im.get('page')}** — {im.get('description')}")

with T[8]:
    for x in d.get("definitions", []):
        st.markdown(f"- **{x.get('term')}**: {x.get('meaning')}")

with T[9]:
    for i, q in enumerate(d.get("questions", []), 1):
        st.markdown(f"**Q{i}.** {q}")