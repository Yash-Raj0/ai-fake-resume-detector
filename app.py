"""AI Fake Resume Detector - Flask web app (uses model v4 saved from the notebook)."""
import re
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import pdfplumber
from flask import Flask, render_template_string, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Load models saved by the notebook ----------
gb_model = joblib.load("models/authenticity_model_v4.pkl")   # features model (14 features)
FEATS = joblib.load("models/features_v4.pkl")                # exact feature names and order
lr_bal = joblib.load("models/tfidf_model_v2.pkl")            # TF-IDF second opinion
tfidf = joblib.load("models/tfidf_vectorizer_v2.pkl")

# ---------- Feature engineering (identical to the notebook) ----------
STOPWORDS = set("a an the and or but if is are was were to of in on at for with as by from up down".split())
SKILLS = ["Python", "TensorFlow", "PyTorch", "Kubernetes", "Docker", "AWS", "Spark", "Tableau", "React",
          "Node.js", "SQL", "Java", "C++", "NLP", "Computer Vision", "Hadoop", "Kafka", "Azure", "Scala",
          "MongoDB", "Power BI", "Machine Learning", "Deep Learning", "Flask", "Django"]
BUZZ = {"passionate", "dynamic", "synergy", "results", "driven", "proven", "cutting", "edge", "leveraging",
        "innovative", "motivated", "dedicated", "strategic", "seamless", "robust", "scalable", "excellence"}
SENIOR = {"chief", "director", "vp", "vice", "president", "head", "principal", "officer", "cto", "ceo"}


def tokenize(t):
    words = re.findall(r"[a-z]+", t.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


SKILL_TOKENS = set(tokenize(" ".join(SKILLS)))


def extract_features(t):
    tokens = tokenize(t)
    uniq = len(set(tokens)) / len(tokens) if tokens else 0.0
    stuff = Counter(tokens).most_common(1)[0][1] / len(tokens) if tokens else 0.0
    if len(tokens) < 4:
        rep = 0.0
    else:
        grams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
        c = Counter(grams)
        rep = sum(v for v in c.values() if v > 1) / len(grams)
    sents = [s for s in re.split(r"[.!?\n]+", t) if s.strip()]
    lens = [len(tokenize(s)) for s in sents]
    lens = [l for l in lens if l > 0]
    return {"unique_word_ratio": uniq, "keyword_stuffing_score": stuff,
            "repeated_ngram_ratio": rep, "avg_sentence_length": float(np.mean(lens)) if lens else 0.0}


def more_features(t):
    toks = tokenize(t)
    n = max(len(toks), 1)
    return {"n_words": len(t.split()),
            "buzz_ratio": sum(w in BUZZ for w in toks) / n,
            "skill_density": sum(w in SKILL_TOKENS for w in toks) / n,
            "digits_per_100w": len(re.findall(r"\d+", t)) / max(len(t.split()), 1) * 100,
            "avg_word_len": float(np.mean([len(w) for w in toks])) if toks else 0.0}


def skill_features(t):
    """Skills listed in the SKILLS section vs skills mentioned anywhere else."""
    inline = re.findall(r"[Ss]kills:\s*([^.\n]*)", t)
    rest = re.sub(r"[Ss]kills:\s*[^.\n]*", " ", t)
    skills_lines, other_lines, in_skills = [], [], False
    for line in rest.split("\n"):
        s = line.strip()
        if s.isupper() and 0 < len(s.split()) <= 4:          # a heading such as SKILLS or EXPERIENCE
            in_skills = "SKILL" in s
            continue
        (skills_lines if in_skills else other_lines).append(line)
    skills_text = " ".join(skills_lines + inline)
    other_low = " ".join(other_lines).lower()
    items = [x.strip().lower() for x in re.split(r"[,|;\n:•]", skills_text)]
    items = [x for x in items if 1 < len(x) <= 30]
    unsupported = [x for x in items if x not in other_low]
    tech_unsupported = [s for s in SKILLS if s.lower() in skills_text.lower() and s.lower() not in other_low]
    return {"skills_listed": len(items),
            "unsupported_skill_ratio": len(unsupported) / len(items) if items else 0.0,
            "unsupported_tech_skills": len(tech_unsupported)}


def robust_extra(t):
    lines = [l.strip().lower() for l in t.split("\n") if l.strip()]
    seen, dup = set(), 0
    for l in lines:
        if l in seen:
            dup += 1
        seen.add(l)
    toks = re.findall(r"[a-z]+", t.lower())
    n = max(len(toks), 1)
    return {"dup_line_ratio": dup / max(len(lines), 1),
            "senior_terms_per_100w": 100 * sum(w in SENIOR for w in toks) / n}


def all_features(text):
    feats = {**extract_features(text), **more_features(text), **skill_features(text), **robust_extra(text)}
    return pd.DataFrame([feats])[FEATS]


def similarity(a, b):
    if not a.strip() or not b.strip():
        return 0.0
    vec = TfidfVectorizer(stop_words="english")
    m = vec.fit_transform([a, b])
    return float(cosine_similarity(m[0], m[1])[0][0])


def read_pdf(file_storage):
    """Returns the text of a PDF, or an empty string if the file is not a readable PDF."""
    text = ""
    try:
        with pdfplumber.open(file_storage.stream) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception:
        return ""
    return text


# ---------- Web app ----------
app = Flask(__name__)

PAGE = """
<!doctype html>
<title>AI Fake Resume Detector</title>
<body style="font-family:Arial;max-width:720px;margin:30px auto;padding:0 12px">
<h1>AI Fake Resume Detector</h1>
<form method=post enctype=multipart/form-data>
  <p>Resume (PDF): <input type=file name=resume accept=".pdf" required></p>
  <p>Second resume for duplicate check (optional): <input type=file name=resume2 accept=".pdf"></p>
  <p>Job description (optional):<br><textarea name=jd rows=6 cols=70></textarea></p>
  <input type=submit value=Analyze>
</form>
{% if r %}
<hr>
<h2>Report</h2>
{% if r.error %}<p><b>{{ r.error }}</b></p>{% else %}
<p><b>Authenticity:</b> {{ "%.1f"|format(r.authenticity) }} / 100
   <small>(second opinion, TF-IDF model: {{ "%.1f"|format(r.second) }})</small></p>
{% if r.jd_given %}<p><b>Relevance:</b> {{ "%.1f"|format(r.relevance) }} / 100</p>{% endif %}
<p><b>Completeness:</b> {{ "%.1f"|format(r.completeness) }} / 100</p>
{% if r.dup is not none %}<p><b>Duplicate similarity:</b> {{ "%.1f"|format(r.dup) }}%</p>{% endif %}
<h3>Overall Quality Score: {{ "%.1f"|format(r.quality) }} / 100</h3>
<p><b>Missing:</b> {{ r.missing }}</p>
<h3>Feedback</h3>
<ul>{% for f in r.feedback %}<li>{{ f }}</li>{% endfor %}</ul>
{% endif %}
<p><small>Screening aid only. The model was trained on synthetic fake resumes and can miss
manipulation types it has not seen. Never use it as the only basis for a decision.</small></p>
{% endif %}
</body>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        text = read_pdf(request.files["resume"])
        if len(text.split()) < 30:
            return render_template_string(PAGE, r={"error": "Not enough text found. Upload a text-based PDF, not a scan."})
        jd = request.form.get("jd", "")
        jd_given = bool(jd.strip())

        authenticity = (1 - gb_model.predict_proba(all_features(text))[0][1]) * 100
        second = (1 - lr_bal.predict_proba(tfidf.transform([text]))[0][1]) * 100
        relevance = similarity(text, jd) * 100 if jd_given else 0.0

        checks = {
            "experience/projects": bool(re.search(r"experience|projects", text, re.I)),
            "education": bool(re.search(r"education", text, re.I)),
            "skills": bool(re.search(r"skills", text, re.I)),
            "email": bool(re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text)),
            "phone": bool(re.search(r"\d{10}", re.sub(r"[\s-]", "", text))),
        }
        completeness = sum(checks.values()) / len(checks) * 100
        missing_list = [k for k, v in checks.items() if not v]

        dup = None
        f2 = request.files.get("resume2")
        if f2 and f2.filename:
            dup = similarity(text, read_pdf(f2)) * 100

        if jd_given:
            quality = 0.4 * authenticity + 0.35 * relevance + 0.25 * completeness
        else:
            quality = 0.6 * authenticity + 0.4 * completeness

        feedback = []
        if authenticity < 50:
            feedback.append("Authenticity signals are weak; check for repetitive, padded, keyword-heavy or copy-pasted content.")
        if jd_given and relevance < 40:
            feedback.append("Low match with the job description; tailor language and skills to the role.")
        if dup is not None and dup > 75:
            feedback.append("High similarity to the other resume; personalize wording to avoid template-reuse flags.")
        if missing_list:
            feedback.append("Add missing sections/info: " + ", ".join(missing_list) + ".")
        if not feedback:
            feedback.append("No major issues detected. Resume looks reasonably clean and complete.")

        result = dict(authenticity=authenticity, second=second, relevance=relevance, jd_given=jd_given,
                      completeness=completeness, dup=dup, quality=quality,
                      missing=", ".join(missing_list) or "None", feedback=feedback, error=None)
    return render_template_string(PAGE, r=result)


if __name__ == "__main__":
    app.run(port=5000)
