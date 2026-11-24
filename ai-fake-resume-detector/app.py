"""AI Fake Resume Detector - Flask web app (uses the v2 models saved from the notebook)."""
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
gb_model = joblib.load("models/authenticity_model_v2.pkl")
lr_bal = joblib.load("models/tfidf_model_v2.pkl")
tfidf = joblib.load("models/tfidf_vectorizer_v2.pkl")
FEATS = joblib.load("models/features_v2.pkl")

# ---------- Feature engineering (same as the notebook) ----------
STOPWORDS = set("a an the and or but if is are was were to of in on at for with as by from up down".split())
SKILLS = ["Python", "TensorFlow", "PyTorch", "Kubernetes", "Docker", "AWS", "Spark", "Tableau", "React",
          "Node.js", "SQL", "Java", "C++", "NLP", "Computer Vision", "Hadoop", "Kafka", "Azure", "Scala",
          "MongoDB", "Power BI", "Machine Learning", "Deep Learning", "Flask", "Django"]
BUZZ = {"passionate", "dynamic", "synergy", "results", "driven", "proven", "cutting", "edge", "leveraging",
        "innovative", "motivated", "dedicated", "strategic", "seamless", "robust", "scalable", "excellence"}


def tokenize(t):
    words = re.findall(r"[a-z]+", t.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


SKILL_TOKENS = set(tokenize(" ".join(SKILLS)))


def extract_features(text):
    tokens = tokenize(text)
    unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0.0
    stuffing = Counter(tokens).most_common(1)[0][1] / len(tokens) if tokens else 0.0
    grams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)] if len(tokens) > 3 else []
    counts = Counter(grams)
    repeated = sum(c for c in counts.values() if c > 1) / len(grams) if grams else 0.0
    sents = [s for s in re.split(r"[.!?\n]+", text) if s.strip()]
    lens = [len(tokenize(s)) for s in sents if len(tokenize(s)) > 0]
    n = max(len(tokens), 1)
    words = text.split()
    return {
        "unique_word_ratio": unique_ratio,
        "keyword_stuffing_score": stuffing,
        "repeated_ngram_ratio": repeated,
        "avg_sentence_length": float(np.mean(lens)) if lens else 0.0,
        "n_words": len(words),
        "buzz_ratio": sum(w in BUZZ for w in tokens) / n,
        "skill_density": sum(w in SKILL_TOKENS for w in tokens) / n,
        "digits_per_100w": len(re.findall(r"\d+", text)) / max(len(words), 1) * 100,
        "avg_word_len": float(np.mean([len(w) for w in tokens])) if tokens else 0.0,
    }


def similarity(a, b):
    if not a.strip() or not b.strip():
        return 0.0
    vec = TfidfVectorizer(stop_words="english")
    m = vec.fit_transform([a, b])
    return float(cosine_similarity(m[0], m[1])[0][0])


def read_pdf(file_storage):
    text = ""
    with pdfplumber.open(file_storage.stream) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
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
</body>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        text = read_pdf(request.files["resume"])
        jd = request.form.get("jd", "")
        jd_given = bool(jd.strip())

        feats = extract_features(text)
        x = pd.DataFrame([feats])[FEATS]
        authenticity = (1 - gb_model.predict_proba(x)[0][1]) * 100
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
            feedback.append("Authenticity signals are weak; check for repetitive, padded or keyword-heavy content.")
        if feats["n_words"] > 300:
            feedback.append("Resume is much longer than typical; long documents can be over-flagged by this model.")
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
                      missing=", ".join(missing_list) or "None", feedback=feedback)
    return render_template_string(PAGE, r=result)


if __name__ == "__main__":
    app.run(port=5000)
