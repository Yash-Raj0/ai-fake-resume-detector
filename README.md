# AI Fake Resume Detector

A machine-learning tool that checks how trustworthy a resume looks. Upload a resume PDF and get a quality report with an authenticity score, job-description relevance, completeness, duplicate-content check and actionable feedback.

## What it does

| Score | How it is computed |
|---|---|
| **Authenticity** | Gradient Boosting model on text-pattern features (unique-word ratio, keyword stuffing, repeated phrases, buzzword ratio, skill density, etc.). A TF-IDF + Logistic Regression model gives a second opinion. |
| **Relevance** | TF-IDF cosine similarity between the resume and a job description |
| **Completeness** | Checks for experience/projects, education, skills, email and phone |
| **Duplicate similarity** | TF-IDF cosine similarity between two resumes (flags copy-paste above 75%) |
| **Overall quality** | Weighted blend of the scores above |

## How the model was trained

1. Started with a public resume dataset (~9.5k rows); built ~2,500 clean resumes as the "genuine" class.
2. Created "fake" resumes by injecting three kinds of problems into genuine ones: **keyword stuffing**, **unsupported skills** and **generic buzzword padding**.
3. Split train/test by source resume (`GroupShuffleSplit`) so a resume and its fake copy never land on both sides.
4. Compared TF-IDF + Logistic Regression, class-balanced Logistic Regression, and a feature-based Gradient Boosting model using accuracy, precision, recall, F1 and AUC.
5. Ran a feature ablation (drop one feature at a time) to see which signals matter.

## Results

Evaluated on a held-out test set of 1,000 resumes, split by source resume so a resume and its fake copy never appear on both sides.

| model                      |   acc |   prec |   recall |    f1 |   auc |
|:---------------------------|------:|-------:|---------:|------:|------:|
| TF-IDF + LogReg            | 0.888 |  1     |    0.776 | 0.874 | 0.998 |
| Features + GradBoost       | 0.919 |  0.969 |    0.866 | 0.914 | 0.985 |
| TF-IDF + LogReg (balanced) | 0.934 |  1     |    0.868 | 0.929 | 0.998 |

**Share of fakes caught, by type**

| fake type          | TF-IDF | TF-IDF (balanced) | Features |
|:-------------------|-------:|------------------:|---------:|
| generic padding    |  1.000 |             1.000 |    1.000 |
| keyword stuffing   |  1.000 |             1.000 |    1.000 |
| unsupported skills |  0.329 |             0.605 |    0.599 |

- Genuine resumes wrongly flagged: 0% (TF-IDF models), 2.8% (features model).
- Most important feature: `skill_density`. Removing it drops F1 from 0.914 to 0.799.
- Weakest spot: fakes with unsupported skills, since only about 60% are caught. This is the first thing to improve.

## Limitations (please read)

- The "fake" labels are **synthetic**. The model learns to spot these specific manipulations, not every kind of fake resume, so high test scores do not guarantee real-world accuracy.
- Very long resumes can be over-flagged.
- This is a screening aid, not proof of fraud. Never use it as the only basis for rejecting someone.

## Run it locally

```bash
git clone https://github.com/Yash-Raj0/ai-fake-resume-detector.git
cd ai-fake-resume-detector
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:5000 in your browser.

## Project structure

```
├── app.py                      # Flask web app
├── models/                     # trained models saved from the notebook
├── notebooks/                  # Colab notebook (data prep, training, evaluation)
├── requirements.txt
└── README.md
```

## Tech stack

Python, scikit-learn, pandas, NumPy, pdfplumber, Flask

## Author

Yash Raj, B.Tech CSE, KIIT University
