"""Compare the existing SMS Shield model with the uploaded notebook's model.

Run from the repository root:
    python compare_models.py

Both models use the same cleaned dataset and the same stratified train/test split.
This script does not overwrite spam_model.joblib.
"""
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATA_PATH = Path("data/SMSSpamCollection")
RANDOM_STATE = 42

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. Put the UCI SMSSpamCollection "
        "file inside the data/ folder."
    )

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8",
)
df = df.dropna(subset=["label", "message"])
df["label"] = df["label"].astype(str).str.strip().str.lower()
df["message"] = df["message"].astype(str).str.strip()
df = df[df["label"].isin(["ham", "spam"])]
df = df.drop_duplicates(subset=["label", "message"]).reset_index(drop=True)

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

models = {
    # Current repository model: bigrams, sublinear term frequency, balanced classes.
    "Current SMS Shield model": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True,
            ),
        ),
        ("classifier", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)),
    ]),
    # Model from the uploaded notebook: English stop-word removal and unigrams.
    "Uploaded notebook model": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(lowercase=True, stop_words="english"),
        ),
        ("classifier", LinearSVC(random_state=RANDOM_STATE)),
    ]),
}

scoring = {
    "accuracy": "accuracy",
    "precision_spam": make_scorer(precision_score, pos_label="spam", zero_division=0),
    "recall_spam": make_scorer(recall_score, pos_label="spam", zero_division=0),
    "f1_spam": make_scorer(f1_score, pos_label="spam", zero_division=0),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

summary = []
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    test_metrics = {
        "Model": name,
        "Test accuracy": accuracy_score(y_test, predictions),
        "Spam precision": precision_score(
            y_test, predictions, pos_label="spam", zero_division=0
        ),
        "Spam recall": recall_score(
            y_test, predictions, pos_label="spam", zero_division=0
        ),
        "Spam F1": f1_score(y_test, predictions, pos_label="spam", zero_division=0),
    }
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=None)
    for metric in scoring:
        test_metrics[f"CV {metric} mean"] = cv_results[f"test_{metric}"].mean()
        test_metrics[f"CV {metric} std"] = cv_results[f"test_{metric}"].std()
    summary.append(test_metrics)

    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    print(classification_report(y_test, predictions, labels=["ham", "spam"], zero_division=0))
    print("Confusion matrix (rows=true [ham, spam], columns=predicted [ham, spam]):")
    print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

results = pd.DataFrame(summary).set_index("Model")
print("\nCOMPARISON (all metrics are proportions from 0 to 1)")
print(results.round(4).to_string())
print(
    "\nChoose based on spam F1/recall and false positives, not accuracy alone. "
    "This comparison does not change spam_model.joblib."
)
