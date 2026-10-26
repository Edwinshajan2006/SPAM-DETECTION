from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

DATA_PATH = Path("data/SMSSpamCollection")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. Download the SMS Spam Collection "
        "dataset and place the SMSSpamCollection file inside the data/ folder."
    )

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8",
)

df = df.dropna(subset=["label", "message"])
df["label"] = df["label"].str.strip().str.lower()
df["message"] = df["message"].astype(str).str.strip()
df = df[df["label"].isin(["ham", "spam"])]
df = df.drop_duplicates(subset=["label", "message"])

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
        ),
    ),
    ("classifier", LinearSVC(class_weight="balanced")),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

metrics = {
    "Accuracy": accuracy_score(y_test, predictions),
    "Spam Precision": precision_score(
        y_test, predictions, pos_label="spam", zero_division=0
    ),
    "Spam Recall": recall_score(
        y_test, predictions, pos_label="spam", zero_division=0
    ),
    "Spam F1-score": f1_score(
        y_test, predictions, pos_label="spam", zero_division=0
    ),
}

print("\nMODEL EVALUATION")
for name, value in metrics.items():
    print(f"{name}: {value:.4f} ({value * 100:.2f}%)")

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, predictions, zero_division=0))

print("\nCONFUSION MATRIX (labels: ham, spam)")
print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

model_path = Path("spam_model.joblib")
joblib.dump(model, model_path)

print(f"\nModel saved as {model_path}")
print(f"Messages used: {len(df)}")
print(f"Training messages: {len(X_train)}")
print(f"Testing messages: {len(X_test)}")
