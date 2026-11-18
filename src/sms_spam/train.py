"""Train and evaluate the TF-IDF + Multinomial Naive Bayes pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sms_spam.data import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def build_pipeline() -> Pipeline:
    """Create the complete text vectorisation and classification pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", MultinomialNB(alpha=1.0)),
        ]
    )


def main() -> None:
    data = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    labels = ["ham", "spam"]
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    report = classification_report(
        y_test, predictions, labels=labels, target_names=labels, zero_division=0
    )
    metrics = {
        "dataset": "UCI SMS Spam Collection",
        "dataset_rows_used": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_fraction": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "vectorizer": "TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)",
        "classifier": "MultinomialNB(alpha=1.0)",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "spam_precision": float(precision_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "spam_recall": float(recall_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "spam_f1": float(f1_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "confusion_matrix_labels": labels,
        "confusion_matrix": matrix.tolist(),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")
    (ARTIFACTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (ARTIFACTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    output = pd.DataFrame(
        {"text": X_test.to_numpy(), "actual": y_test.to_numpy(), "predicted": predictions}
    )
    output.to_csv(ARTIFACTS_DIR / "test_predictions.csv", index=False)

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(cmap="Blues", values_format="d")
    plt.title("SMS Spam Detection — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    print("Training and evaluation complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts saved to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
