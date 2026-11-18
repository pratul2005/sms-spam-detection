"""Predict whether a supplied SMS message is ham or spam."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one SMS message as ham or spam.")
    parser.add_argument("message", help="SMS message text to classify")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help=f"Path to a trained model (default: {MODEL_PATH})",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(
            f"Model not found at {args.model}. Run 'python -m sms_spam.train' first."
        )

    model = joblib.load(args.model)
    prediction = model.predict([args.message])[0]
    probabilities = model.predict_proba([args.message])[0]
    class_probabilities = dict(zip(model.classes_, probabilities))

    print(f"Prediction: {prediction.upper()}")
    print("Estimated class probabilities:")
    for label, probability in class_probabilities.items():
        print(f"  {label}: {probability:.4f}")
    print("Note: predictions are estimates, not a guarantee that a message is safe.")


if __name__ == "__main__":
    main()
