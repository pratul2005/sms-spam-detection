# SMS Spam Detection Using Machine Learning and NLP

A reproducible academic project that classifies SMS messages as **ham** (legitimate) or **spam** using TF-IDF text features and a Multinomial Naive Bayes classifier.

> **Project status:** Initial implementation committed. Run the experiment locally before reporting any performance figures. No model results are claimed in this README.

## 1. Dataset

This project uses the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), a public collection of labelled SMS messages. The UCI page reports 5,574 instances and a CC BY 4.0 licence. The dataset is downloaded automatically the first time the training command is run; the raw data is not committed to this repository.

Dataset citation: Almeida, T. and Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84

## 2. Method

1. Download and parse the tab-separated UCI data.
2. Validate labels and remove empty message rows.
3. Split the data into training and held-out test sets using stratification and a fixed random seed.
4. Convert message text into TF-IDF features (word unigrams and bigrams).
5. Train a Multinomial Naive Bayes classifier.
6. Evaluate on the held-out test set using accuracy, precision, recall, F1-score and a confusion matrix.

The test set is kept separate from model fitting. The random seed and test fraction are recorded in the training script for reproducibility.

## 3. Requirements

- Python 3.10 or newer
- Internet access on the first run to download the dataset

## 4. Installation

```bash
git clone https://github.com/pratul2005/sms-spam-detection.git
cd sms-spam-detection
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

## 5. Train and evaluate

```bash
python -m sms_spam.train
```

The script downloads the dataset to `data/SMSSpamCollection` if it is not already present, trains the pipeline and writes reproducible outputs to `artifacts/`:

- `metrics.json` — evaluation metrics and split metadata
- `classification_report.txt` — per-class precision, recall and F1
- `confusion_matrix.png` — held-out test confusion matrix
- `test_predictions.csv` — test messages, true labels and predictions
- `model.joblib` — fitted TF-IDF + Naive Bayes pipeline

These generated files are excluded from Git by default. Record the actual values from `metrics.json` in the project report only after running the experiment.

## 6. Classify a message

After training:

```bash
python -m sms_spam.predict "Congratulations! You have won a free prize"
```

The command prints the predicted class and the model's estimated class probabilities. These are model estimates, not guarantees that a message is safe.

## 7. Project structure

```text
sms-spam-detection/
├── data/                  # downloaded dataset (not committed)
├── artifacts/             # model and evaluation outputs (not committed)
├── docs/
│   └── report_results_template.md
├── src/
│   └── sms_spam/
│       ├── __init__.py
│       ├── data.py
│       ├── train.py
│       └── predict.py
├── tests/
│   └── test_data.py
├── .gitignore
├── pyproject.toml
└── README.md
```

## 8. Limitations

- The corpus consists of English SMS messages and may not represent current spam patterns, other languages, messaging apps or regional usage.
- Dataset class balance and historical collection sources can affect measured performance.
- A false negative can allow spam through; a false positive can incorrectly flag a legitimate message. Both should be considered when interpreting metrics.
- A prediction should not be treated as a security guarantee.

## 9. Academic integrity and reproducibility

This repository contains the implementation and instructions. Run it, inspect the generated outputs, and understand the preprocessing, split and metrics before submitting or presenting the work. Do not invent results or describe unperformed experiments as completed. A report template is provided in `docs/report_results_template.md`.
