"""Unit tests for SMS dataset validation."""

import pandas as pd
import pytest

from sms_spam.data import load_dataset


def test_load_dataset_rejects_unknown_labels(tmp_path, monkeypatch):
    data_file = tmp_path / "bad.tsv"
    data_file.write_text("ham\tHello\nunknown\tBad label\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected labels"):
        load_dataset(data_file)


def test_load_dataset_keeps_duplicate_rows(tmp_path):
    data_file = tmp_path / "messages.tsv"
    data_file.write_text(
        "ham\tHello there\nham\tHello there\nspam\tClaim a prize\n",
        encoding="utf-8",
    )
    frame = load_dataset(data_file)
    assert list(frame.columns) == ["label", "text"]
    assert len(frame) == 3
    assert set(frame["label"]) == {"ham", "spam"}
