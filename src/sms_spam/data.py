"""Download and load the UCI SMS Spam Collection dataset."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "SMSSpamCollection"


def download_dataset(path: Path = DEFAULT_DATA_PATH) -> Path:
    """Download and extract the official UCI file if it is not already present."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        DATA_URL, headers={"User-Agent": "sms-spam-detection-student-project/0.1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        member = next(
            (name for name in archive.namelist() if name.endswith("SMSSpamCollection")),
            None,
        )
        if member is None:
            raise FileNotFoundError("The downloaded UCI archive has no SMSSpamCollection file.")
        payload = archive.read(member)

    path.write_bytes(payload)
    return path


def load_dataset(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Return a frame with 'label' and 'text' columns; retain duplicate messages."""
    dataset_path = download_dataset(path)
    frame = pd.read_csv(
        dataset_path,
        sep="\t",
        header=None,
        names=["label", "text"],
        quoting=3,
        encoding="utf-8",
        on_bad_lines="error",
    )
    frame = frame.dropna(subset=["label", "text"]).copy()
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["text"].ne("")]
    unexpected = sorted(set(frame["label"]) - {"ham", "spam"})
    if unexpected:
        raise ValueError(f"Unexpected labels in dataset: {unexpected}")
    if frame.empty:
        raise ValueError("The dataset contains no usable messages.")
    return frame.reset_index(drop=True)
