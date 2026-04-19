import re
import unicodedata

import numpy as np
from datasets import Audio, DatasetDict

MIN_DURATION = 0.5
MAX_DURATION = 30.0
MAX_TOKEN = 448
MAX_CHARS = 2


def normalize_uzbek_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s'ʻʼ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_batch(batch: dict) -> dict:
    batch["sentence"] = [normalize_uzbek_text(s) for s in batch["sentence"]]
    return batch


def _is_valid(sample: dict) -> bool:
    arr = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]
    duration = len(arr) / sr
    text = sample["sentence"].strip()
    if not (MIN_DURATION <= duration <= MAX_DURATION):
        return False
    if len(text) < MIN_CHARS:
        return False

    return True


def clean_datasets(raw: DatasetDict) -> DatasetDict:
    raw = raw.map(_normalize_batch, batched=True, num_proc=4, desc="Normalizing text")
    before = {k: len(v) for k, v in raw.items()}
    for k in raw:
        print(f"{k}: {before[k]} → {len(raw[k])} (removed {before[k] - len(raw[k])})")
    return raw
