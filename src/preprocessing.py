import re
import unicodedata

import numpy as np
from datasets import Audio, DatasetDict
from transformers import WhisperProcessor

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


def get_processor(cfg) -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(
        cfg.model.name,
        language=cfg.model.language,
        task=cfg.model.task,
    )


def make_prepare_fn(processor: WhisperProcessor):
    def prepare_dataset(batch: dict) -> dict:
        audios = [s["array"] for s in batch["audio"]]
        inputs = processor.feature_extractor(
            audios,
            sampling_rate=16_000,
            return_tensors="np",
        )
        labels = processor.tokenizer(
            batch["sentence"],
            max_length=MAX_TOKENS,
            truncation=True,
        )

        batch["input_length"] = [len(a) / 16_000 for a in audios]
        return batch

    return prepare_dataset


def extract_features(
    cleaned: DatasetDict, processor: WhisperProcessor, cfg
) -> DatasetDict:

    cache = cfg.data.processed_cache
    from pathlib import Path

    if Path(cache).exists():
        from datasets import load_from_disk

        print(f"Loading feature cache from {cache}")
        return load_from_disk(cache)

    cleaned = cleaned.cast_column("audio", Audio(sampling_rate=16_000))

    prepare_fn = make_prepare_fn(processor)
    processed = cleaned.map(
        prepare_fn,
        batched=True,
        batch_size=32,
        num_proc=4,
        remove_columns=cleaned["train"].column_names,
        desc="Extracting features",
    )

    processed = processed.filter(
        lambda x: len(x["labels"]) < MAX_TOKENS,
        desc="Filtering overlength labels",
    )

    processed.save_to_disk(cache)
    print(f"Feature cache saved to {cache}")
    return processed
