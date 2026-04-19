# src/data_pipeline.py
import os
from pathlib import Path

from datasets import (
    Audio,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from dotenv import load_dotenv

load_dotenv()

AUDIO_FEATURE = Audio(sampling_rate=16_000)


def standardize(ds):
    if "transcription" in ds.column_names and "sentence" not in ds.column_names:
        ds = ds.rename_column("transcription", "sentence")
    ds = ds.select_columns(["audio", "sentence"])
    ds = ds.cast_column("audio", AUDIO_FEATURE)
    return ds


def load_raw_datasets(cfg) -> DatasetDict:
    cache_path = Path("data/processed/raw_combined")

    if cache_path.exists():
        print(f"Loading cached dataset from {cache_path}")
        return load_from_disk(str(cache_path))

    # --- murodbek/uzbek-speech-corpus ---
    print("Downloading murodbek/uzbek-speech-corpus ...")
    raw_datasets = DatasetDict(
        {
            "train": standardize(
                load_dataset(
                    cfg.data.dataset_name,
                    split="train+validation",
                    trust_remote_code=True,
                    token=os.environ["HF_TOKEN"],
                )
            ),
            "test": standardize(
                load_dataset(
                    cfg.data.dataset_name,
                    split="test",
                    trust_remote_code=True,
                    token=os.environ["HF_TOKEN"],
                )
            ),
        }
    )
    print(raw_datasets)

    # --- google/fleurs uz_uz ---
    print("Downloading google/fleurs uz_uz ...")
    fleurs_uz = load_dataset("google/fleurs", "uz_uz", trust_remote_code=True)
    for split in fleurs_uz.keys():
        fleurs_uz[split] = standardize(fleurs_uz[split])

    raw_datasets["train"] = concatenate_datasets(
        [
            raw_datasets["train"],
            fleurs_uz["train"],
        ]
    )
    print(f"Combined train size: {len(raw_datasets['train'])}")

    raw_datasets.save_to_disk(str(cache_path))
    print(f"Saved to {cache_path}")
    return raw_datasets


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/train_config.yaml")
    ds = load_raw_datasets(cfg)
