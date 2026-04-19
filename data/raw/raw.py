from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset

AUDIO_FEATURE = Audio(sampling_rate=16000)


def standardize(ds):  # Standardize sample rate for both datasets
    if "transcription" in ds.column_names and "sentence" not in ds.column_names:
        ds = ds.rename_column("transcription", "sentence")
    ds = ds.select_columns(["audio", "sentence"])
    ds = ds.cast_column("audio", AUDIO_FEATURE)
    return ds


raw_datasets = DatasetDict()
# %%% DOWNLOADING UZBEK SPEECH CORPUS DATASET ....
raw_datasets["train"] = load_dataset(
    "murodbek/uzbek-speech-corpus",
    split="train+validation",
    trust_remote_code=True,
    token=True,
)

raw_datasets["test"] = load_dataset(
    "murodbek/uzbek-speech-corpus",
    split="test",
    trust_remote_code=True,
    token=True,
)
raw_train = raw_datasets["train"].select_columns(["audio", "sentence"])
raw_test = raw_datasets["test"].select_columns(["audio", "sentence"])

raw_datasets = DatasetDict(
    {
        "train": standardize(raw_train),
        "test": standardize(raw_test),
    }
)
print(raw_datasets)

# %%% DOWNLOADING FLEURS DATASET ....
fleurs_uz = load_dataset(
    "google/fleurs",
    "uz_uz",
    trust_remote_code=True,
)

for split in fleurs_uz.keys():
    ds = standardize(fleurs_uz[split])
    if "transcription" in ds.column_names:
        ds = ds.rename_column("transcription", "sentence")
    ds = ds.select_columns(["audio", "sentence"])
    fleurs_uz[split] = ds


raw_datasets["train"] = concatenate_datasets(
    [
        raw_datasets["train"],
        fleurs_uz["train"],
    ]
)


print(f"Combined train size: {len(raw_datasets['train'])}")


# Save to disk (Arrow format — extremely fast I/O)
raw_datasets.save_to_disk("data/processed/raw_combined")

# Load on subsequent runs (instant)
from datasets import load_from_disk

raw_datasets = load_from_disk("data/processed/raw_combined")

# Verify checksums to ensure data integrity
print(raw_datasets.info())
