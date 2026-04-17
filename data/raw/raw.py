from datasets import load_dataset, DatasetDict

raw_datasets = DatasetDict()
# %%% DOWNLOADING UZBEK SPEECH CORPUS DATASET ....
raw_datasets['train'] = load_dataset(
    "murodbek/uzbek-speech-corpus",
    split = "train+validation",
    trust_remote_code = True,
    token = True,
)

raw_datasets['test'] = load_dataset(
    "murodbek/uzbek-speech-corpus",
    split = "test",
    trust_remote_code = True,
    token = True,
)
raw_train = raw_datasets['train'].select_columns(["audio", "sentence"])
raw_test = raw_datasets['test'].select_columns(["audio", "sentence"])

raw_datasets = DatasetDict({
    "train": raw_train,
    "test": raw_test,
})
print(raw_datasets)

# %%% DOWNLOADING FLEURS DATASET ....
fleurs_uz = load_dataset(
    "google/fleurs",
    "uz_uz",
    trust_remote_code = True,
)

for split in fleurs_uz.keys():

    ds = fleurs_uz[split]
    if "transcription" in ds.column_names:
        ds = ds.rename_column("transcription", "sentence")
    ds = ds.select_columns(["audio", "sentence"])
    fleurs_uz[split] = ds

from datasets import concatenate_datasets

raw_datasets['train'] = concatenate_datasets([
    raw_datasets['train'],
    fleurs_uz['train'],
])


print(f"Combined train size: {len(raw_datasets['train'])}")
