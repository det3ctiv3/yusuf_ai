import os
import random

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.data_pipeline import build_train_test_split, load_raw_datasets
from src.model import DataCollatorSpeechSeq2Seq, load_model
from src.preprocessing import clean_datasets, extract_features, get_processor
from src.trainer import run_training


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


def main():
    load_dotenv()

    cfg = OmegaConf.load("configs/train_config.yaml")
    set_seed(cfg.training.seed)

    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        tags=list(cfg.wandb.tags),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    print("Loading raw dataset (1/5)")
    raw = load_raw_datasets(cfg)
    raw = build_train_test_split(raw, cfg)

    print("Cleaning and normalization (2/5")
    cleaned = clean_datasets(raw)

    print("Loading processor and extracting features (3/5)")
    processor = get_processor(cfg)
    processed = extract_features(cleaned, processor, cfg)

    print("Loading model and LoRA (4/5")
    model = load_model(cfg)
    collator = DataCollatorSpeechSeq2Seq(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    print("Train (5/5")
    run_training(model, processed, collator, processor, cfg, resume_from=True)
