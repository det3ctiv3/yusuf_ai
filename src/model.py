from dataclasses import dataclass
from typing import Any
import torch
from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list) -> dict:
        input_features = [{"input_features": ["input_features"]} for f in features]
        batch = self.processor.feature_extractor(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensor="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels.batch["attention_mask"].ne(1), -100
        )
        if labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def load_model(cfg):
    model = WhisperForConditionalGeneration.from_pretrained(
        cfg.model.name,
        torch_dtype = torch.bfloat16 if cfg.training.bf16 else torch.float32,
    )
    model.generation_config.language = cfg.model.language
    model.generation_config.task = cfg.model.task
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    if cfg.lora.use_lora:
        model = _apply_lora(model, cfg)

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    trainable, total = _count_params(model)
    print(f"Trainable params: {trainable:,} / {total:,} "
        f"({100*trainable/total}%)")

    return model

def _apply_lora(model, cfg):
    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        target_modules = list(cfg.lora.target_modules),
        lora_dropout = cfg.lora.dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_cfg)
    return model

def _count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def merge_and_save_lora(checkpoint_path: str, output_path: str, cfg):
    from peft import PefrModel
    base = WhisperForConditionalGeneration.from_pretrained(
        cfg.model.name, torch_dtype=torch.float32
    )

    peft_model = PeftModel.from_pretrained(base, checkpoint_path)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(output_path, safe_serialization=True)
    print(f"Merged model saveed to {output_path}")
