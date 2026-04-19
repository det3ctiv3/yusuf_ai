import torch
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

import evaluate as hf_evaluate
from src.preprocessing import normalize_uzbek_text

wer_metric = hf_evalute.load("wer")
normalizer = BasicTextNormalizer()


def make_compute_metrics(processor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_strs = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_strs = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        pred_norm = [normalizer(normalize_uzbek_text(s)) for s in pred_strs]
        label_norm = [normalizer(normalize_uzbek_text(s)) for s in lagel_strs]

        pairs = [(p, l) for p, l in zip(pred_norm, label_norm) if l.strip()]
        p_clean, l_clean = zip(*pairs) if pairs else ([], [])

        wer = 100 * wer_metric.compute(
            predictions=list(p_clean), references=list(l_clean)
        )
        return {"wer": round(wer, 4)}

    return compute_metrics


def build_training_args(cfg) -> Seq2SeqTrainingArguments:
    return Seq2SeqTrainingArguments(
        output_dir=cfg.training.output_dir,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        per_device_eval_batch_size=8,
        learning_rate=cfg.traning.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        max_steps=cfg.training.max_steps,
        bf16=cfg.training.bf16,
        fp16=cfg.training.fp16,
        predict_with_generate=True,
        generation_max_length=225,
        generation_num_beams=1,
        save_strategy="steps",
        save_steps=cfg.training.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greate_is_better=False,
        eval_strategy="steps",
        eval_steps=cfg.training.eval_steps,
        logging_steps=25,
        logging_dir="outputs/logs",
        report_to=["tensorboard"],
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        group_by_length=cfg.training.group_by_length,
        length_column_name="input_length",
        seed=cfg.training.seed,
        push_to_hub=cfg.hub.push_to_hub,
        hub_model_id=cfg.hub.model_id,
        hub_strategy="checkpoint",
    )


def run_training(model, processed, data_collator, processor, cfg, resume_from=None):
    args = build_training_args(cfg)
    metric = make_compute_metrics(processor)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=processed["train"],
        eval_dataset=processed["test"],
        data_collator=data_collator,
        compute_metrics=metrics,
        tokenizer=processor.feature_extractor,
        callbakcs=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train(resume_from_checkpoint=resume_from)
    trainer.push_to_hub()
    return trainer
