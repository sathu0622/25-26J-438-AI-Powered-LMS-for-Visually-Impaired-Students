import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


def load_grade_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    return df.fillna("")


def build_examples(df: pd.DataFrame) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []

    for _, row in df.iterrows():
        chapter = str(row.get("chapter", "")).strip()
        topic = str(row.get("Grade/Topic", "")).strip()
        original_text = str(row.get("original_text", "")).strip()
        simplified_text = str(row.get("simplified_text", "")).strip()
        emotion = str(row.get("emotion", "")).strip()
        sound_effects = str(row.get("sound_effects", "")).strip()

        source = (
            f"chapter: {chapter}\n"
            f"topic: {topic}\n"
            f"source: {original_text[:1200]}"
        )

        if simplified_text:
            examples.append({
                "input_text": f"task: narrate\n{source}",
                "target_text": simplified_text,
            })

        if emotion:
            examples.append({
                "input_text": f"task: emotion\n{source}",
                "target_text": emotion,
            })

        if sound_effects:
            examples.append({
                "input_text": f"task: sound_effects\n{source}",
                "target_text": sound_effects,
            })

    return examples


def split_examples(examples: List[Dict[str, str]], val_ratio: float, seed: int) -> (pd.DataFrame, pd.DataFrame):
    df = pd.DataFrame(examples)
    if df.empty:
        raise ValueError("No training examples created. Check dataset columns and content.")

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_size = max(1, int(len(df) * val_ratio))
    val_df = df.iloc[:val_size].reset_index(drop=True)
    train_df = df.iloc[val_size:].reset_index(drop=True)

    if train_df.empty:
        train_df = df.iloc[:-1].reset_index(drop=True)
        val_df = df.iloc[-1:].reset_index(drop=True)

    return train_df, val_df


@dataclass
class Seq2SeqRow:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]


class SimpleSeq2SeqDataset:
    def __init__(self, rows: List[Seq2SeqRow]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        return {
            "input_ids": row.input_ids,
            "attention_mask": row.attention_mask,
            "labels": row.labels,
        }


def encode_dataframe(df: pd.DataFrame, tokenizer, max_source_len: int, max_target_len: int) -> SimpleSeq2SeqDataset:
    rows: List[Seq2SeqRow] = []
    for _, r in df.iterrows():
        tokenized_source = tokenizer(
            text=str(r["input_text"]),
            truncation=True,
            max_length=max_source_len,
        )
        tokenized_target = tokenizer(
            text=str(r["target_text"]),
            truncation=True,
            max_length=max_target_len,
        )

        rows.append(
            Seq2SeqRow(
                input_ids=tokenized_source["input_ids"],
                attention_mask=tokenized_source["attention_mask"],
                labels=tokenized_target["input_ids"],
            )
        )

    return SimpleSeq2SeqDataset(rows)


def train(args):
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
    from peft import LoraConfig, TaskType, get_peft_model

    grade10 = load_grade_csv(args.grade10_csv)
    grade11 = load_grade_csv(args.grade11_csv)

    all_examples = build_examples(grade10) + build_examples(grade11)
    train_df, val_df = split_examples(all_examples, val_ratio=args.val_ratio, seed=args.seed)

    if args.max_train_samples > 0:
        train_df = train_df.head(args.max_train_samples).reset_index(drop=True)
        max_val = max(1, int(args.max_train_samples * args.val_ratio))
        val_df = val_df.head(max_val).reset_index(drop=True)

    os.makedirs(args.examples_dir, exist_ok=True)
    train_csv = os.path.join(args.examples_dir, "train_examples.csv")
    val_csv = os.path.join(args.examples_dir, "val_examples.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q", "v"],
    )
    model = get_peft_model(base_model, lora_config)

    train_ds = encode_dataframe(train_df, tokenizer, args.max_source_len, args.max_target_len)
    val_ds = encode_dataframe(val_df, tokenizer, args.max_source_len, args.max_target_len)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    fp16 = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=20,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        predict_with_generate=False,
        report_to="none",
        fp16=fp16,
        dataloader_num_workers=0,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    trainer.train()

    os.makedirs(args.adapter_dir, exist_ok=True)
    model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)

    print("Training completed.")
    print(f"Adapter saved to: {args.adapter_dir}")
    print(f"Train examples: {train_csv}")
    print(f"Val examples: {val_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train multitask LoRA model for lesson narration/emotion/sound effects")
    parser.add_argument("--base-model", default="google/flan-t5-small")
    parser.add_argument("--grade10-csv", default=os.path.join("data", "grade10_dataset.csv"))
    parser.add_argument("--grade11-csv", default=os.path.join("data", "grade11_dataset.csv"))
    parser.add_argument("--output-dir", default=os.path.join("models", "lesson_multitask_lora", "runs"))
    parser.add_argument("--adapter-dir", default=os.path.join("models", "lesson_multitask_lora"))
    parser.add_argument("--examples-dir", default=os.path.join("models", "lesson_multitask_lora", "examples"))

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-source-len", type=int, default=384)
    parser.add_argument("--max-target-len", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
