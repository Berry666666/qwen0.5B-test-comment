import argparse
import json

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    ds = load_from_disk(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    def preprocess(batch):
        return tokenizer(batch["review"], truncation=True, max_length=args.max_length)

    tokenized = ds.map(preprocess, batched=True, remove_columns=["review"])

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, trust_remote_code=True)
    model.config.pad_token_id = tokenizer.pad_token_id

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="binary", zero_division=0),
            "precision": precision_score(labels, preds, average="binary", zero_division=0),
            "recall": recall_score(labels, preds, average="binary", zero_division=0),
        }

    eval_args = TrainingArguments(
        output_dir="/tmp/weibo_eval",
        per_device_eval_batch_size=64,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
