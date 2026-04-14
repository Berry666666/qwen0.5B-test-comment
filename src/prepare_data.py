import argparse
import json
from pathlib import Path

from datasets import DatasetDict, load_dataset
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_repo", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = load_dataset(args.dataset_repo)["train"]

    labels = ds["label"]
    idx = list(range(len(ds)))

    train_idx, temp_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=args.seed,
        stratify=temp_labels,
    )

    split_ds = DatasetDict(
        {
            "train": ds.select(train_idx),
            "validation": ds.select(val_idx),
            "test": ds.select(test_idx),
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_ds.save_to_disk(str(output_dir))

    stats = {
        "train": len(split_ds["train"]),
        "validation": len(split_ds["validation"]),
        "test": len(split_ds["test"]),
        "label_distribution": {
            split: split_ds[split].to_pandas()["label"].value_counts().to_dict()
            for split in ["train", "validation", "test"]
        },
    }
    with (output_dir / "split_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
