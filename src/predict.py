import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, trust_remote_code=True)
    model.config.pad_token_id = tokenizer.pad_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    encoded = tokenizer(
        args.text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    logits = model(**encoded).logits
    probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()
    pred = int(torch.argmax(logits, dim=-1).item())

    label = "positive" if pred == 1 else "negative"
    print({"label": label, "negative": round(probs[0], 6), "positive": round(probs[1], 6)})


if __name__ == "__main__":
    main()
