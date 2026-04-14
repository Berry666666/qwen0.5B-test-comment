import argparse

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class PredictRequest(BaseModel):
    text: str
    max_length: int = 256


def create_app(model_dir: str) -> FastAPI:
    app = FastAPI(title="Weibo Sentiment API", version="1.0.0")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, trust_remote_code=True)
    model.config.pad_token_id = tokenizer.pad_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    @app.get("/health")
    def health():
        return {"status": "ok", "device": device}

    @app.post("/predict")
    def predict(req: PredictRequest):
        with torch.inference_mode():
            encoded = tokenizer(
                req.text,
                return_tensors="pt",
                truncation=True,
                max_length=req.max_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()
            pred = int(torch.argmax(logits, dim=-1).item())
            label = "positive" if pred == 1 else "negative"
            return {
                "label": label,
                "negative": round(probs[0], 6),
                "positive": round(probs[1], 6),
            }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(args.model_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
