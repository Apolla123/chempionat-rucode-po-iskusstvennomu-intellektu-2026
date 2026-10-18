import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from train import get_SEED, load_audio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


SEED = get_SEED()  # Используй тот же SEED, что и при обучении

SUBMISSION_FILE = os.path.join(f"output/{SEED}",f"submission_seed_{SEED}.csv")

model = AutoModelForAudioClassification.from_pretrained(f"output/{SEED}")
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

feature_extractor = AutoFeatureExtractor.from_pretrained(f"output/{SEED}")


def predict_file(path):
    processed = feature_extractor(load_audio(path), sampling_rate=16000, return_tensors="pt")
    processed = {key: value.to("cuda" if torch.cuda.is_available() else "cpu") for key, value in processed.items()}

    with torch.no_grad():
        logits = model(**processed).logits

    logits = logits.cpu().numpy()[0]
    logits = logits - logits.max()
    exp_logits = np.exp(logits)
    probabilities = exp_logits / exp_logits.sum()

    return float(probabilities[1])


results = []

for i, filename in enumerate(pd.read_csv("private_sample_submission.csv")["idx"]):
    score = predict_file(os.path.join("data/private_test", str(filename)))
    results.append({"idx": filename, "score": score})

submission = pd.DataFrame(results, columns=["idx", "score"])
submission.to_csv(SUBMISSION_FILE, index=False)