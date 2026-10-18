import random
import os
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer, set_seed


MODEL_NAME = "facebook/wav2vec2-base"

import zipfile

def unzip_archive(zip_path, output_folder):
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_folder)

unzip_archive("train.zip", "data/")
unzip_archive("private_test.zip", "data/")

SEED = random.randint(1, 10000)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)


def get_SEED():
    return SEED


bona_fide = 0
spoof = 1
label_to_id = {"bonafide": bona_fide, "spoof": spoof}
rows = []


with open("data/train/targets.txt", "r", encoding="utf-8") as file:
    for line in file:
        filename, other = line.split(' ', maxsplit=1)[1].split(" -", maxsplit=1)
        A, label = other.split()
        rows.append({"filename": filename, "label": label})

data = pd.DataFrame(rows)


def load_audio(path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1) # моно

    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

    return audio


feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        processed = feature_extractor(load_audio("data/train/flac/" + row["filename"] + '.flac'), sampling_rate=16000, return_tensors="pt",)
        return {"input_values": processed["input_values"][0], "labels": torch.tensor(label_to_id[row["label"]], dtype=torch.long)}


def make_batch(samples):
    processed = feature_extractor.pad({"input_values": [sample["input_values"] for sample in samples]}, padding=True, return_tensors="pt")
    return {"input_values": processed["input_values"], "labels": torch.stack([sample["labels"] for sample in samples])}

train_data, valid_data = train_test_split(data, test_size=0.1, random_state=SEED, stratify=data["label"])

train_data = train_data.sample(n=200, random_state=SEED).reset_index(drop=True)
valid_data = valid_data.sample(n=50, random_state=SEED).reset_index(drop=True)
train_dataset = AudioDataset(train_data)
valid_dataset = AudioDataset(valid_data)

model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME, num_labels=2, id2label={bona_fide: "bona_fide", spoof: "spoof"}, label2id={"bona_fide": bona_fide, "spoof": spoof}, ignore_mismatched_sizes=True)
model.to("cuda" if torch.cuda.is_available() else "cpu")

def calculate_metrics(prediction):
    logits = prediction.predictions
    true_labels = prediction.label_ids

    # Вероятность класса 1 = вероятность синтетики
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    spoof_scores = probabilities[:, spoof]
    auc = roc_auc_score(true_labels, spoof_scores)

    return {"roc_auc": auc}

output_dir = os.path.join("output", str(SEED))
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(output_dir=output_dir, seed=SEED, per_device_train_batch_size=2, per_device_eval_batch_size=2,
                                  gradient_accumulation_steps=1, num_train_epochs=1, max_steps=20, weight_decay=0.01, fp16=False,
                                  gradient_checkpointing=False, eval_strategy="steps", eval_steps=10, save_strategy="steps",
                                  save_steps=10, save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="roc_auc",
                                  greater_is_better=True, logging_steps=1, report_to="none")
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=valid_dataset, data_collator=make_batch, compute_metrics=calculate_metrics)


trainer.train()

trainer.save_model(output_dir)
feature_extractor.save_pretrained(output_dir)
