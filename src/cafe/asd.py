import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# =========================
# 1. 데이터 로드
# =========================
df = pd.read_csv('aspect_cleaned.csv', encoding='utf-8-sig')

# =========================
# 2. 라벨 인코딩 (aspect)
# =========================
le = LabelEncoder()
df['label'] = le.fit_transform(df['aspect'])
num_labels = len(le.classes_)

print("Aspect mapping:")
for i, label in enumerate(le.classes_):
    print(f"{i} -> {label}")

# =========================
# 3. Train / Validation split
# =========================
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['sentence'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# =========================
# 4. Tokenizer
# =========================
model_name = "klue/bert-base"
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128
    )

train_encodings = tokenize(train_texts)
val_encodings = tokenize(val_texts)

# =========================
# 5. Dataset
# =========================
class CafeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CafeDataset(train_encodings, train_labels)
val_dataset = CafeDataset(val_encodings, val_labels)

# =========================
# 6. Model
# =========================
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

# =========================
# 7. Metrics
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

# =========================
# 8. TrainingArguments
# =========================
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    report_to="none",
    fp16=torch.cuda.is_available(),   # ⭐ GPU 있으면 자동
)

# =========================
# 9. Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# =========================
# 10. 모델 저장
# =========================
model.save_pretrained("./cafe_aspect_model")
tokenizer.save_pretrained("./cafe_aspect_model")

# label encoder 저장
pd.Series(le.classes_).to_csv(
    "./cafe_aspect_model/label_map.csv",
    index=False,
    header=False
)

print(f"✅ 학습 완료! 총 {num_labels}개 aspect 분류 모델 저장됨")
