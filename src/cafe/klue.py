import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from torch import nn
import os  # 추가

# 1. 데이터 로드
df = pd.read_csv('aspect_cleaned.csv', encoding='utf-8-sig')

# 2. 라벨 인코딩
# 측면(aspect) 라벨 인코딩
aspect_le = LabelEncoder()
df['aspect_label'] = aspect_le.fit_transform(df['aspect'])
num_aspects = len(aspect_le.classes_)

# 감정(sentiment) 라벨 인코딩
sentiment_le = LabelEncoder()
df['sentiment_label'] = sentiment_le.fit_transform(df['sentiment'])
num_sentiments = len(sentiment_le.classes_)

# 3. 데이터셋 분리
train_texts, val_texts, train_aspect_labels, val_aspect_labels, train_sentiment_labels, val_sentiment_labels = train_test_split(
    df['text'].tolist(),
    df['aspect_label'].tolist(),
    df['sentiment_label'].tolist(),
    test_size=0.2,
    random_state=42
)

# 4. 토크나이저 로드
model_name = "klue/bert-base"
tokenizer = BertTokenizer.from_pretrained(model_name)


# 5. 인코딩 함수
def tokenize_function(texts):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    return encodings


train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)


# 6. 멀티태스크 데이터셋 클래스
class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, aspect_labels, sentiment_labels):
        self.encodings = encodings
        self.aspect_labels = aspect_labels
        self.sentiment_labels = sentiment_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['aspect_labels'] = torch.tensor(self.aspect_labels[idx])
        item['sentiment_labels'] = torch.tensor(self.sentiment_labels[idx])
        return item

    def __len__(self):
        return len(self.aspect_labels)


train_dataset = MultiTaskDataset(train_encodings, train_aspect_labels, train_sentiment_labels)
val_dataset = MultiTaskDataset(val_encodings, val_aspect_labels, val_sentiment_labels)


# 7. 멀티태스크 모델 정의
class MultiTaskBertModel(nn.Module):
    def __init__(self, model_name, num_aspects, num_sentiments):
        super(MultiTaskBertModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_aspects).bert

        # 측면 분류 헤드
        self.aspect_classifier = nn.Linear(self.bert.config.hidden_size, num_aspects)

        # 감정 분류 헤드
        self.sentiment_classifier = nn.Linear(self.bert.config.hidden_size, num_sentiments)

        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, aspect_labels=None, sentiment_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # 두 개의 출력 헤드
        aspect_logits = self.aspect_classifier(pooled_output)
        sentiment_logits = self.sentiment_classifier(pooled_output)

        loss = None
        if aspect_labels is not None and sentiment_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            aspect_loss = loss_fct(aspect_logits, aspect_labels)
            sentiment_loss = loss_fct(sentiment_logits, sentiment_labels)
            loss = aspect_loss + sentiment_loss

        return {
            'loss': loss,
            'aspect_logits': aspect_logits,
            'sentiment_logits': sentiment_logits
        }


model = MultiTaskBertModel(model_name, num_aspects, num_sentiments)


# 8. 커스텀 트레이너 정의
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        aspect_labels = inputs.pop("aspect_labels")
        sentiment_labels = inputs.pop("sentiment_labels")

        outputs = model(**inputs, aspect_labels=aspect_labels, sentiment_labels=sentiment_labels)
        loss = outputs['loss']

        return (loss, outputs) if return_outputs else loss


# 9. 학습 파라미터 설정
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

# 10. 트레이너 실행
trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# 11. 모델 저장 (디렉토리 생성 추가)
save_dir = "./cafe_multitask_model"
os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성

torch.save(model.state_dict(), f"{save_dir}/model.pt")
tokenizer.save_pretrained(save_dir)

# 라벨 인코더도 함께 저장
import pickle

with open(f'{save_dir}/aspect_le.pkl', 'wb') as f:
    pickle.dump(aspect_le, f)
with open(f'{save_dir}/sentiment_le.pkl', 'wb') as f:
    pickle.dump(sentiment_le, f)

print(f"학습 완료! 측면 {num_aspects}개, 감정 {num_sentiments}개 분류 모델이 저장되었습니다.")
print(f"측면 클래스: {list(aspect_le.classes_)}")
print(f"감정 클래스: {list(sentiment_le.classes_)}")