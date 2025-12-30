import torch
import pickle
from transformers import BertTokenizer

# 1. 모델과 토크나이저 로드
model_path = "./cafe_multitask_model"
tokenizer = BertTokenizer.from_pretrained(model_path)

# 라벨 인코더 로드
with open(f'{model_path}/aspect_le.pkl', 'rb') as f:
    aspect_le = pickle.load(f)
with open(f'{model_path}/sentiment_le.pkl', 'rb') as f:
    sentiment_le = pickle.load(f)

# 모델 구조 재정의 (학습 때와 동일하게)
from torch import nn
from transformers import BertForSequenceClassification


class MultiTaskBertModel(nn.Module):
    def __init__(self, model_name, num_aspects, num_sentiments):
        super(MultiTaskBertModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_aspects).bert
        self.aspect_classifier = nn.Linear(self.bert.config.hidden_size, num_aspects)
        self.sentiment_classifier = nn.Linear(self.bert.config.hidden_size, num_sentiments)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        aspect_logits = self.aspect_classifier(pooled_output)
        sentiment_logits = self.sentiment_classifier(pooled_output)

        return aspect_logits, sentiment_logits


# 모델 로드
num_aspects = len(aspect_le.classes_)
num_sentiments = len(sentiment_le.classes_)
model = MultiTaskBertModel("klue/bert-base", num_aspects, num_sentiments)
model.load_state_dict(torch.load(f"{model_path}/model.pt"))
model.eval()


# 2. 예측 함수
def predict(text):
    # 토크나이징
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # 예측
    with torch.no_grad():
        aspect_logits, sentiment_logits = model(inputs['input_ids'], inputs['attention_mask'])

    # 측면 예측 (가장 높은 확률)
    aspect_pred = torch.argmax(aspect_logits, dim=1).item()
    aspect_name = aspect_le.inverse_transform([aspect_pred])[0]

    # 감정 예측 (가장 높은 확률)
    sentiment_pred = torch.argmax(sentiment_logits, dim=1).item()
    sentiment_name = sentiment_le.inverse_transform([sentiment_pred])[0]

    # 확률 계산 (softmax)
    aspect_probs = torch.softmax(aspect_logits, dim=1)[0]
    sentiment_probs = torch.softmax(sentiment_logits, dim=1)[0]

    return {
        'text': text,
        'aspect': aspect_name,
        'aspect_confidence': aspect_probs[aspect_pred].item(),
        'sentiment': sentiment_name,
        'sentiment_confidence': sentiment_probs[sentiment_pred].item()
    }


# 3. 테스트 리뷰들
test_reviews = [
    "커피가 정말 맛있어요!",
    "빵이 너무 딱딱하고 맛없어요",
    "직원분들이 친절하시네요",
    "가격이 좀 비싼 것 같아요",
    "분위기가 좋고 조용해서 좋아요",
    "주차하기 너무 불편해요",
    "아메리카노 진짜 맛있습니다",
    "케이크가 별로였어요"
]

# 4. 예측 실행
print("=" * 80)
print("카페 리뷰 측면 & 감정 분석 결과")
print("=" * 80)

for review in test_reviews:
    result = predict(review)
    print(f"\n📝 리뷰: {result['text']}")
    print(f"   🏷️  측면: {result['aspect']} (확률: {result['aspect_confidence']:.2%})")
    print(f"   💭 감정: {result['sentiment']} (확률: {result['sentiment_confidence']:.2%})")


# 5. 상세 분석 (모든 측면과 감정의 확률 보기)
def predict_with_details(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        aspect_logits, sentiment_logits = model(inputs['input_ids'], inputs['attention_mask'])

    aspect_probs = torch.softmax(aspect_logits, dim=1)[0]
    sentiment_probs = torch.softmax(sentiment_logits, dim=1)[0]

    print(f"\n{'=' * 80}")
    print(f"📝 리뷰: {text}")
    print(f"{'=' * 80}")

    print("\n📊 측면별 확률:")
    for i, aspect in enumerate(aspect_le.classes_):
        print(f"   {aspect}: {aspect_probs[i].item():.2%}")

    print("\n💭 감정별 확률:")
    for i, sentiment in enumerate(sentiment_le.classes_):
        print(f"   {sentiment}: {sentiment_probs[i].item():.2%}")


# 상세 분석 예시
print("\n\n" + "=" * 80)
print("상세 분석 예시")
predict_with_details("커피는 맛있는데 가격이 너무 비싸요")