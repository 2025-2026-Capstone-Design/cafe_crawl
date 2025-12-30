import json

import torch
import pickle
from transformers import BertTokenizer
import re

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


# 2. 개선된 문장 분리 함수
def split_sentences(text):
    """
    한국어 접속사와 연결어미를 기준으로 문장 분리
    """
    # 먼저 명확한 구두점으로 1차 분리
    sentences = []

    # 접속사와 연결어미 패턴 (더 많이 추가)
    patterns = [
        r'는데\s+',  # ~는데
        r'은데\s+',  # ~은데
        r'ㄴ데\s+',  # ~ㄴ데
        r'지만\s+',  # ~지만
        r'하고\s+',  # ~하고
        r'고\s+',  # ~고
        r'며\s+',  # ~며
        r'면서\s+',  # ~면서
        r'[,]\s*',  # 쉼표
        r'그런데\s+',  # 그런데
        r'근데\s+',  # 근데
        r'하지만\s+',  # 하지만
        r'그리고\s+',  # 그리고
    ]

    # 모든 패턴을 하나로 합치기
    combined_pattern = '|'.join(patterns)

    # 분리하되, 구분자도 포함해서 어디서 나뉘었는지 확인
    parts = re.split(f'({combined_pattern})', text)

    current = ""
    for part in parts:
        if part.strip():
            current += part
            # 접속사나 연결어미를 만나면 문장 분리
            if re.match(combined_pattern, part):
                if current.strip() and len(current.strip()) > 2:
                    # 접속사/연결어미 제거하고 저장
                    cleaned = re.sub(combined_pattern, '', current).strip()
                    if cleaned:
                        sentences.append(cleaned)
                current = ""

    # 마지막 남은 부분 추가
    if current.strip() and len(current.strip()) > 2:
        cleaned = re.sub(combined_pattern, '', current).strip()
        if cleaned:
            sentences.append(cleaned)

    return sentences if sentences else [text]


# 3. 단일 문장 예측 함수
def predict_single(text):
    """단일 문장에 대한 예측"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        aspect_logits, sentiment_logits = model(inputs['input_ids'], inputs['attention_mask'])

    # 측면 예측
    aspect_pred = torch.argmax(aspect_logits, dim=1).item()
    aspect_name = aspect_le.inverse_transform([aspect_pred])[0]
    aspect_probs = torch.softmax(aspect_logits, dim=1)[0]

    # 감정 예측
    sentiment_pred = torch.argmax(sentiment_logits, dim=1).item()
    sentiment_name = sentiment_le.inverse_transform([sentiment_pred])[0]
    sentiment_probs = torch.softmax(sentiment_logits, dim=1)[0]

    return {
        'text': text,
        'aspect': aspect_name,
        'aspect_confidence': aspect_probs[aspect_pred].item(),
        'sentiment': sentiment_name,
        'sentiment_confidence': sentiment_probs[sentiment_pred].item(),
        'all_aspect_probs': aspect_probs,
        'all_sentiment_probs': sentiment_probs
    }


# 4. 복합측면 분석 함수
def predict_multi_aspect(text, threshold=0.25):
    """
    문장을 분리해서 복합측면 분석
    threshold: 측면으로 인정할 최소 확률 (기본 25%)
    """
    # 문장 분리
    sentences = split_sentences(text)

    print(f"[DEBUG] 분리된 문장 수: {len(sentences)}")
    for i, s in enumerate(sentences, 1):
        print(f"[DEBUG] 문장 {i}: '{s}'")

    # 분리된 문장이 없으면 전체 문장으로 분석
    if not sentences or len(sentences) == 0:
        sentences = [text]

    # 각 문장별 분석 결과 저장
    sentence_results = []
    all_aspects = {}  # {측면: [(감정, 확률, 문장)]}

    for sentence in sentences:
        result = predict_single(sentence)
        sentence_results.append(result)

        # 측면별로 결과 정리
        aspect = result['aspect']
        sentiment = result['sentiment']
        confidence = result['aspect_confidence']

        # threshold 이상인 경우만 포함
        if confidence >= threshold:
            if aspect not in all_aspects:
                all_aspects[aspect] = []
            all_aspects[aspect].append({
                'sentiment': sentiment,
                'confidence': confidence,
                'sentence': sentence
            })

    return {
        'original_text': text,
        'sentences': sentence_results,
        'aspects': all_aspects,
        'is_multi_aspect': len(all_aspects) > 1
    }


# 5. 결과 출력 함수
def print_result(result):
    print(f"\n{'=' * 80}")
    print(f"📝 원문: {result['original_text']}")
    print(f"{'=' * 80}")

    if result['is_multi_aspect']:
        print("🔍 복합측면 감지됨!")

    print(f"\n📊 감지된 측면: {len(result['aspects'])}개")

    for aspect, details in result['aspects'].items():
        print(f"\n🏷️  {aspect}")
        for detail in details:
            print(f"   └─ '{detail['sentence']}'")
            print(f"      💭 감정: {detail['sentiment']} (확률: {detail['confidence']:.2%})")

    print(f"\n📄 문장별 상세 분석:")
    for i, sent_result in enumerate(result['sentences'], 1):
        print(f"   [{i}] {sent_result['text']}")
        print(f"       측면: {sent_result['aspect']} ({sent_result['aspect_confidence']:.2%})")
        print(f"       감정: {sent_result['sentiment']} ({sent_result['sentiment_confidence']:.2%})")


# 6. 테스트 리뷰들 (복합측면 포함)
test_reviews = [
    "커피가 정말 맛있어요!",
    "빵이 너무 딱딱하고 맛없어요",
    "커피는 맛있는데 가격이 너무 비싸요",
    "직원분들이 친절하고 분위기도 좋아요",
    "아메리카노는 괜찮은데 케이크가 별로였어요",
    "가격은 비싸지만 음료 맛이 좋고 주차도 편해요",
    "분위기가 좋고 조용해서 좋은데 직원이 불친절해요",
    "커피 맛도 좋고 빵도 맛있고 인테리어도 예뻐요"
]

# 7. 예측 실행
print("=" * 80)
print("카페 리뷰 복합측면 & 감정 분석 결과")
print("=" * 80)

for review in test_reviews:
    result = predict_multi_aspect(review, threshold=0.25)
    print_result(result)


# 8. 상세 분석 함수 (특정 리뷰 집중 분석)
def analyze_in_depth(text):
    """특정 리뷰에 대한 심층 분석"""
    print(f"\n{'=' * 80}")
    print(f"🔬 심층 분석")
    print(f"{'=' * 80}")
    print(f"원문: {text}")

    sentences = split_sentences(text)
    print(f"\n분리된 문장 수: {len(sentences)}개")

    for i, sentence in enumerate(sentences, 1):
        print(f"\n[문장 {i}] {sentence}")

        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            aspect_logits, sentiment_logits = model(inputs['input_ids'], inputs['attention_mask'])

        aspect_probs = torch.softmax(aspect_logits, dim=1)[0]
        sentiment_probs = torch.softmax(sentiment_logits, dim=1)[0]

        print("  측면별 확률:")
        for j, aspect in enumerate(aspect_le.classes_):
            if aspect_probs[j].item() > 0.1:  # 10% 이상만 표시
                print(f"    {aspect}: {aspect_probs[j].item():.2%}")

        print("  감정별 확률:")
        for j, sentiment in enumerate(sentiment_le.classes_):
            print(f"    {sentiment}: {sentiment_probs[j].item():.2%}")


# 9. 심층 분석 예시
print("\n\n" + "=" * 80)
print("심층 분석 예시")
analyze_in_depth("커피는 정말 맛있는데 가격이 너무 비싸고 주차가 불편해요")
def process_jsonl_file(input_path, output_path=None):
    """
    JSONL 파일을 읽어서 각 리뷰의 'body'를 분석하고 결과를 반환/저장합니다.
    """
    results = []

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx == 50:
                    break

                # 1. JSON 데이터 파싱
                data = json.loads(line)
                review_text = data.get('body', '')
                author_id = data.get('author_id', 'unknown')

                if not line.strip:
                    continue

                # 2. 기존 분석 함수 호출 (복합 측면 분석)
                analysis_result = predict_multi_aspect(review_text)

                # 3. 결과 데이터 결합 (기존 정보 + 분석 결과)
                combined_result = {
                    'author_id': author_id,
                    'original_body': review_text,
                    'visit_count': data.get('visit_count'),
                    'visit_time': data.get('visit_time'),
                    'analysis': {
                        'is_multi_aspect': analysis_result['is_multi_aspect'],
                        'extracted_aspects': []
                    }
                }

                # 분석된 측면 정보를 정리해서 추가
                for aspect, details in analysis_result['aspects'].items():
                    for detail in details:
                        combined_result['analysis']['extracted_aspects'].append({
                            'aspect': aspect,
                            'sentiment': detail['sentiment'],
                            'confidence': round(detail['confidence'], 4),
                            'sub_sentence': detail['sentence']
                        })

                results.append(combined_result)

                # 결과 출력 (확인용)
                print(f"✅ 분석 완료 ({author_id}): {review_text[:30]}...")

        # 4. 결과 저장 (선택 사항)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for res in results:
                    f_out.write(json.dumps(res, ensure_ascii=False) + '\n')
            print(f"\n💾 분석 결과가 {output_path}에 저장되었습니다.")

    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {input_path}")

    return results


# --- 실행 부분 ---
input_file = r"C:\Users\Algor\PycharmProjects\cafe_crawl\src\cafe_reviews\11690063_reviews.jsonl"
output_file = "analyzed_reviews.jsonl"

# 분석 실행
final_results = process_jsonl_file(input_file, output_file)
def sentiment_style(sentiment):
    if sentiment.lower() == "positive":
        return "😊", "\033[92m"   # 초록
    elif sentiment.lower() == "negative":
        return "😡", "\033[91m"   # 빨강
    elif sentiment.lower() == "neutral":
        return "😐", "\033[93m"   # 노랑
    else:
        return "❓", "\033[0m"

RESET = "\033[0m"

# 상위 3개만 콘솔에 예쁘게 출력
for res in final_results[:3]:
    print("\n" + "═" * 60)
    print(f"🆔 ID: {res['author_id']}")
    print(f"📝 본문: {res['original_body']}")

    for aspect_info in res['analysis']['extracted_aspects']:
        emoji, color = sentiment_style(aspect_info['sentiment'])
        confidence = aspect_info['confidence'] * 100

        print(
            f" {emoji} {color}[{aspect_info['aspect']}] "
            f"{aspect_info['sentiment'].upper()} "
            f"({confidence:.1f}%)"
            f"{RESET}"
        )
