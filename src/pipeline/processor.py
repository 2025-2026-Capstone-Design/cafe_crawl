"""
processors.py
6단계 파이프라인 기반 텍스트 분류 시스템
Step 1-2: KiwiAnalyzer (품사 보존형 형태소 분석)
Step 3: SBERTMapper (앵커 기반 유사도)
Step 4: ConfidenceMonitor (경계 샘플 필터링)
Step 5: LLM_Refiner (Groq 기반 정제)
Step 6: FinalClassifier (최종 통합)
"""

import re
import time
import json
import glob
import requests
import pandas as pd
import numpy as np
from collections import Counter
from kiwipiepy import Kiwi
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from bertopic.representation._base import BaseRepresentation
import spacy

# 전역 초기화
kiwi = Kiwi()
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


# ============================================================
# 1. 텍스트 전처리
# ============================================================

def clean_text(text):
    """한글 리뷰 텍스트 정제"""
    if not isinstance(text, str):
        return ""

    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    text = re.sub(r'\^+', '', text)
    text = re.sub(r'[ㅋㅎㅠㅜㅡㅗ]+', '', text)
    text = re.sub(r'([~!?.])\1{2,}', r'\1', text)
    text = re.sub(r'([a-zA-Z])\1{3,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text if len(text) >= 5 else ""


def clean_english_text(text):
    """영문 번역 후 노이즈 제거"""
    if not isinstance(text, str):
        return ""

    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )

    text = emoji_pattern.sub('', text)
    text = re.sub(r'([~!?.,:;])\1{2,}', r'\1', text)
    text = re.sub(r'[ㄱ-ㅎㅏ-ㅣ]+', '', text)
    text = re.sub(r'[가-힣]+', '', text)
    text = re.sub(r'[\u200b-\u200f\u2028-\u202f\u205f-\u206f]', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'\"-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text if len(text) >= 5 else ""


# ============================================================
# 2. 데이터 로드
# ============================================================

def load_and_split_sentences(folder_path, max_reviews=10000):
    """JSONL 폴더에서 문장 단위 데이터 로드"""
    all_bodies = []
    jsonl_files = glob.glob(f"{folder_path}/*.jsonl")

    print(f"총 {len(jsonl_files)}개 파일 발견")

    for file_path in jsonl_files:
        if len(all_bodies) >= max_reviews:
            break

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    body = data.get('body', '')
                    if body and len(body) > 5:
                        all_bodies.append(body)

                    if len(all_bodies) >= max_reviews:
                        break
                except:
                    continue

    # 문장 분리
    sentences = []
    for review in all_bodies:
        sents = kiwi.split_into_sents(review)
        sentences.extend([sent.text for sent in sents])

    df = pd.DataFrame({'text': sentences})
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)

    print(f"{len(all_bodies)}건 리뷰 → {len(df)}개 고유 문장 추출")
    return df


# ============================================================
# 3. 계층적 샘플링
# ============================================================

def stratified_sampling_for_translation(df, rare_keywords, sample_size=2000, seed=42):
    """다양성을 보장하는 계층적 샘플링"""
    print("\n🎯 계층적 샘플링 시작...")

    df['text_clean'] = df['text'].apply(clean_text)
    df = df[df['text_clean'].str.len() >= 10].copy()
    df['text_len'] = df['text_clean'].str.len()

    print(f"전처리 후: {len(df)}개 문장")

    samples = []

    # 희귀 측면 우선 샘플링
    print("\n📌 희귀 측면 우선 샘플링:")
    for category, keywords in rare_keywords.items():
        pattern = '|'.join(keywords)
        mask = df['text_clean'].str.contains(pattern, case=False, na=False)

        if mask.sum() > 0:
            n_samples = min(50, mask.sum())
            category_samples = df[mask].sample(n=n_samples, random_state=seed)
            samples.append(category_samples)
            print(f"  - {category}: {n_samples}개 선택")

    # 선택된 문장 제외
    selected_indices = pd.concat(samples).index if samples else pd.Index([])
    df_remaining = df[~df.index.isin(selected_indices)].copy()

    # 길이별 균등 샘플링
    current_count = len(selected_indices)
    remaining_needed = sample_size - current_count

    if remaining_needed > 0:
        print(f"\n📊 길이별 균등 샘플링 ({remaining_needed}개):")

        length_targets = {
            'short': (0, 30, 0.2),
            'medium': (30, 60, 0.5),
            'long': (60, 999, 0.3)
        }

        for length_type, (min_len, max_len, ratio) in length_targets.items():
            mask = (df_remaining['text_len'] >= min_len) & (df_remaining['text_len'] < max_len)
            available = df_remaining[mask]
            n_samples = min(int(remaining_needed * ratio), len(available))

            if n_samples > 0:
                length_samples = available.sample(n=n_samples, random_state=seed)
                samples.append(length_samples)
                print(f"  - {length_type} ({min_len}~{max_len}자): {n_samples}개")

    df_sample = pd.concat(samples).drop_duplicates(subset=['text_clean'])
    df_sample = df_sample.sample(n=min(sample_size, len(df_sample)), random_state=seed)

    print(f"\n✅ 최종 샘플: {len(df_sample)}개")

    df_sample['text'] = df_sample['text_clean']
    return df_sample.drop(columns=['text_clean', 'text_len'])


# ============================================================
# 4. 번역
# ============================================================

def sample_and_translate(df, rare_keywords, sample_size=2000, cache_file='translated_sample.csv'):
    """샘플링 + 번역 + 정제"""

    # 캐시 확인
    try:
        df_cached = pd.read_csv(cache_file)
        if len(df_cached) >= sample_size * 0.9:
            print(f"✅ 캐시 로드: {len(df_cached)}건")
            df_cached['text_en'] = df_cached['text_en'].apply(clean_english_text)
            df_cached = df_cached[df_cached['text_en'].str.len() > 0].copy()
            return df_cached
    except FileNotFoundError:
        pass

    # 샘플링
    df_sample = stratified_sampling_for_translation(df, rare_keywords, sample_size)

    # 번역
    print(f"\n🌐 번역 시작: {len(df_sample)}건...")
    translator = GoogleTranslator(source='ko', target='en')

    df_sample['text_en'] = None
    failed_count = 0

    for idx, row in df_sample.iterrows():
        try:
            translated = translator.translate(str(row['text']))
            df_sample.at[idx, 'text_en'] = translated

            if len(df_sample[df_sample['text_en'].notna()]) % 100 == 0:
                print(f"번역 진행: {len(df_sample[df_sample['text_en'].notna()])} / {len(df_sample)}")
                df_sample.to_csv(cache_file, index=False)

            time.sleep(0.1)
        except Exception as e:
            df_sample.at[idx, 'text_en'] = ""
            failed_count += 1
            time.sleep(1)

    # 영문 정제
    print("\n🧹 영문 텍스트 정제 중...")
    df_sample['text_en'] = df_sample['text_en'].apply(clean_english_text)
    df_sample = df_sample[df_sample['text_en'].str.len() > 0].copy()

    df_sample.to_csv(cache_file, index=False)
    print(f"\n✅ 번역 완료: {len(df_sample)}건")

    return df_sample


# ============================================================
# 5. 키워드 확장 (SBERT)
# ============================================================

def expand_english_keyword_dict(df_sample, base_dict_en, stopwords, noise_words):
    """SBERT로 영문 키워드 확장"""
    print("\n🔍 SBERT 기반 키워드 확장 시작...")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    clean_sentences = df_sample['text_en'].dropna().astype(str).tolist()

    expanded_dict = {aspect: list(keywords) for aspect, keywords in base_dict_en.items()}
    aspect_embs = {cat: model.encode(" ".join(words)) for cat, words in base_dict_en.items()}
    sentence_embeddings = model.encode(clean_sentences, show_progress_bar=True)

    for aspect, target_emb in aspect_embs.items():
        sims = cosine_similarity([target_emb], sentence_embeddings)[0]
        top_indices = np.where(sims > 0.45)[0]

        new_kws = set()
        for idx in top_indices:
            doc = nlp(clean_sentences[idx].lower())
            for token in doc:
                if token.pos_ in ["NOUN", "ADJ"] and len(token.text) > 2:
                    if token.text not in stopwords and token.text not in noise_words:
                        new_kws.add(token.text)

        expanded_dict[aspect].extend(list(new_kws))
        expanded_dict[aspect] = list(set(expanded_dict[aspect]))
        print(f"✅ {aspect}: {len(expanded_dict[aspect])}개")

    return expanded_dict


# ============================================================
# 6. 한글 번역
# ============================================================

def translate_keywords_to_korean(aspect_keywords_en, essential_kr, cache_file='korean_keywords.json'):
    """영문 키워드 → 한글 번역"""

    # 캐시 확인
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            korean_dict = json.load(f)
        if korean_dict:
            print(f"✅ 한글 사전 로드: {cache_file}")
            return korean_dict
    except FileNotFoundError:
        pass

    translator = GoogleTranslator(source='en', target='ko')
    raw_korean_dict = {}

    print("\n🌐 영문 → 한글 번역 시작...")

    for aspect, keywords_en in aspect_keywords_en.items():
        print(f"\n🔍 [{aspect}] {len(keywords_en)}개 번역...", end=" ")
        translated_set = set()

        for kw in keywords_en:
            try:
                translated = translator.translate(kw).strip()
                if len(translated) > 1 and not translated.isdigit():
                    translated = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', translated)
                    translated_set.add(translated)
                time.sleep(0.1)
            except:
                time.sleep(1)
                continue

        raw_korean_dict[aspect] = list(translated_set)
        print(f"완료 ({len(translated_set)}개)")

    # 중복 제거
    print("\n⚖️ 중복 키워드 제거 중...")
    all_words = []
    for kws in raw_korean_dict.values():
        all_words.extend(kws)

    counts = Counter(all_words)
    duplicates = {kw for kw, count in counts.items() if count > 1}

    final_korean_dict = {}
    for aspect, kws in raw_korean_dict.items():
        clean_kws = [kw for kw in kws if kw not in duplicates]
        final_korean_dict[aspect] = sorted(clean_kws)

    # 필수 단어 추가
    for aspect, essentials in essential_kr.items():
        if aspect in final_korean_dict:
            final_korean_dict[aspect] = list(set(final_korean_dict[aspect] + essentials))

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(final_korean_dict, f, ensure_ascii=False, indent=2)

    print(f"\n💾 한글 사전 저장: {cache_file}")
    return final_korean_dict


# ============================================================
# STEP 1-2: KiwiAnalyzer (품사 보존형 형태소 분석)
# ============================================================

class KiwiAnalyzer:
    """품사 태그 보존형 형태소 분석기"""

    def __init__(self):
        self.kiwi = Kiwi()
        self.target_pos = ['VA', 'VV', 'NNG', 'NNP', 'MAG', 'XR']

    def analyze_with_pos(self, text):
        """품사 태그를 포함한 토큰 반환"""
        tokens = self.kiwi.tokenize(text)
        return [(t.form, t.tag) for t in tokens if t.tag in self.target_pos]

    def get_morphs_for_matching(self, text):
        """매칭용 정규화된 형태소 딕셔너리 (품사별 가중치 적용 가능)"""
        analyzed = self.analyze_with_pos(text)
        return {form: tag for form, tag in analyzed}

    def extract_stems_dict(self, keyword_dict):
        """키워드 사전에서 어간 추출"""
        print("\n✂️ 한글 어간 추출 중...")
        stems_dict = {}

        for aspect, keywords in keyword_dict.items():
            stems = set()
            for kw in keywords:
                tokens = self.kiwi.tokenize(kw)
                for t in tokens:
                    if t.tag in self.target_pos:
                        stems.add(t.form)

            stems_dict[aspect] = list(stems)
            print(f"  [{aspect}]: {len(stems)}개 어간")

        return stems_dict


# ============================================================
# STEP 3: SBERTMapper (앵커 기반 유사도 계산)
# ============================================================

class SBERTMapper:
    """앵커 문장 기반 유사도 매핑"""

    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
        self.anchor_embeddings = {}
        self.aspect_names = []

    def set_anchors(self, aspect_keywords_en):
        """각 측면별 앵커 문장 임베딩 생성"""
        print("\n🎯 앵커 임베딩 생성 중...")

        self.aspect_names = list(aspect_keywords_en.keys())

        for aspect, keywords in aspect_keywords_en.items():
            # 키워드를 자연스러운 문장으로 변환
            kw_sample = keywords[:5] if len(keywords) >= 5 else keywords
            anchor_text = f"This cafe has good {', '.join(kw_sample)}"
            self.anchor_embeddings[aspect] = self.model.encode(anchor_text)
            print(f"  ✓ {aspect}: {anchor_text[:50]}...")

    def compute_similarities(self, texts):
        """모든 텍스트에 대해 앵커별 유사도 계산"""
        print(f"\n🔍 유사도 계산 중: {len(texts)}건...")

        text_embs = self.model.encode(texts, show_progress_bar=True)

        results = []
        for text, emb in zip(texts, text_embs):
            scores = {}
            for aspect, anchor_emb in self.anchor_embeddings.items():
                scores[aspect] = float(cosine_similarity([emb], [anchor_emb])[0][0])
            results.append(scores)

        return results  # [{aspect: score, ...}, ...]


# ============================================================
# STEP 4: ConfidenceMonitor (경계 샘플 필터링)
# ============================================================

class ConfidenceMonitor:
    """신뢰도 기반 경계 샘플 탐지"""

    def __init__(self, high_threshold=0.6, low_threshold=0.35, margin_threshold=0.15):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.margin_threshold = margin_threshold

    def classify_by_confidence(self, similarity_scores):
        """
        유사도 점수를 기반으로 샘플 분류

        Returns:
            confident: 확신 있는 샘플 (바로 라벨링)
            boundary: 경계 샘플 (LLM 검증 필요)
            uncertain: 불확실 샘플 (제외)
        """
        confident = []
        boundary = []
        uncertain = []

        for idx, scores in enumerate(similarity_scores):
            if not scores:
                uncertain.append(idx)
                continue

            max_score = max(scores.values())
            max_aspect = max(scores, key=scores.get)

            # 2등과의 차이 계산
            sorted_scores = sorted(scores.values(), reverse=True)
            margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0

            if max_score >= self.high_threshold and margin > self.margin_threshold:
                # 확신 있는 샘플
                confident.append({
                    'index': idx,
                    'aspect': [max_aspect],
                    'confidence': max_score,
                    'method': 'sbert_high_conf'
                })
            elif max_score >= self.low_threshold:
                # 경계 영역 - LLM 검증 필요
                candidates = [asp for asp, sc in scores.items()
                             if sc >= self.low_threshold]
                boundary.append({
                    'index': idx,
                    'candidates': candidates,
                    'scores': scores,
                    'max_score': max_score,
                    'margin': margin
                })
            else:
                # 불확실 샘플
                uncertain.append(idx)

        print(f"\n📊 신뢰도 필터링 결과:")
        print(f"  ✅ 확신: {len(confident)}건 (바로 라벨링)")
        print(f"  🤔 경계: {len(boundary)}건 (LLM 검증 필요)")
        print(f"  ❌ 불확실: {len(uncertain)}건 (제외)")

        return confident, boundary, uncertain


# ============================================================
# STEP 5: LLM_Refiner (Groq 기반 경계 샘플 정제)
# ============================================================

class LLM_Refiner:
    """Groq를 이용한 경계 샘플 정제 및 키워드 학습"""

    def __init__(self, groq_api_key, model="llama-3.3-70b-versatile"):
        self.api_key = groq_api_key
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        self.discovered_keywords = {}  # 새로 발견된 키워드 저장
        self.refinement_stats = {'success': 0, 'failed': 0, 'skipped': 0}

    def refine_boundary_samples(self, df, boundary_samples, aspect_keywords_en, max_samples=None):
        """경계 샘플의 진짜 라벨 결정 및 키워드 학습"""
        print(f"\n🔬 LLM 정제 시작: {len(boundary_samples)}건...")

        refined_labels = []
        samples_to_process = boundary_samples[:max_samples] if max_samples else boundary_samples

        for i, sample in enumerate(samples_to_process):
            idx = sample['index']
            text = df.iloc[idx]['text']
            candidates = sample['candidates']
            scores = sample['scores']

            if (i + 1) % 10 == 0:
                print(f"  진행: {i+1}/{len(samples_to_process)}")

            # Groq에게 물어보기
            chosen_aspect, new_keywords, confidence = self._ask_llm(
                text, candidates, scores, aspect_keywords_en
            )

            if chosen_aspect:
                refined_labels.append({
                    'index': idx,
                    'aspect': [chosen_aspect],
                    'method': 'llm_refined',
                    'confidence': confidence,
                    'new_keywords': new_keywords
                })

                # 새 키워드 수집
                if new_keywords:
                    if chosen_aspect not in self.discovered_keywords:
                        self.discovered_keywords[chosen_aspect] = set()
                    self.discovered_keywords[chosen_aspect].update(new_keywords)

                self.refinement_stats['success'] += 1
            else:
                self.refinement_stats['failed'] += 1

            time.sleep(0.5)  # Rate limit

        print(f"\n✅ LLM 정제 완료:")
        print(f"  성공: {self.refinement_stats['success']}건")
        print(f"  실패: {self.refinement_stats['failed']}건")

        return refined_labels

    def _ask_llm(self, text, candidates, scores, aspect_keywords_en):
        """Groq API로 진짜 측면 판단"""

        # 후보 측면 설명 생성
        candidate_desc = []
        for asp in candidates:
            kws = aspect_keywords_en.get(asp, [])[:5]
            score = scores.get(asp, 0)
            candidate_desc.append(f"- {asp} (score: {score:.2f}): {', '.join(kws)}")

        prompt = f"""You are analyzing a Korean cafe review.

Review: "{text}"

Candidate aspects (with similarity scores):
{chr(10).join(candidate_desc)}

Task:
1. Choose the MOST relevant aspect from the candidates above
2. Suggest 2-3 new English keywords that represent this aspect in the review
3. Rate your confidence (0.0-1.0)

Output ONLY valid JSON:
{{
  "aspect": "chosen_aspect_name",
  "new_keywords": ["keyword1", "keyword2"],
  "confidence": 0.85
}}"""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=20
            )

            if response.status_code != 200:
                return None, [], 0.0

            result = response.json()
            content = json.loads(result['choices'][0]['message']['content'])

            aspect = content.get('aspect')
            keywords = content.get('new_keywords', [])
            confidence = content.get('confidence', 0.5)

            # 유효성 검사
            if aspect not in candidates:
                return None, [], 0.0

            # 키워드 정제
            clean_keywords = [kw.strip().lower() for kw in keywords
                             if isinstance(kw, str) and len(kw) > 2 and kw.isalpha()]

            return aspect, clean_keywords, confidence

        except Exception as e:
            print(f"\n⚠️ LLM 오류: {e}")
            return None, [], 0.0

    def export_discovered_keywords(self, filename='discovered_keywords.json'):
        """발견된 키워드를 파일로 저장"""
        export_dict = {k: list(v) for k, v in self.discovered_keywords.items()}

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_dict, f, indent=2, ensure_ascii=False)

        print(f"\n💾 발견된 키워드 저장: {filename}")

        for aspect, keywords in export_dict.items():
            if keywords:
                print(f"  {aspect}: {keywords[:5]}")

    def get_enhanced_keywords(self, original_keywords_en):
        """원본 키워드에 발견된 키워드 병합"""
        enhanced = {}
        for aspect, orig_kws in original_keywords_en.items():
            enhanced[aspect] = list(set(orig_kws))
            if aspect in self.discovered_keywords:
                enhanced[aspect].extend(list(self.discovered_keywords[aspect]))
                enhanced[aspect] = list(set(enhanced[aspect]))

        return enhanced


# ============================================================
# 기존 키워드 기반 분류 (개선)
# ============================================================

def classify_with_korean_keywords(df, aspect_keywords_kr, analyzer=None):
    """한글 키워드 기반 분류 (형태소 분석 활용)"""
    print("\n🔍 키워드 기반 분류 시작...")

    if analyzer is None:
        analyzer = KiwiAnalyzer()

    df['aspect'] = None
    df['classification_method'] = None

    # 키워드를 형태소로 변환
    aspect_morphs = {}
    for aspect, keywords in aspect_keywords_kr.items():
        morphs = set()
        for kw in keywords:
            morph_dict = analyzer.get_morphs_for_matching(kw)
            morphs.update(morph_dict.keys())
        aspect_morphs[aspect] = morphs

    for idx, row in df.iterrows():
        text = row['text']
        text_morphs = set(analyzer.get_morphs_for_matching(text).keys())

        found_aspects = []
        for aspect, k_morphs in aspect_morphs.items():
            if not text_morphs.isdisjoint(k_morphs):
                found_aspects.append(aspect)

        if found_aspects:
            df.at[idx, 'aspect'] = list(set(found_aspects))
            df.at[idx, 'classification_method'] = 'keyword'

    labeled = df[df['aspect'].notna()].copy()
    unlabeled = df[df['aspect'].isna()].copy()

    print(f"✅ 라벨링: {len(labeled)}건 ({len(labeled)/len(df)*100:.1f}%)")
    print(f"❌ 미분류: {len(unlabeled)}건")

    return labeled, unlabeled


# ============================================================
# STEP 6: FinalClassifier (최종 통합)
# ============================================================

def classify_with_pipeline(df, aspect_keywords_en, aspect_keywords_kr, groq_api_key):
    """6단계 파이프라인 실행"""

    print("\n" + "=" * 60)
    print("🚀 6단계 분류 파이프라인 시작")
    print("=" * 60)

    # Step 1-2: 형태소 분석
    analyzer = KiwiAnalyzer()
    df['morphs'] = df['text'].apply(analyzer.get_morphs_for_matching)

    # 키워드 매칭 (기존 방식)
    labeled_kw, unlabeled = classify_with_korean_keywords(df, aspect_keywords_kr)
    print(f"\n📍 Step 1-2 완료: {len(labeled_kw)}건 키워드 매칭")

    # Step 3: SBERT 유사도 계산
    mapper = SBERTMapper()
    mapper.set_anchors(aspect_keywords_en)
    similarity_scores = mapper.compute_similarities(unlabeled['text'].tolist())
    print(f"\n📍 Step 3 완료: 유사도 계산")

    # Step 4: 신뢰도 기반 필터링
    monitor = ConfidenceMonitor()
    confident, boundary, uncertain = monitor.classify_by_confidence(similarity_scores)
    print(f"\n📍 Step 4 완료: 경계 샘플 {len(boundary)}건 탐지")

    # Step 5: LLM으로 경계 샘플 정제
    refiner = LLM_Refiner(groq_api_key)
    refined = refiner.refine_boundary_samples(unlabeled, boundary, aspect_keywords_en)
    refiner.export_discovered_keywords()
    print(f"\n📍 Step 5 완료: {len(refined)}건 정제")

    # Step 6: 최종 통합
    final_df = pd.concat([
        labeled_kw,
        _build_df_from_results(unlabeled, confident),
        _build_df_from_results(unlabeled, refined)
    ], ignore_index=True)

    print(f"\n✅ 최종 분류 완료: {len(final_df)}건 / {len(df)}건")
    print("=" * 60)

    return final_df


def _build_df_from_results(original_df, results):
    """인덱스 기반으로 결과를 DataFrame으로 변환"""
    rows = []
    for r in results:
        row = original_df.iloc[r['index']].to_dict()
        row['aspect'] = r['aspect']
        row['classification_method'] = r.get('method', 'unknown')
        rows.append(row)
    return pd.DataFrame(rows)