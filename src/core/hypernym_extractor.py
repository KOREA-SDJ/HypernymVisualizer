"""멀티모달 상위어 결정(Reasoning) 모듈.

이 모듈은 WordNet의 계층적 구조에서 추출된 상위어 후보군과
CLIP 모델이 계산한 이미지-텍스트 유사도 점수를 결합하여,
주어진 입력 데이터(이미지+텍스트)를 가장 잘 포괄하는 최적의 상위어를 결정합니다.
"""

import torch
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional

from src.core import clip_processor

try:
    from src.external import wordnet_handler
except ImportError:
    print("경고: src.external.wordnet_handler 모듈을 찾을 수 없습니다.")
    wordnet_handler = None


def determine_best_hypernym(
    input_images: List[Image.Image],
    input_texts: List[str],
    clip_model_components: Tuple[Any, Any] = None
) -> Tuple[Optional[str], float]:
    """입력된 이미지와 텍스트 쌍에 대해 최적의 상위어를 결정합니다.

    1. WordNet을 통해 입력 텍스트들의 공통 상위어 후보군을 추출합니다.
    2. CLIP 모델을 사용하여 입력 이미지들과 후보군 텍스트 간의 의미적 유사도를 계산합니다.
    3. WordNet의 깊이 가중치와 CLIP 유사도 점수를 결합하여 최종 점수를 산출합니다.

    Args:
        input_images (List[Image.Image]): 분석할 PIL 이미지 객체 리스트.
        input_texts (List[str]): 각 이미지에 대응하는 텍스트(단어) 리스트.
        clip_model_components (Tuple[Any, Any], optional): 
            이미 로드된 (processor, model) 튜플. 
            None일 경우 함수 내부에서 모델을 새로 로드합니다.

    Returns:
        Tuple[Optional[str], float]: 
            - 결정된 최적 상위어 텍스트 (실패 시 None).
            - 해당 상위어의 최종 신뢰도 점수 (실패 시 0.0).
    """
    
    if not input_images or not input_texts or wordnet_handler is None:
        print("오류: 입력 데이터가 부족하거나 워드넷 핸들러가 로드되지 않았습니다.")
        return None, 0.0

    if clip_model_components:
        processor, model = clip_model_components
    else:
        processor, model = clip_processor.load_clip_model()

    # WordNet 상위어 후보군 및 가중치 추출
    candidates = wordnet_handler.find_common_hypernym_candidates(input_texts)
    
    if not candidates:
        print(f"경고: '{input_texts}'에 대한 공통 상위어를 찾을 수 없습니다.")
        return None, 0.0

    candidate_texts = [c['text'] for c in candidates]
    candidate_weights = torch.tensor([c['weight'] for c in candidates])

    print(f"워드넷 후보군 ({len(candidate_texts)}개): {candidate_texts[:10]}...")

    # CLIP 임베딩 생성
    text_features = clip_processor.get_text_features(candidate_texts, processor, model)
    image_features = clip_processor.get_image_features(input_images, processor, model)

    # CLIP 유사도 행렬 계산 (Images x Candidates)
    similarity_matrix = clip_processor.calculate_similarity(text_features, image_features)
    
    # 평균 유사도 계산 (여러 이미지에 대한 평균 적합도)
    avg_similarity = similarity_matrix.mean(dim=0)
    
    # 최종 점수 계산: CLIP 유사도 * WordNet 가중치
    final_scores = avg_similarity.cpu() * candidate_weights.cpu()

    # 최적 상위어 선정
    best_idx = torch.argmax(final_scores).item()
    best_hypernym = candidate_texts[best_idx]
    best_score = final_scores[best_idx].item()

    print(f"--> 최종 결정: '{best_hypernym}' (점수: {best_score:.4f})")
    
    return best_hypernym, best_score