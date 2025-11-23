"""
이 모듈은 워드넷 후보군과 CLIP 멀티모달 점수를 결합하여 최적의 상위어를 추출합니다.

입력된 이미지와 텍스트를 기반으로 워드넷에서 공통 상위어 후보를 찾고,
CLIP 모델을 통해 시각적 유사도를 계산한 뒤, 가중치를 적용하여 가장 적절한
단 하나의 상위어(Hypernym)를 결정합니다.
"""

import torch
from PIL import Image
from typing import List, Tuple, Dict, Any

# 내부 모듈 임포트
from src.core import clip_processor

try:
    from src.external import wordnet_handler
except ImportError:
    print("경고: src.external.wordnet_handler 모듈을 찾을 수 없습니다. (아직 미구현)")
    wordnet_handler = None


def determine_best_hypernym(
    input_images: List[Image.Image],
    input_texts: List[str],
    clip_model_components: Tuple[Any, Any] = None
) -> Tuple[str, float]:
    """입력된 이미지와 텍스트 쌍에 대해 최적의 상위어를 결정합니다.

    1. 워드넷을 통해 텍스트들의 공통 상위어 후보군과 가중치를 가져옵니다.
    2. CLIP을 통해 후보군과 이미지 간의 유사도를 계산합니다.
    3. 유사도와 가중치를 결합하여 최종 점수를 산출합니다.

    Args:
        input_images (List[Image.Image]): 입력 이미지 리스트.
        input_texts (List[str]): 입력 텍스트 리스트.
        clip_model_components (Tuple, optional): (processor, model) 튜플. 
            None일 경우 clip_processor 내부에서 로드합니다.

    Returns:
        Tuple[str, float]: (최종 결정된 상위어 텍스트, 해당 상위어의 최종 점수)
            실패 시 (None, 0.0)을 반환합니다.
    """
    
    # 0. 유효성 검사
    if not input_images or not input_texts:
        print("오류: 입력 이미지 또는 텍스트가 없습니다.")
        return None, 0.0

    # 1. CLIP 모델 준비
    if clip_model_components:
        processor, model = clip_model_components
    else:
        processor, model = clip_processor.load_clip_model()

    # 2. [WordNet] 상위어 후보군 및 가중치 추출
    # wordnet_handler가 구현되어 있어야 함
    if wordnet_handler is None:
        raise ImportError("wordnet_handler 모듈이 구현되지 않았습니다.")

    # candidates: [{'text': 'sporting goods', 'weight': 1.0}, ...]
    candidates = wordnet_handler.find_common_hypernym_candidates(input_texts)
    
    if not candidates:
        print(f"경고: '{input_texts}'에 대한 공통 상위어를 찾을 수 없습니다.")
        return None, 0.0

    # 후보 텍스트 리스트와 가중치 리스트 분리
    candidate_texts = [c['text'] for c in candidates]
    candidate_weights = torch.tensor([c['weight'] for c in candidates])

    print(f"워드넷 후보군 ({len(candidate_texts)}개): {candidate_texts}")

    # 3. [CLIP] 임베딩 생성
    # 상위어 후보(Text) 임베딩
    text_features = clip_processor.get_text_features(candidate_texts, processor, model)
    # 입력 이미지(Image) 임베딩
    image_features = clip_processor.get_image_features(input_images, processor, model)

    # 4. [CLIP] 유사도 행렬 계산 (Images x Candidates)
    similarity_matrix = clip_processor.calculate_similarity(text_features, image_features)
    
    # 5. 점수 계산 및 결합
    # 이미지별 유사도의 평균을 구함 -> (Candidates,) 크기의 텐서
    avg_similarity = similarity_matrix.mean(dim=0)
    
    # 워드넷 가중치 적용 (Element-wise multiplication)
    # Score = CLIP_Similarity * WordNet_Weight
    # candidate_weights를 CPU/GPU 장치 맞춰줘야 함 (여기선 CPU 가정)
    final_scores = avg_similarity * candidate_weights

    # 6. 최적 상위어 선정 (Argmax)
    best_idx = torch.argmax(final_scores).item()
    best_hypernym = candidate_texts[best_idx]
    best_score = final_scores[best_idx].item()

    print(f"--> 최종 결정: '{best_hypernym}' (점수: {best_score:.4f})")
    
    return best_hypernym, best_score