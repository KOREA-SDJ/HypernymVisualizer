"""
이 모듈은 CLIP 모델을 로드하고 텍스트 및 이미지 임베딩을 생성하는 기능을 담당합니다.

Hugging Face Transformers 라이브러리를 사용하여 사전 학습된 CLIP 모델을 불러오고,
텍스트와 이미지를 정규화된 벡터 임베딩으로 변환하는 함수들을 제공합니다.
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple, Union
from PIL import Image

# 모델의 중복 로드를 방지하기 위한 싱글톤 패턴용 전역 변수
_CLIP_PROCESSOR: Union[CLIPProcessor, None] = None
_CLIP_MODEL: Union[CLIPModel, None] = None
_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_clip_model(model_name: str = "openai/clip-vit-base-patch32") -> Tuple[CLIPProcessor, CLIPModel]:
    """사전 학습된 CLIP 모델과 프로세서를 로드하고 초기화합니다.

    사용 가능한 장치(CUDA 또는 CPU)에 모델을 로드합니다.
    전역 변수를 활용한 싱글톤 패턴을 사용하여, 함수가 여러 번 호출되더라도
    모델을 메모리에 중복으로 로드하지 않도록 합니다.

    Args:
        model_name (str, optional): Hugging Face Hub에 있는 사전 학습된 모델의 저장소 이름입니다.
            기본값은 "openai/clip-vit-base-patch32"입니다.

    Returns:
        Tuple[CLIPProcessor, CLIPModel]: 다음 요소들을 포함하는 튜플을 반환합니다:
            - processor (CLIPProcessor): 텍스트 토큰화 및 이미지 전처리를 담당하는 프로세서.
            - model (CLIPModel): 적절한 장치로 이동된 로드된 CLIP 모델.

    Raises:
        OSError: 모델 이름을 찾을 수 없거나 로드 중 오류가 발생한 경우.
        Exception: 기타 로딩 관련 예외 발생 시.
    """
    global _CLIP_PROCESSOR, _CLIP_MODEL

    if _CLIP_MODEL is None:
        print(f"CLIP 모델 로드 중: {model_name} (사용 장치: {_DEVICE})...")
        try:
            _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_name)
            _CLIP_MODEL = CLIPModel.from_pretrained(model_name).to(_DEVICE)
            print("CLIP 모델 로드 완료.")
        except Exception as e:
            print(f"CLIP 모델 로드 실패: {e}")
            raise e

    return _CLIP_PROCESSOR, _CLIP_MODEL


def get_text_features(
    texts: List[str], 
    processor: CLIPProcessor, 
    model: CLIPModel
) -> torch.Tensor:
    """텍스트 문자열 리스트에 대한 정규화된 CLIP 임베딩을 생성합니다.

    Args:
        texts (List[str]): 인코딩할 텍스트 문자열들의 리스트입니다.
        processor (CLIPProcessor): 로드된 CLIP 프로세서 객체입니다.
        model (CLIPModel): 로드된 CLIP 모델 객체입니다.

    Returns:
        torch.Tensor: (N, D) 크기의 정규화된 텍스트 임베딩 텐서를 반환합니다.
            여기서 N은 입력 텍스트의 수, D는 임베딩 차원입니다.
            입력 리스트가 비어있으면 빈 텐서를 반환합니다.
    """
    if not texts:
        return torch.empty(0)

    # 텍스트 토큰화 및 모델 입력 준비 (Truncation 및 Padding 적용)
    inputs = processor(
        text=texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )
    
    with torch.no_grad():
        # 텍스트 특징 벡터 추출
        features = model.get_text_features(**inputs.to(_DEVICE))
    
    # 임베딩 벡터 정규화 (L2 Normalization)
    features /= features.norm(dim=-1, keepdim=True)
    
    return features


def get_image_features(
    images: List[Image.Image], 
    processor: CLIPProcessor, 
    model: CLIPModel
) -> torch.Tensor:
    """PIL 이미지 리스트에 대한 정규화된 CLIP 임베딩을 생성합니다.

    Args:
        images (List[Image.Image]): 인코딩할 PIL Image 객체들의 리스트입니다.
        processor (CLIPProcessor): 로드된 CLIP 프로세서 객체입니다.
        model (CLIPModel): 로드된 CLIP 모델 객체입니다.

    Returns:
        torch.Tensor: (N, D) 크기의 정규화된 이미지 임베딩 텐서를 반환합니다.
            여기서 N은 입력 이미지의 수, D는 임베딩 차원입니다.
            입력 리스트가 비어있으면 빈 텐서를 반환합니다.
    """
    if not images:
        return torch.empty(0)

    # 이미지 전처리 및 모델 입력 준비
    inputs = processor(images=images, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        # 이미지 특징 벡터 추출
        features = model.get_image_features(**inputs.to(_DEVICE))

    # 임베딩 벡터 정규화 (L2 Normalization)
    features /= features.norm(dim=-1, keepdim=True)
    
    return features


def calculate_similarity(
    text_features: torch.Tensor, 
    image_features: torch.Tensor
) -> torch.Tensor:
    """텍스트 임베딩과 이미지 임베딩 간의 코사인 유사도 행렬을 계산합니다.

    CLIP 임베딩은 이미 정규화되어 있으므로, 두 벡터의 내적(Dot Product)이
    곧 코사인 유사도와 같습니다.

    Args:
        text_features (torch.Tensor): (M, D) 크기의 텍스트 임베딩 텐서.
        image_features (torch.Tensor): (N, D) 크기의 이미지 임베딩 텐서.

    Returns:
        torch.Tensor: (N, M) 크기의 유사도 행렬을 반환합니다.
            N은 이미지 개수, M은 텍스트 개수입니다.
            결과 텐서는 후속 처리를 위해 CPU로 이동되어 반환됩니다.
            입력 중 하나라도 비어있으면 빈 텐서를 반환합니다.
    """
    if text_features.numel() == 0 or image_features.numel() == 0:
        return torch.empty(0)
    
    # 행렬 곱셈: (N, D) @ (M, D).T -> (N, M)
    similarity_matrix = torch.matmul(image_features, text_features.T)
    
    # 결과를 CPU로 이동하여 반환 (Numpy 변환 등을 용이하게 하기 위함)
    return similarity_matrix.cpu()