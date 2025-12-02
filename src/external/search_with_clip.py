"""CLIP 기반 이미지 재순위화(Reranking) 모듈.

이 모듈은 Google Custom Search API를 사용하여 다수의 후보 이미지를 검색한 뒤,
CLIP 모델을 활용하여 텍스트 쿼리(상위어)와 의미적으로 가장 유사한 이미지를
선별(Reranking)하는 기능을 제공합니다.
"""

import requests
import torch
from typing import Dict, Optional, Any, List, Tuple
from PIL import Image
from io import BytesIO

from src.core import clip_processor


def _download_and_process_images(items: List[Dict[str, Any]]) -> Tuple[List[Image.Image], List[str]]:
    """검색 결과에서 이미지를 다운로드하고 PIL 객체로 변환합니다.

    Google API 검색 결과 리스트를 순회하며 이미지를 다운로드합니다.
    다운로드에 실패하거나 유효하지 않은 이미지는 제외합니다.

    Args:
        items (List[Dict[str, Any]]): Google API 검색 결과 아이템 리스트.

    Returns:
        Tuple[List[Image.Image], List[str]]: 
            - 성공적으로 변환된 PIL 이미지 리스트.
            - 해당 이미지들의 원본 URL 리스트.
    """
    images = []
    valid_urls = []
    
    for item in items:
        url = item.get('link')
        if not url:
            continue
            
        try:
            image_response = requests.get(url, timeout=3)
            image_response.raise_for_status()
            
            img = Image.open(BytesIO(image_response.content)).convert("RGB")
            
            images.append(img)
            valid_urls.append(url)
            
        except Exception:
            continue
            
    return images, valid_urls


def search_and_rerank_image(
    hypernym_text: str, 
    api_config: Dict[str, str],
    clip_components: Tuple[Any, Any]
) -> Optional[str]:
    """이미지 검색 후 CLIP 점수를 기반으로 최적의 이미지를 재선별합니다.

    1. Google API로 10개의 후보 이미지를 검색합니다.
    2. 후보 이미지를 다운로드하여 CLIP 모델로 임베딩을 생성합니다.
    3. 텍스트(상위어)와의 코사인 유사도를 계산하여 가장 점수가 높은 이미지를 반환합니다.

    Args:
        hypernym_text (str): 검색할 상위어 텍스트.
        api_config (Dict[str, str]): API 설정 딕셔너리 (GOOGLE_API_KEY, GOOGLE_CX 포함).
        clip_components (Tuple[Any, Any]): 로드된 CLIP 프로세서와 모델 튜플.

    Returns:
        Optional[str]: 가장 적합한 이미지의 URL. 실패 시 None 반환.
    """
    api_key = api_config.get("GOOGLE_API_KEY")
    cx = api_config.get("GOOGLE_CX")
    
    if not api_key or not cx:
        print("API 키/CX가 누락되어 검색을 수행할 수 없습니다.")
        return None

    # '모음', '흰 배경' 키워드를 추가하여 개념적인 이미지 검색 유도
    query = f"{hypernym_text} assortment collection photo white background"
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'cx': cx,
        'key': api_key,
        'searchType': 'image',
        'num': 10,              
        'safe': 'active',
        'imgType': 'photo',      
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'items' not in data or not data['items']:
            print("Google API: 검색 결과가 없습니다.")
            return None

        print(f"    [CLIP Rerank] 상위 10개 이미지 다운로드 및 분석 중...")
        processor, model = clip_components
        candidate_images, candidate_urls = _download_and_process_images(data['items'])
        
        if not candidate_images:
            print("후보 이미지를 다운로드하거나 처리할 수 없습니다.")
            return None

        # CLIP 유사도 계산 및 재랭킹
        text_features = clip_processor.get_text_features([hypernym_text], processor, model)
        image_features = clip_processor.get_image_features(candidate_images, processor, model)
        
        similarity_scores = clip_processor.calculate_similarity(text_features, image_features).flatten()
        
        # 최고 점수 이미지 선정
        best_score_index = torch.argmax(similarity_scores).item()
        best_url = candidate_urls[best_score_index]
        best_score = similarity_scores[best_score_index].item()
        
        print(f"    [CLIP Rerank] 최고 점수: {best_score:.4f} / URL: {best_url}")
        
        return best_url

    except requests.exceptions.RequestException as e:
        print(f"Google API 요청 실패: {e}")
        return None
    except Exception as e:
        print(f"CLIP 재랭킹 중 오류 발생: {e}")
        return None