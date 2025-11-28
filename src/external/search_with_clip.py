"""
이 모듈은 Google 검색 결과에 CLIP 기반 재랭킹(Reranking)을 적용하여
상위어(Hypernym)와 의미적으로 가장 유사한 이미지를 선별합니다.
"""

import requests
import torch
from typing import Dict, Optional, Any, List, Tuple
from PIL import Image
from io import BytesIO

# CLIP 임베딩 처리를 위해 core 모듈 사용
from src.core import clip_processor

def _download_images(items: List[Dict[str, Any]]) -> Tuple[List[Image.Image], List[str]]:
    """검색 결과 URL들에서 이미지를 다운로드합니다."""
    images = []
    valid_urls = []
    
    print(f"    [CLIP Rerank] 상위 {len(items)}개 이미지 다운로드 및 분석 중...")
    
    for item in items:
        url = item['link']
        try:
            # 타임아웃을 짧게 주어 빠르게 다운로드 시도
            resp = requests.get(url, timeout=3)
            resp.raise_for_status()
            
            # 이미지 변환
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            
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
    """
    1. Google API로 10개 이미지 검색
    2. CLIP으로 텍스트-이미지 유사도 측정
    3. 가장 점수가 높은 이미지 URL 반환
    """
    
    api_key = api_config.get("GOOGLE_API_KEY")
    cx = api_config.get("GOOGLE_CX")
    processor, model = clip_components

    if not hypernym_text or not api_key or not cx:
        return None

    # 1. 쿼리 구성 (일반적인 모음/대표성 강조)
    # 기존 search_api_client와 다르게 좀 더 넓은 범위를 검색해도 CLIP이 걸러줍니다.
    query = f"{hypernym_text} assortment collection photo white background"
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'cx': cx,
        'key': api_key,
        'searchType': 'image',
        'num': 10,               # [중요] 후보군 10개 확보
        'safe': 'active',
        'imgType': 'photo',
    }

    try:
        # 2. API 호출
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if 'items' not in data:
            return None

        # 3. 이미지 다운로드
        candidate_images, candidate_urls = _download_images(data['items'])
        
        if not candidate_images:
            return None

        # 4. CLIP 임베딩 생성 및 점수 계산
        # 텍스트: "edible fruit"
        text_features = clip_processor.get_text_features([hypernym_text], processor, model)
        # 이미지: 다운로드한 10장
        image_features = clip_processor.get_image_features(candidate_images, processor, model)
        
        # 유사도 계산
        scores = clip_processor.calculate_similarity(text_features, image_features).flatten()
        
        # 5. 최고 점수 이미지 선정
        best_idx = torch.argmax(scores).item()
        best_url = candidate_urls[best_idx]
        best_score = scores[best_idx].item()
        
        print(f"    [CLIP Rerank] 최고 점수: {best_score:.4f} / URL: {best_url}")
        
        return best_url

    except Exception as e:
        print(f"CLIP 검색 중 오류: {e}")
        return None