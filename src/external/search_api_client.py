"""
이 모듈은 외부 검색 엔진 API(Google Custom Search)를 호출하여 이미지를 검색합니다.

상위어 텍스트를 검색 쿼리로 변환하고, API를 호출하여 결과 JSON을 파싱한 뒤,
가장 적절한 이미지의 URL을 반환하는 기능을 제공합니다.
"""

import requests
from typing import Dict, Optional, Any

def _optimize_query(hypernym_text: str) -> str:
    """검색 정확도를 높이기 위해 텍스트를 검색 쿼리 형태로 최적화합니다.
    
    Args:
        hypernym_text (str): 모델 1에서 추출된 상위어 (예: "footwear")
        
    Returns:
        str: 최적화된 쿼리 (예: "footwear representative object photo")
    """
    # 텍스트가 비어있으면 그대로 반환
    if not hypernym_text:
        return ""
        
    # 검색어 뒤에 키워드를 붙여서 추상적인 이미지나 사전적 정의 텍스트 이미지가 
    # 나오는 것을 방지하고, 사물 위주의 사진이 나오도록 유도
    keywords = ["variety", "collection", "assortment", "photo", "white background"]
    return f"{hypernym_text} {' '.join(keywords)}"


def search_image(
    hypernym_text: str, 
    api_config: Dict[str, str]
) -> Optional[str]:
    """Google Custom Search API를 사용하여 상위어에 해당하는 이미지를 검색합니다.

    Args:
        hypernym_text (str): 검색할 상위어 텍스트.
        api_config (Dict[str, str]): API 설정 정보가 담긴 딕셔너리.
            필수 키: 'GOOGLE_API_KEY', 'GOOGLE_CX' (Search Engine ID)

    Returns:
        Optional[str]: 검색된 이미지의 URL. 검색 실패 시 None을 반환합니다.
    """
    
    if not hypernym_text:
        print("경고: 검색할 텍스트가 비어 있습니다.")
        return None

    api_key = api_config.get("GOOGLE_API_KEY")
    cx = api_config.get("GOOGLE_CX")

    if not api_key or not cx:
        print("오류: API Key 또는 CX(Custom Search Engine ID)가 설정되지 않았습니다.")
        return None

    # 1. 쿼리 최적화
    query = _optimize_query(hypernym_text)
    print(f"검색 API 호출 중... 쿼리: '{query}'")

    # 2. API 요청 파라미터 설정
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'cx': cx,
        'key': api_key,
        'searchType': 'image',   # 이미지 검색 모드
        'num': 1,                # 1개의 결과만 가져옴
        'safe': 'active',        # 세이프 서치 적용
        'imgType': 'photo',      # 사진 유형의 이미지만 검색 (클립아트 등 제외)
        # 'rights': 'cc_publicdomain' # 필요 시 저작권 필터 추가 가능
    }

    try:
        # 3. API 호출
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status() # 4xx, 5xx 에러 발생 시 예외 처리
        
        data = response.json()

        # 4. 결과 파싱 및 URL 추출
        if 'items' in data and len(data['items']) > 0:
            image_url = data['items'][0]['link']
            print(f"이미지 검색 성공: {image_url}")
            return image_url
        else:
            print("검색 결과가 없습니다.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"API 요청 실패: {e}")
        return None
    except KeyError as e:
        print(f"API 응답 파싱 오류: {e}")
        return None