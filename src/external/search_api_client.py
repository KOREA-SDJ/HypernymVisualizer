"""Google Custom Search API 기반 이미지 검색 모듈.

이 모듈은 외부 검색 엔진 API(Google Custom Search)를 호출하여 이미지를 검색합니다.
상위어 텍스트를 검색 쿼리로 변환하고, API를 호출하여 결과 JSON을 파싱한 뒤,
가장 적절한 이미지의 URL을 반환하는 기능을 제공합니다.
"""

import requests
from typing import Dict, Optional, Any


def _optimize_query(hypernym_text: str) -> str:
    """검색 정확도를 높이기 위해 텍스트를 검색 쿼리 형태로 최적화합니다.

    검색어 뒤에 'variety', 'collection' 등의 키워드를 추가하여
    추상적인 이미지나 사전적 정의가 아닌, 사물 위주의 사진이 검색되도록 유도합니다.

    Args:
        hypernym_text (str): 모델 1에서 추출된 상위어 (예: "footwear").

    Returns:
        str: 최적화된 쿼리 문자열 (예: "footwear representative object photo").
    """
    if not hypernym_text:
        return ""

    keywords = ["variety", "collection", "assortment", "photo", "white background"]
    return f"{hypernym_text} {' '.join(keywords)}"


def search_image(
    hypernym_text: str,
    api_config: Dict[str, str]
) -> Optional[str]:
    """Google Custom Search API를 사용하여 상위어에 해당하는 이미지를 검색합니다.

    입력된 상위어 텍스트를 최적화된 쿼리로 변환한 후, Google API를 호출하여
    가장 관련성이 높은 첫 번째 이미지의 URL을 반환합니다.

    Args:
        hypernym_text (str): 검색할 상위어 텍스트.
        api_config (Dict[str, str]): API 설정 정보가 담긴 딕셔너리.
            필수 키: 'GOOGLE_API_KEY', 'GOOGLE_CX'.

    Returns:
        Optional[str]: 검색된 이미지의 URL. 검색 실패 시 None.
    """
    if not hypernym_text:
        print("경고: 검색할 텍스트가 비어 있습니다.")
        return None

    api_key = api_config.get("GOOGLE_API_KEY")
    cx = api_config.get("GOOGLE_CX")

    if not api_key or not cx:
        print("오류: API Key 또는 CX(Custom Search Engine ID)가 설정되지 않았습니다.")
        return None

    query = _optimize_query(hypernym_text)
    print(f"검색 API 호출 중... 쿼리: '{query}'")

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'cx': cx,
        'key': api_key,
        'searchType': 'image',
        'num': 1,
        'safe': 'active',
        'imgType': 'photo',
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

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