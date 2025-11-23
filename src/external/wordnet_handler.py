"""
이 모듈은 NLTK WordNet 인터페이스를 사용하여 텍스트 간의 상위어 관계를 탐색합니다.

입력된 단어들의 Synset(동의어 집합)을 찾고, 그들 간의 공통 상위어(Lowest Common Hypernym)를
추출하며, 너무 추상적인 개념(예: Entity, Object)을 필터링하는 기능을 제공합니다.
"""

import nltk
from nltk.corpus import wordnet as wn
from typing import List, Dict, Set, Any

# 필터링할 추상적인 상위어 목록 (너무 포괄적인 개념 제외)
STOP_CONCEPTS = {
    'entity', 'physical_entity', 'abstraction', 'object', 'whole', 
    'artifact', 'unit', 'matter', 'thing', 'being'
}

def _ensure_resources_loaded():
    """NLTK WordNet 데이터가 있는지 확인하고 없으면 다운로드합니다."""
    try:
        wn.ensure_loaded()
    except LookupError:
        print("NLTK WordNet 데이터를 다운로드합니다...")
        nltk.download('wordnet')
        nltk.download('omw-1.4')


def _calculate_weight(synset) -> float:
    """Synset의 구체성(깊이)에 따라 가중치를 계산합니다.
    
    워드넷 계층 구조에서 더 깊이 있을수록(더 구체적일수록) 높은 가중치를 부여합니다.
    """
    # max_depth(): 루트(Entity)에서 해당 노드까지의 최대 깊이
    depth = synset.max_depth()
    
    # 깊이가 0이면(루트) 가중치 0.1, 깊을수록 1.0에 가까워짐 (단순 휴리스틱)
    # 예: depth 8 -> weight 0.8
    weight = min(depth * 0.1, 2.0) 
    
    # 너무 낮은 가중치는 보정
    return max(weight, 0.1)


def find_common_hypernym_candidates(texts: List[str]) -> List[Dict[str, Any]]:
    """입력된 텍스트들의 공통 상위어 후보군과 가중치를 반환합니다.

    Args:
        texts (List[str]): 입력 텍스트 리스트 (예: ['sneaker', 'boot'])

    Returns:
        List[Dict[str, Any]]: 후보군 리스트.
            예: [{'text': 'footwear', 'weight': 0.8}, {'text': 'covering', 'weight': 0.5}]
    """
    _ensure_resources_loaded()
    
    if not texts:
        return []

    # 1. 각 입력 단어의 Synset 찾기
    synsets_list = []
    for text in texts:
        # 텍스트를 소문자로 변환하고 '_'로 공백 대체 (WordNet 포맷)
        clean_text = text.lower().replace(" ", "_")
        synsets = wn.synsets(clean_text, pos=wn.NOUN) # 명사만 검색
        
        if not synsets:
            print(f"경고: 워드넷에서 단어를 찾을 수 없습니다: '{text}'")
            continue
        synsets_list.append(synsets)

    if len(synsets_list) < 2:
        # 비교할 대상이 없으면 해당 단어의 상위어를 바로 반환 (단일 입력 시나리오 등)
        if len(synsets_list) == 1:
            common_hypernyms = {hyp for s in synsets_list[0] for hyp in s.hypernyms()}
        else:
            return []
    else:
        # 2. 공통 상위어 탐색 (Pairwise Intersection)
        # 첫 번째 단어의 Synset들과 두 번째 단어의 Synset들 간의 LCH(최저 공통 상위어)를 찾음
        # 간단한 구현을 위해 첫 두 단어 기준으로 탐색 (확장 가능)
        synsets_a = synsets_list[0]
        synsets_b = synsets_list[1]
        
        common_candidates = set()
        
        for sa in synsets_a:
            for sb in synsets_b:
                # 두 Synset의 최저 공통 상위어 리스트 반환
                lchs = sa.lowest_common_hypernyms(sb)
                for lch in lchs:
                    common_candidates.add(lch)
                    # LCH의 상위어들도 후보에 포함 (경로 추적)
                    for ancestor in lch.hypernym_paths()[0]:
                        common_candidates.add(ancestor)

    # 3. 필터링 및 가중치 계산
    results = []
    seen_texts = set()

    # common_candidates는 Set이므로 순서가 없음 -> 리스트로 변환
    if 'common_candidates' not in locals(): # 단일 단어 처리 등 예외 케이스
         common_candidates = set()

    for synset in common_candidates:
        # Lemma 이름 추출 (예: Synset('dog.n.01') -> 'dog')
        lemma_name = synset.lemmas()[0].name().replace('_', ' ')
        
        # 필터링: 추상적인 개념 제외
        if lemma_name.lower() in STOP_CONCEPTS:
            continue
            
        # 중복 텍스트 제외
        if lemma_name in seen_texts:
            continue

        weight = _calculate_weight(synset)
        
        results.append({
            'text': lemma_name,
            'weight': weight,
            'synset': synset.name() # 디버깅용
        })
        seen_texts.add(lemma_name)

    # 가중치 높은 순으로 정렬
    results.sort(key=lambda x: x['weight'], reverse=True)
    
    return results