"""NLTK WordNet 기반 상위어 탐색 모듈.

이 모듈은 NLTK WordNet 인터페이스를 사용하여 다수의 텍스트 입력에 대한
공통된 상위어(Hypernym) 관계를 탐색합니다.
특히 2개 이상의 입력이 주어졌을 때, 모든 단어가 공유하는 상위어 집합의 교집합을
계산하여 가장 적합한 공통 조상 후보군을 추출합니다.
"""

import nltk
from nltk.corpus import wordnet as wn
from typing import List, Dict, Set, Any

# 필터링할 추상적인 상위어 목록 (너무 포괄적이거나 시각화 불가능한 개념 제외)
STOP_CONCEPTS = {
    'entity', 'physical_entity', 'abstraction', 'object', 'whole', 
    'artifact', 'unit', 'matter', 'thing', 'being', 'causal_agent',
    'measure', 'psychological_feature', 'attribute', 'group', 'relation'
}

def _ensure_resources_loaded():
    """NLTK WordNet 데이터 리소스의 존재 여부를 확인하고 다운로드합니다.

    `wordnet`과 `omw-1.4` 코퍼스가 로컬에 없는 경우 자동으로 다운로드를 시도합니다.
    """
    try:
        wn.ensure_loaded()
    except LookupError:
        print("NLTK WordNet 데이터를 다운로드합니다...")
        nltk.download('wordnet')
        nltk.download('omw-1.4')


def _calculate_weight(synset) -> float:
    """Synset의 계층적 깊이를 기반으로 가중치를 계산합니다.

    깊이가 깊을수록(구체적일수록) 높은 점수를 부여하되, 지나친 편향을 막기 위해
    완화된 계산식을 사용합니다.

    Args:
        synset: 가중치를 계산할 NLTK Synset 객체.

    Returns:
        float: 계산된 가중치 (소수점 4자리 반올림).
    """
    depth = synset.max_depth()
    
    # 깊이 가산점을 완화하여 일반적인 단어의 선택 기회를 높임
    # [가중치 공식] 기본 점수 0.5에 깊이 1당 0.05점씩 가산합니다.
    weight = 0.5 + (depth * 0.05)
    
    return round(weight, 4)


def find_common_hypernym_candidates(texts: List[str]) -> List[Dict[str, Any]]:
    """모든 입력 텍스트들의 공통 상위어(교집합) 후보군을 추출합니다.

    입력된 모든 단어가 공통적으로 포함된 상위어 경로를 찾기 위해,
    첫 번째 단어의 상위어 집합을 기준으로 나머지 단어들의 상위어 집합과
    순차적인 교집합(Intersection) 연산을 수행합니다.

    Args:
        texts (List[str]): 분석할 입력 텍스트 리스트 (예: ['car', 'bus', 'bicycle']).

    Returns:
        List[Dict[str, Any]]: 공통 상위어 후보 딕셔너리의 리스트.
            각 딕셔너리는 다음 키를 포함하며, 가중치 내림차순으로 정렬됩니다.
            - 'text' (str): 상위어의 표제어.
            - 'weight' (float): 계층 깊이에 따른 가중치.
            - 'synset' (str): Synset 식별자.
    """
    _ensure_resources_loaded()
    
    # 1. 입력이 하나뿐인 경우 해당 단어의 모든 상위어 반환
    if len(texts) == 1:
        synsets = wn.synsets(texts[0].lower().replace(" ", "_"), pos=wn.NOUN)
        common_candidates = {hyp for s in synsets for path in s.hypernym_paths() for hyp in path}
    
    # 2. 입력이 2개 이상인 경우 교집합 탐색
    elif len(texts) >= 2:
        synsets_list = []
        for text in texts:
            clean_text = text.lower().replace(" ", "_")
            synsets = wn.synsets(clean_text, pos=wn.NOUN)
            
            if not synsets:
                print(f"경고: 워드넷에서 단어를 찾을 수 없어 '{text}'는 제외됩니다.")
                continue
            synsets_list.append(synsets)

        if not synsets_list:
            return []

        # 첫 번째 텍스트의 모든 상위어 집합을 초기 기준으로 설정
        initial_synsets = synsets_list[0]
        all_hypernyms = {hyp for s in initial_synsets for path in s.hypernym_paths() for hyp in path}

        # 나머지 텍스트들에 대해 순차적으로 교집합 연산 수행
        for synsets_of_text in synsets_list[1:]:
            current_text_hypernyms = {hyp for s in synsets_of_text for path in s.hypernym_paths() for hyp in path}
            all_hypernyms = all_hypernyms.intersection(current_text_hypernyms)
            
            # 교집합이 공집합이 되면 조기 종료
            if not all_hypernyms:
                break
        
        common_candidates = all_hypernyms

    else:
        # 입력이 없는 경우
        return []

    # 3. 필터링 및 가중치 계산
    results = []
    seen_texts = set()

    for synset in common_candidates:
        lemma_name = synset.lemmas()[0].name().replace('_', ' ')
        
        # 필터링: 추상적 개념 및 중복 제외
        if lemma_name.lower() in STOP_CONCEPTS:
            continue
        if lemma_name in seen_texts:
            continue

        weight = _calculate_weight(synset)
        
        results.append({
            'text': lemma_name,
            'weight': weight,
            'synset': synset.name()
        })
        seen_texts.add(lemma_name)

    # 가중치 높은 순으로 정렬
    results.sort(key=lambda x: x['weight'], reverse=True)
    
    return results