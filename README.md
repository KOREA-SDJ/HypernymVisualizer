{\rtf1}# Hypernym Visualizer

이미지와 텍스트 쌍을 입력받아 공통 상위어(Hypernym)를 추출하고,
이를 시각화(Google 검색 및 Stable Diffusion 생성)하는 멀티모달 AI 프로젝트입니다.

## 기능
1. **Model 1:** CLIP + WordNet을 이용한 최적 상위어 추출
2. **Model 2:**
    - **BASIC_SEARCH:** Google Custom Search API 기반 이미지 검색
    - **CLIP_RERANK:** 검색 결과 중 CLIP 유사도가 가장 높은 이미지 선별 (추천)
    - **GENERATIVE:** Stable Diffusion을 이용한 이미지 생성

## 실행 방법
1. 필수 라이브러리 설치: `pip install -r requirements.txt`
2. `.env` 파일 설정 (Google API Key 등)
3. 서버 실행: `uvicorn src.api.main:app --reload`
4. 테스트 실행: `python test_api.py`