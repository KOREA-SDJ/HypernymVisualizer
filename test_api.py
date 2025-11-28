import requests
import json
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any

# .env 파일 로드 (API 키/CX ID가 스크립트 실행 시 로드되도록 함)
load_dotenv() 

# --- 환경 설정 ---
API_URL = "http://127.0.0.1:8000/extract_and_visualize/"
MODES_TO_TEST = ["BASIC_SEARCH", "CLIP_RERANK", "GENERATIVE"]
# 테스트할 입력 데이터 (파일이 없다면 create_dummy_image 함수를 활성화하세요)
TEST_IMAGE_PATHS = ['data/banana.png', 'data/apple.png']
TEST_TEXTS = ['banana', 'apple']


def create_dummy_image(filename):
    """테스트용 더미 이미지 파일을 생성합니다 (파일이 없을 경우)."""
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    if not os.path.exists(filename):
        img = Image.new('RGB', (50, 50), color='white')
        img.save(filename)
        print(f"경고: 파일이 없어 {filename} 더미 이미지 생성됨.")


def run_test_for_mode(mode: str):
    """특정 시각화 모드를 사용하여 API를 호출하고 결과를 출력합니다."""
    
    print(f"\n--- 🚀 Model 2-C: {mode} 모드 테스트 시작 ---")
    
    # 1. 파일 핸들링 및 데이터 포맷팅
    files_data = []
    
    # 더미 이미지 생성 확인 (실제 이미지 파일 경로로 변경해야 함)
    for path in TEST_IMAGE_PATHS:
        # data 폴더가 없다면 생성 (test_api.py가 root에 있으므로)
        os.makedirs('data', exist_ok=True) 
        create_dummy_image(path) 
        
        # 파일 핸들러 (API 호출 시 열어서 전송)
        files_data.append(('files', (os.path.basename(path), open(path, 'rb'), 'image/png')))

    # 'texts' 필드를 폼 데이터로 변환 (각 항목은 별도의 튜플로 전달)
    texts_data = [('texts', t) for t in TEST_TEXTS]
    
    # 2. API 호출 (파라미터로 모드 전달)
    try:
        response = requests.post(
            API_URL, 
            files=files_data, 
            data=texts_data,
            params={'visualization_mode': mode} # <--- 모드 전달
        )
        
        # 3. 파일 핸들 닫기
        for _, file_tuple in files_data:
            file_tuple[1].close()

        # 4. 결과 출력
        response.raise_for_status() # HTTP 오류 발생 시 예외 발생
        result_json = response.json()
        
        print(f"✅ 상태 코드: 200 OK")
        print(f"   Hypernym: {result_json.get('hypernym')}")
        print(f"   Image URL: {result_json.get('final_image_url')}")
        print(f"   Mode Used: {result_json.get('visualization_mode')}")
        
        return result_json

    except requests.exceptions.RequestException as e:
        print(f"❌ 요청 실패 또는 서버 오류 발생 (Code: {response.status_code if 'response' in locals() else 'N/A'})")
        print(f"   오류 상세: {e}")
        return None


if __name__ == "__main__":
    print("--- 모든 시각화 모델 비교 테스트 시작 ---")
    
    # 테스트 데이터가 저장될 data 폴더 생성
    os.makedirs('data', exist_ok=True)
    
    # 모든 모드를 순회하며 테스트 실행
    for mode in MODES_TO_TEST:
        run_test_for_mode(mode)

    print("\n--- 모든 테스트 완료 ---")