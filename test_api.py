import requests
import os
from PIL import Image
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv() 

# --- 환경 설정 ---
API_URL = "http://127.0.0.1:8000/extract_and_visualize/"

# [수정] 3가지 모드 모두 테스트 (비교용)
MODES_TO_TEST = ["BASIC_SEARCH", "CLIP_RERANK", "GENERATIVE"] 

# --- 🧪 테스트 시나리오 정의 ---
TEST_SCENARIOS = [
    {
        "name": "🍌 과일 5종 테스트",
        "data": [
            {"path": "data/apple.png", "text": "apple"},
            {"path": "data/banana.png", "text": "banana"},
            {"path": "data/grape.png", "text": "grape"},
            {"path": "data/orange.png", "text": "orange"},
            {"path": "data/peach.png", "text": "peach"},
        ]
    },
    {
        "name": "🚗 운송 수단 5종 테스트",
        "data": [
            {"path": "data/car.png", "text": "passenger car"},
            {"path": "data/bus.png", "text": "city bus"},
            {"path": "data/bicycle.png", "text": "bicycle"},
            {"path": "data/train.png", "text": "train"},
            {"path": "data/airplane.png", "text": "airplane"},
        ]
    }
]

def create_dummy_image_if_missing(filename):
    """파일이 없을 경우 테스트를 위해 더미 이미지를 생성합니다."""
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.exists(filename):
        print(f"⚠️ 경고: '{filename}' 파일이 없어 더미 이미지를 생성합니다. (결과 정확도 하락 가능)")
        img = Image.new('RGB', (100, 100), color='skyblue')
        img.save(filename)

def run_scenario(scenario):
    """하나의 시나리오(이미지 묶음)에 대해 API를 호출합니다."""
    print(f"\n\n============================================")
    print(f"📢 시나리오 실행: {scenario['name']}")
    print(f"============================================")

    # 1. 데이터 준비
    files_data = []
    texts_data = []

    for item in scenario['data']:
        path = item['path']
        text = item['text']
        
        # 파일 확인 및 준비
        create_dummy_image_if_missing(path)
        
        # 파일 핸들러 열기 ('files' 키 사용)
        files_data.append(('files', (os.path.basename(path), open(path, 'rb'), 'image/png')))
        # 텍스트 데이터 준비 ('texts' 키 사용)
        texts_data.append(('texts', text))

    # 2. 각 모드별로 호출
    for mode in MODES_TO_TEST:
        print(f"\n--- [Mode: {mode}] 요청 중... ---")
        try:
            # 파일 포인터를 처음으로 되돌림 (재사용 위해)
            for _, (_, f, _) in files_data:
                f.seek(0)

            response = requests.post(
                API_URL, 
                files=files_data, 
                data=texts_data,
                params={'visualization_mode': mode},
                timeout=180 # 생성형 모델 대기 시간 고려 (넉넉하게 3분)
            )
            
            response.raise_for_status()
            result = response.json()

            print(f"✅ 성공!")
            print(f"   - 추출된 상위어: {result.get('hypernym')}")
            print(f"   - 신뢰도 점수: {result.get('confidence_score')}")
            print(f"   - 결과 이미지 URL: {result.get('final_image_url')}")

        except Exception as e:
            print(f"❌ 실패: {e}")
            if 'response' in locals():
                print(f"   서버 응답: {response.text}")

    # 3. 파일 닫기
    for _, (_, f, _) in files_data:
        f.close()

if __name__ == "__main__":
    # data 폴더 생성
    os.makedirs('data', exist_ok=True)
    
    # 모든 시나리오 실행
    for scenario in TEST_SCENARIOS:
        run_scenario(scenario)