import requests
import os
import shutil
import torch
from PIL import Image
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import urlencode
from transformers import CLIPProcessor, CLIPModel

load_dotenv() 

# --- 설정 상수는 전역 변수로 관리 ---
API_URL = "http://127.0.0.1:8000/extract_and_visualize/"
OUTPUT_DIR = "test_outputs"
MODES_TO_TEST = ["BASIC_SEARCH", "CLIP_RERANK", "GENERATIVE"] 

# CLIP 평가 모델 싱글톤 인스턴스
_EVAL_MODEL = None
_EVAL_PROCESSOR = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_eval_model() -> Tuple[Optional[CLIPModel], Optional[CLIPProcessor]]:
    """평가용 CLIP 모델과 프로세서를 로드하거나 기존 인스턴스를 반환합니다.

    싱글톤 패턴을 사용하여 모델이 중복 로드되는 것을 방지합니다.

    Returns:
        Tuple[Optional[CLIPModel], Optional[CLIPProcessor]]: 
            로드된 모델과 프로세서 튜플. 로드 실패 시 (None, None) 반환.
    """
    global _EVAL_MODEL, _EVAL_PROCESSOR
    
    if _EVAL_MODEL is None:
        print(f"\n[Evaluation] 평가용 CLIP 모델 로드 중... (Device: {_DEVICE})")
        try:
            model_id = "openai/clip-vit-base-patch32"
            _EVAL_MODEL = CLIPModel.from_pretrained(model_id).to(_DEVICE)
            _EVAL_PROCESSOR = CLIPProcessor.from_pretrained(model_id)
        except Exception as e:
            print(f" 평가 모델 로드 실패: {e}")
            return None, None
            
    return _EVAL_MODEL, _EVAL_PROCESSOR


def calculate_similarity(image_path: str, text: str) -> float:
    """이미지와 텍스트 사이의 코사인 유사도(Cosine Similarity)를 계산합니다.

    로컬에 저장된 이미지 파일과 비교할 텍스트(상위어)를 CLIP 모델에 입력하여
    임베딩 벡터 간의 유사도를 산출합니다.

    Args:
        image_path (str): 분석할 이미지 파일의 로컬 경로.
        text (str): 이미지와 비교할 텍스트 (예: 상위어).

    Returns:
        float: 계산된 코사인 유사도 점수. (오류 발생 시 0.0 반환)
    """
    model, processor = get_eval_model()
    if not model or not processor:
        return 0.0

    try:
        image = Image.open(image_path).convert("RGB")
        
        inputs = processor(
            text=[text], 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(_DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 임베딩 정규화
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        
        # 내적(Dot Product)을 통한 유사도 계산
        score = torch.matmul(text_embeds, image_embeds.t()).item()
        return score

    except Exception as e:
        print(f"    점수 계산 중 에러: {e}")
        return 0.0


def create_dummy_image_if_missing(filename: str) -> None:
    """테스트 파일이 없을 경우 빈 더미 이미지를 생성합니다.

    Args:
        filename (str): 생성할 이미지 파일의 경로.
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
    if not os.path.exists(filename):
        print(f" 경고: '{filename}' 파일이 없어 더미 이미지를 생성합니다.")
        img = Image.new('RGB', (100, 100), color='skyblue')
        img.save(filename)


def save_result_image(image_url_or_path: str, mode: str, scenario_name: str) -> Optional[str]:
    """API 결과 이미지를 로컬 디렉토리에 저장합니다.

    URL인 경우 다운로드를 수행하고, 로컬 경로인 경우 파일을 복사합니다.

    Args:
        image_url_or_path (str): 이미지의 URL 또는 로컬 파일 경로.
        mode (str): 현재 테스트 중인 시각화 모드 (예: BASIC_SEARCH).
        scenario_name (str): 테스트 시나리오 이름.

    Returns:
        Optional[str]: 저장된 파일의 로컬 경로. 저장 실패 시 None.
    """
    if not image_url_or_path:
        print("    결과 이미지 경로가 비어 있습니다.")
        return None

    # 파일명 안전하게 변환 (공백 제거 등)
    safe_name = scenario_name.split()[0]
    save_name = f"{safe_name}_{mode}_result.png"
    save_path = os.path.join(OUTPUT_DIR, save_name)

    try:
        # Case 1: URL 다운로드
        if image_url_or_path.startswith("http"):
            print(f"    이미지 다운로드 시도: {image_url_or_path}")
            img_response = requests.get(image_url_or_path, stream=True, timeout=30)
            
            if img_response.status_code == 200:
                with open(save_path, 'wb') as out_file:
                    shutil.copyfileobj(img_response.raw, out_file)
                print(f"    이미지 저장 완료: {save_path}")
                return save_path
            else:
                print(f"    이미지 다운로드 실패 (Status: {img_response.status_code})")
                return None
        
        # Case 2: 로컬 파일 복사
        elif os.path.exists(image_url_or_path):
            shutil.copy(image_url_or_path, save_path)
            print(f"    로컬 파일 복사 완료: {save_path}")
            return save_path
            
        else:
            print(f"    경로 접근 불가: {image_url_or_path}")
            return None

    except Exception as e:
        print(f"    이미지 저장 중 에러: {e}")
        return None


def run_scenario(scenario: Dict[str, Any]) -> None:
    """단일 테스트 시나리오를 실행하고 모드별 결과를 평가합니다.

    Args:
        scenario (Dict[str, Any]): 테스트 시나리오 정보 (이름, 데이터 리스트 포함).
    """
    print(f"\n\n============================================")
    print(f"📢 시나리오 실행: {scenario['name']}")
    print(f"============================================")

    files_data = []
    texts_data = []

    # 데이터 준비
    for item in scenario['data']:
        path = item['path']
        text = item['text']
        create_dummy_image_if_missing(path)
        files_data.append(('files', (os.path.basename(path), open(path, 'rb'), 'image/png')))
        texts_data.append(('texts', text))

    # 모드별 테스트 실행
    for mode in MODES_TO_TEST:
        print(f"\n--- [Mode: {mode}] 요청 중... ---")

        mode_params = {'visualization_mode': mode}
        current_url = f"{API_URL}?{urlencode(mode_params)}"
        print(f"    [DEBUG] URL: {current_url}")

        try:
            # 파일 포인터 초기화 (재사용)
            for _, (_, f, _) in files_data:
                f.seek(0)

            # API 호출 (생성형 모델 고려 Timeout 설정)
            response = requests.post(
                API_URL, 
                files=files_data, 
                data=texts_data,
                params={'visualization_mode': mode},
                timeout=300 
            )
            response.raise_for_status() 
            result = response.json() 

            hypernym = result.get('hypernym', 'unknown')
            final_image_url = result.get('final_image_url')

            print(f"✅ 성공! (Mode: {mode})")
            print(f"   - 추출된 상위어: {hypernym}")
            print(f"   - 결과 URL: {final_image_url}")

            # 1. 결과 이미지 저장
            saved_path = save_result_image(final_image_url, mode, scenario['name'])
            
            # 2. 유사도(CLIP Score) 평가
            if saved_path:
                score = calculate_similarity(saved_path, hypernym)
                print(f"    [유사도 평가] '{hypernym}' vs 결과 이미지")
                print(f"       Score: {score:.4f}")

        except Exception as e:
            print(f" 에러 발생: {e}")

    # 리소스 정리
    for _, (_, f, _) in files_data:
        f.close()


if __name__ == "__main__":
    # 필수 디렉토리 생성
    os.makedirs('data', exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 테스트 데이터셋 정의
    test_scenarios = [
        {
            "name": " 과일 5종 테스트",
            "data": [
                {"path": "data/apple.png", "text": "apple"},
                {"path": "data/banana.png", "text": "banana"},
                {"path": "data/grape.png", "text": "grape"},
                {"path": "data/orange.png", "text": "orange"},
                {"path": "data/peach.png", "text": "peach"},
            ]
        }
    ]
    
    for scenario in test_scenarios:
        run_scenario(scenario)