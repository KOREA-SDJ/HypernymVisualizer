"""
이 모듈은 FastAPI를 사용하여 API 엔드포인트를 정의하고,
전체 상위어 추출 및 이미지 시각화 파이프라인을 통합 실행합니다.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
load_dotenv() # .env 파일에서 환경 변수를 로드합니다.

# 설정 파일 로드 (pydantic-settings 또는 dotenv를 사용한다고 가정)
# 실제 구현 시 환경 변수 로딩 코드가 필요합니다.
# 예시:
API_CONFIG = {
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    "GOOGLE_CX": os.getenv("GOOGLE_CX"),
}

# --- 핵심 모듈 임포트 ---
from src.core.clip_processor import load_clip_model
from src.core.hypernym_extractor import determine_best_hypernym
from src.external.search_api_client import search_image

# --- 전역 변수 ---
app = FastAPI(title="Hypernym Visualizer API")
CLIP_COMPONENTS = None # (processor, model) 튜플 저장


@app.on_event("startup")
async def startup_event():
    """
    서버 시작 시 CLIP 모델을 로드하여 GPU/메모리에 올려둡니다. (초기 지연을 줄이기 위함)
    """
    global CLIP_COMPONENTS
    try:
        # 모델을 한 번만 로드하여 전역 변수에 저장
        CLIP_COMPONENTS = load_clip_model()
        print("API 서버 시작 준비 완료.")
    except Exception as e:
        print(f"CLIP 모델 로드 중 심각한 오류 발생: {e}")
        # 모델 로드 실패 시 서버 시작을 중단할 수도 있습니다.
        # raise HTTPException(status_code=500, detail="서버 초기화 실패: CLIP 모델 로드 불가")


@app.post("/extract_and_visualize/")
async def extract_and_visualize(
    files: List[UploadFile] = File(..., description="N개의 이미지 파일"),
    texts: List[str] = Form(..., description="각 이미지에 대응하는 N개의 텍스트")
) -> Dict[str, Any]:
    """
    N개의 이미지와 N개의 텍스트를 입력받아 공통 상위어를 추출하고 해당 이미지를 검색하여 반환합니다.

    Args:
        files (List[UploadFile]): 입력 이미지 파일 리스트.
        texts (List[str]): 각 이미지에 대한 텍스트 설명 리스트.

    Returns:
        Dict[str, Any]: 최종 상위어 텍스트와 검색된 이미지 URL, 또는 오류 메시지.
    """
    
    # 1. 입력 유효성 검사
    if len(files) != len(texts) or len(files) < 2:
        raise HTTPException(
            status_code=400, 
            detail="입력 이미지와 텍스트의 개수는 같아야 하며, 최소 2쌍 이상이어야 합니다."
        )

    # 2. 이미지 데이터 로드
    input_images: List[Image.Image] = []
    try:
        for file in files:
            contents = await file.read()
            img = Image.open(BytesIO(contents)).convert("RGB")
            input_images.append(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 파일 로드 실패: {e}")


    # 3. [모델 1] 최적 상위어 추출
    final_hypernym, score = determine_best_hypernym(
        input_images=input_images,
        input_texts=texts,
        clip_model_components=CLIP_COMPONENTS
    )
    
    if final_hypernym is None:
        return JSONResponse(
            status_code=404, 
            content={"message": "공통 상위어를 찾지 못했습니다.", "hypernym": None}
        )

    # 4. [모델 2] 상위어 기반 이미지 검색
    image_url = search_image(
        hypernym_text=final_hypernym, 
        api_config=API_CONFIG
    )

    # 5. 최종 결과 반환
    if image_url:
        return {
            "status": "success",
            "hypernym": final_hypernym,
            "confidence_score": round(score, 4),
            "visualized_image_url": image_url,
            "message": "상위어 추출 및 시각화 성공"
        }
    else:
        return {
            "status": "success_but_no_image",
            "hypernym": final_hypernym,
            "confidence_score": round(score, 4),
            "visualized_image_url": None,
            "message": "상위어는 추출했으나, 검색 API에서 이미지를 찾지 못했습니다."
        }