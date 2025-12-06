"""
이 모듈은 FastAPI를 사용하여 API 엔드포인트를 정의하고,
전체 상위어 추출 및 세 가지 시각화 방식을 통합 실행합니다.

[새로 추가된 기능]
- 최종 출력 이미지에 대한 CLIP Score를 계산하여 반환합니다. (성능 비교 지표)
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple

import os
from dotenv import load_dotenv
import requests
import numpy as np # CLIP Score 계산을 위해 필요

# --- 환경 변수 로드 ---
load_dotenv() 

API_CONFIG = {
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    "GOOGLE_CX": os.getenv("GOOGLE_CX"),
}

# --- 핵심 모듈 임포트 ---
from src.core.clip_processor import load_clip_model, calculate_similarity, get_text_features, get_image_features
from src.core.hypernym_extractor import determine_best_hypernym

# [Model 2 Clients]
from src.external.search_api_client import search_image          
from src.external.search_with_clip import search_and_rerank_image  
from src.external.sd_generator import generate_image_from_text 


# --- [NEW] CLIP Score 계산 유틸리티 함수 ---
def calculate_clip_score_for_url(
    hypernym_text: str, 
    image_url: Optional[str], 
    clip_components: Tuple[Any, Any]
) -> float:
    """주어진 URL의 이미지를 다운로드하여 텍스트와의 CLIP 유사도를 계산합니다."""
    
    if not image_url:
        return 0.0

    processor, model = clip_components
    
    try:
        # 1. 이미지 다운로드 및 PIL 객체로 변환 (로컬/외부 URL 구분)
        if image_url.startswith("http"):
            img_response = requests.get(image_url, timeout=5)
            img_response.raise_for_status()
            image_obj = Image.open(BytesIO(img_response.content)).convert("RGB")
        else:
            # GENERATIVE 모드에서 로컬 경로를 반환했을 경우
            image_obj = Image.open(image_url).convert("RGB")

        # 2. 임베딩 생성 및 유사도 계산
        text_features = get_text_features([hypernym_text], processor, model)
        image_features = get_image_features([image_obj], processor, model)
        
        # 3. 코사인 유사도 계산
        similarity_matrix = calculate_similarity(text_features, image_features)
        
        # 4. 결과 반환 (numpy로 변환하여 float으로 반환)
        score = similarity_matrix.numpy().flatten()[0]
        return float(score)

    except Exception:
        return 0.0


# --- App 및 전역 설정 ---
app = FastAPI(title="Hypernym Visualizer API")
CLIP_COMPONENTS = None 

GENERATED_IMAGES_DIR = "generated_images"
os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)
app.mount("/generated_images", StaticFiles(directory=GENERATED_IMAGES_DIR), name="generated_images")

# [Front-end 경로 설정]
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    global CLIP_COMPONENTS
    try:
        CLIP_COMPONENTS = load_clip_model()
        print("API 서버 시작 준비 완료.")
    except Exception as e:
        print(f"초기화 오류: {e}")

@app.get("/")
async def read_index():
    from fastapi.responses import FileResponse
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..")
    index_path = os.path.join(project_root, "templates", "index.html")
    
    if not os.path.exists(index_path):
        return {"error": f"File not found at {index_path}. Check folder structure."}
        
    return FileResponse(index_path)


@app.post("/extract_and_visualize/")
async def extract_and_visualize(
    files: List[UploadFile] = File(..., description="N개의 이미지 파일"),
    texts: List[str] = Form(..., description="각 이미지에 대응하는 N개의 텍스트"),
    visualization_mode: str = Query("CLIP_RERANK", 
        description="시각화 방식 선택: CLIP_RERANK(추천), BASIC_SEARCH, GENERATIVE 중 선택") 
) -> Dict[str, Any]:
    
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


    # 3. [Model 1] 최적 상위어 추출
    final_hypernym, score = determine_best_hypernym(
        input_images=input_images,
        input_texts=texts,
        clip_model_components=CLIP_COMPONENTS
    )
    
    if final_hypernym is None:
        return JSONResponse(
            status_code=404, 
            content={"status": "fail", "message": "공통 상위어를 찾지 못했습니다.", "hypernym": None}
        )

    # 4. [Model 2] 시각화 방식 선택 및 실행
    
    final_image_url: Optional[str] = None
    mode_used = visualization_mode.upper()

    # API 키 확인 (외부 API 사용 시)
    if mode_used in ["CLIP_RERANK", "BASIC_SEARCH"] and (not API_CONFIG["GOOGLE_API_KEY"] or not API_CONFIG["GOOGLE_CX"]):
        raise HTTPException(
            status_code=503,
            detail="Google API 설정 오류: .env 파일에 키/CX가 로드되지 않았습니다."
        )
    
    
    if mode_used == "CLIP_RERANK":
        final_image_url = search_and_rerank_image(
            hypernym_text=final_hypernym,
            api_config=API_CONFIG,
            clip_components=CLIP_COMPONENTS
        )
    
    elif mode_used == "BASIC_SEARCH":
        final_image_url = search_image(hypernym_text=final_hypernym, api_config=API_CONFIG)

    elif mode_used == "GENERATIVE":
        generated_path = generate_image_from_text(final_hypernym, GENERATED_IMAGES_DIR)
        
        if generated_path:
            server_host = "http://127.0.0.1:8000"
            url_path = generated_path.replace(os.sep, "/")
            final_image_url = f"{server_host}/{url_path}"
        else:
            final_image_url = None

    else:
        raise HTTPException(status_code=400, detail="유효하지 않은 시각화 모드입니다.")
        
        
    # 5. 최종 평가 및 결과 반환
    
    # 최종 이미지 URL이 확정된 후, CLIP Score 계산
    final_clip_score = calculate_clip_score_for_url(final_hypernym, final_image_url, CLIP_COMPONENTS)

    return {
        "status": "success",
        "hypernym": final_hypernym,
        "confidence_score": round(score, 4), # Model 1의 내부 신뢰도 점수 (WordNet/Penalty)
        "visualization_mode": mode_used,
        "final_image_url": final_image_url,
        "final_clip_score": round(final_clip_score, 4), # Model 2의 외부 평가 점수 (실제 성능)
        "message": f"상위어 추출 및 {mode_used} 시각화 완료"
    }