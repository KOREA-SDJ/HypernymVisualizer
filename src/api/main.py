"""
이 모듈은 FastAPI를 사용하여 API 엔드포인트를 정의하고,
전체 상위어 추출 및 시각화 파이프라인(검색, 재랭킹, 생성)을 통합 실행합니다.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles  # [NEW] 정적 파일 서빙용
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any, Optional
import os

# --- 환경 변수 로드 ---
from dotenv import load_dotenv
load_dotenv() 

# 환경 변수에서 키를 가져와 API_CONFIG를 구성합니다.
API_CONFIG = {
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    "GOOGLE_CX": os.getenv("GOOGLE_CX"),
}

# --- 핵심 모듈 임포트 ---
from src.core.clip_processor import load_clip_model
from src.core.hypernym_extractor import determine_best_hypernym

# [Model 2 Clients]
from src.external.search_api_client import search_image          # 1. BASIC_SEARCH
from src.external.search_with_clip import search_and_rerank_image  # 2. CLIP_RERANK
from src.external.sd_generator import generate_image_from_text     # 3. GENERATIVE (생성)


# --- 전역 변수 및 설정 ---
app = FastAPI(title="Hypernym Visualizer API")
CLIP_COMPONENTS = None # (processor, model) 튜플 저장

# [NEW] 생성된 이미지가 저장될 폴더 설정 및 정적 마운트
# 이렇게 하면 'http://localhost:8000/generated_images/파일명.png'로 접근 가능합니다.
GENERATED_IMAGES_DIR = "generated_images"
os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)
app.mount("/generated_images", StaticFiles(directory=GENERATED_IMAGES_DIR), name="generated_images")


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 CLIP 모델을 로드합니다."""
    global CLIP_COMPONENTS
    print("서버 초기화 시작...")
    try:
        # CLIP 모델 로드 (가장 중요)
        CLIP_COMPONENTS = load_clip_model()
        print("API 서버 시작 준비 완료.")
        
        # 참고: Stable Diffusion 모델은 startup에 로드하면 초기 구동이 너무 느려질 수 있어,
        # 첫 요청 시 로드되도록 sd_generator 내부에서 처리합니다.
    except Exception as e:
        print(f"초기화 오류: {e}")


@app.post("/extract_and_visualize/")
async def extract_and_visualize(
    files: List[UploadFile] = File(..., description="N개의 이미지 파일"),
    texts: List[str] = Form(..., description="각 이미지에 대응하는 N개의 텍스트"),
    visualization_mode: str = Query("CLIP_RERANK", 
        description="시각화 방식 선택: CLIP_RERANK(추천), BASIC_SEARCH, GENERATIVE(생성형) 중 선택") 
) -> Dict[str, Any]:
    """
    공통 상위어를 추출하고, 선택된 방식(검색/재랭킹/생성)에 따라 이미지를 시각화하여 반환합니다.
    """
    
    # 1. 입력 유효성 검사 (N >= 2)
    if len(files) != len(texts) or len(files) < 2:
        raise HTTPException(
            status_code=400, 
            detail="입력 이미지와 텍스트의 개수는 같아야 하며, 최소 2쌍 이상이어야 합니다."
        )

    # 2. 이미지 데이터 로드 및 전처리
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

    # --- A. CLIP 재랭킹 검색 (제안 모델) ---
    if mode_used == "CLIP_RERANK":
        if not API_CONFIG["GOOGLE_API_KEY"] or not API_CONFIG["GOOGLE_CX"]:
             raise HTTPException(status_code=503, detail="Google API 키 설정이 필요합니다.")
             
        final_image_url = search_and_rerank_image(
            hypernym_text=final_hypernym,
            api_config=API_CONFIG,
            clip_components=CLIP_COMPONENTS
        )
    
    # --- B. 기본 Google 검색 (비교군) ---
    elif mode_used == "BASIC_SEARCH":
        if not API_CONFIG["GOOGLE_API_KEY"] or not API_CONFIG["GOOGLE_CX"]:
             raise HTTPException(status_code=503, detail="Google API 키 설정이 필요합니다.")
             
        final_image_url = search_image(hypernym_text=final_hypernym, api_config=API_CONFIG)

    # --- C. 생성형 모델 (Stable Diffusion) ---
    elif mode_used == "GENERATIVE":
        # 로컬 파일 경로 반환 (예: "generated_images/fruit.png")
        generated_path = generate_image_from_text(final_hypernym, GENERATED_IMAGES_DIR)
        
        if generated_path:
            # 로컬 경로를 URL로 변환 (Windows 역슬래시 호환)
            url_path = generated_path.replace(os.sep, "/")
            # 현재 로컬 서버 주소 가정 (배포 시 변경 필요)
            final_image_url = f"http://127.0.0.1:8000/{url_path}"
        else:
            final_image_url = None

    else:
        raise HTTPException(status_code=400, detail="유효하지 않은 시각화 모드입니다. (CLIP_RERANK, BASIC_SEARCH, GENERATIVE 중 선택)")
        
        
    # 5. 최종 결과 반환
    return {
        "status": "success",
        "hypernym": final_hypernym,
        "confidence_score": round(score, 4),
        "visualization_mode": mode_used,
        "final_image_url": final_image_url,
        "message": f"상위어 추출 및 {mode_used} 시각화 완료"
    }