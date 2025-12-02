"""
Hypernym Visualizer API

이 모듈은 FastAPI를 사용하여 멀티모달(이미지+텍스트) 입력으로부터 
공통된 상위어(Hypernym)를 추출하고, 이를 시각화(검색 또는 생성)하는 API를 제공합니다.

주요 기능:
    - 정적 파일 서빙 (프론트엔드 및 생성된 이미지)
    - CORS 설정 지원
    - CLIP 모델 기반 상위어 추출
    - 다양한 모드(재순위화 검색, 기본 검색, 생성형 AI)를 통한 시각화
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

from src.core.clip_processor import load_clip_model
from src.core.hypernym_extractor import determine_best_hypernym
from src.external.search_api_client import search_image
from src.external.search_with_clip import search_and_rerank_image
from src.external.sd_generator import generate_image_from_text

load_dotenv()

API_CONFIG = {
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    "GOOGLE_CX": os.getenv("GOOGLE_CX"),
}

app = FastAPI(title="Hypernym Visualizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GENERATED_IMAGES_DIR = "generated_images"
os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)
app.mount("/generated_images", StaticFiles(directory=GENERATED_IMAGES_DIR), name="generated_images")

if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..")
    
    index_path = os.path.join(project_root, "templates", "index.html")
    
    if not os.path.exists(index_path):
        return {"error": f"File not found at {index_path}. Please check your folder structure."}
        
    return FileResponse(index_path)

@app.on_event("startup")
async def startup_event():
    """
    애플리케이션 시작 시 실행되는 이벤트 핸들러입니다.

    서버 시작 시 CLIP 모델을 메모리에 로드하여 `CLIP_COMPONENTS` 전역 변수에 할당합니다.
    모델 로딩에 실패할 경우 에러 메시지를 출력합니다.
    """
    global CLIP_COMPONENTS
    try:
        CLIP_COMPONENTS = load_clip_model()
        print("API 서버 시작 준비 완료.")
    except Exception as e:
        print(f"초기화 오류: {e}")

@app.post("/extract_and_visualize/")
async def extract_and_visualize(
    files: List[UploadFile] = File(...),
    texts: List[str] = Form(...),
    visualization_mode: str = Query("CLIP_RERANK") 
) -> Dict[str, Any]:
    """
    업로드된 이미지와 텍스트 쌍을 분석하여 최적의 상위어를 추출하고 시각화 결과를 반환합니다.

    사용자로부터 이미지 파일과 텍스트 설명을 입력받아, 내부 로직(Model 1)을 통해 
    공통된 상위어를 도출합니다. 그 후 선택된 시각화 모드(Model 2)에 따라 
    관련 이미지를 검색하거나 생성하여 URL을 제공합니다.

    Args:
        files (List[UploadFile]): 분석할 이미지 파일 리스트. (form-data)
        texts (List[str]): 각 이미지에 대응하는 텍스트 설명 리스트. (form-data)
        visualization_mode (str, optional): 결과 이미지를 도출하는 방식. 
            기본값은 "CLIP_RERANK"입니다.
            - "CLIP_RERANK": 구글 이미지 검색 후 CLIP으로 가장 적합한 이미지 재선별.
            - "BASIC_SEARCH": 단순 구글 이미지 검색 결과 사용.
            - "GENERATIVE": Stable Diffusion 등을 사용하여 이미지 생성.

    Returns:
        Dict[str, Any]: 다음 키를 포함하는 JSON 응답 딕셔너리.
            - status (str): 처리 성공 여부 ("success" 또는 "fail").
            - hypernym (str): 추출된 최적 상위어.
            - confidence_score (float): 추출된 상위어의 신뢰도 점수 (소수점 4자리).
            - visualization_mode (str): 사용된 시각화 모드.
            - final_image_url (str, optional): 결과 이미지의 접근 가능한 URL.

    Raises:
        HTTPException: 
            - 400 Bad Request: 파일과 텍스트의 개수가 일치하지 않거나, 입력 데이터가 2개 미만인 경우.
    """
    
    # 1. 입력 확인
    if len(files) != len(texts) or len(files) < 2:
        raise HTTPException(status_code=400, detail="입력 개수 오류")
    
    input_images = []
    for file in files:
        c = await file.read()
        input_images.append(Image.open(BytesIO(c)).convert("RGB"))

    # 3. Model 1
    final_hypernym, score = determine_best_hypernym(input_images, texts, CLIP_COMPONENTS)
    if not final_hypernym:
        return {"status": "fail"}

    # 4. Model 2
    final_image_url = None
    mode = visualization_mode.upper()
    
    if mode == "CLIP_RERANK":
        final_image_url = search_and_rerank_image(final_hypernym, API_CONFIG, CLIP_COMPONENTS)
    elif mode == "BASIC_SEARCH":
        final_image_url = search_image(final_hypernym, API_CONFIG)
    elif mode == "GENERATIVE":
        path = generate_image_from_text(final_hypernym, GENERATED_IMAGES_DIR)
        if path:
            final_image_url = f"http://127.0.0.1:8000/{path.replace(os.sep, '/')}"

    return {
        "status": "success",
        "hypernym": final_hypernym,
        "confidence_score": round(score, 4),
        "visualization_mode": mode,
        "final_image_url": final_image_url
    }