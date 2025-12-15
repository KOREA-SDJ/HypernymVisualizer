import os
import sys
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# 프로젝트 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 모듈 임포트
from src.core import clip_processor, hypernym_extractor
from src.external import search_api_client, search_with_clip, sd_generator

# 환경 변수 로드
load_dotenv()
API_CONFIG = {
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    "GOOGLE_CX": os.getenv("GOOGLE_CX"),
}
GENERATED_IMAGES_DIR = "generated_images"
os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)

# ==========================================
# ▶ 실험 데이터: 과일(Group 1) + 동물(Group 2) 혼합
# ==========================================
cross_group_inputs = [
    {"text": "apple", "image": "./data/apple.png"},
    {"text": "banana", "image": "./data/banana.png"},
    {"text": "dog", "image": "./data/dog.png"},
    {"text": "cat", "image": "./data/cat.png"}
]

def load_image_content(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def calculate_score(text, image_obj, processor, model):
    if image_obj is None: return 0.0
    try:
        text_feat = clip_processor.get_text_features([text], processor, model)
        img_feat = clip_processor.get_image_features([image_obj], processor, model)
        return round((text_feat @ img_feat.T).item(), 4)
    except:
        return 0.0

def visualize_and_save(mode, hypernym, clip_components):
    save_path = None
    try:
        if mode == "Basic Search":
            url = search_api_client.search_image(hypernym, API_CONFIG)
            return url # URL 반환
        
        elif mode == "CLIP Reranking":
            url = search_with_clip.search_and_rerank_image(hypernym, API_CONFIG, clip_components)
            return url # URL 반환
            
        elif mode == "Generative":
            # 한계점 실험용이므로 파일명을 명확히 구분 (cross_generated.png)
            path = sd_generator.generate_image_from_text(hypernym, GENERATED_IMAGES_DIR)
            if path:
                # 파일명 변경 (덮어쓰기 방지 아님, 그냥 특정 이름으로 저장)
                new_path = os.path.join(GENERATED_IMAGES_DIR, "cross_group_generated.png")
                import shutil
                shutil.move(path, new_path)
                return new_path
    except Exception as e:
        print(f"Visualization Error ({mode}): {e}")
    return None

def run_cross_group_experiment():
    print(">>> ⚠️ [Cross-Group Limit Test] 이종 도메인 결합 실험 시작")
    print(">>> Inputs: Apple, Banana (Fruit) + Dog, Cat (Animal)")
    
    processor, model = clip_processor.load_clip_model()
    clip_components = (processor, model)

    # 1. 데이터 로드
    imgs = [load_image_content(i['image']) for i in cross_group_inputs]
    texts = [i['text'] for i in cross_group_inputs]
    
    if any(i is None for i in imgs):
        print("이미지 로드 실패. 경로를 확인하세요.")
        return

    # 2. 상위어 추론 (예상: organism, living thing, whole 등 아주 추상적인 단어)
    print("\n[1] 상위어 추론 중...")
    hypernym, score = hypernym_extractor.determine_best_hypernym(
        imgs, texts, clip_components
    )
    print(f"   👉 도출된 상위어: '{hypernym}' (Score: {score:.4f})")
    print("   (예상 분석: 서로 다른 카테고리가 섞여서 매우 포괄적인 단어가 나왔을 것임)")

    # 3. 3가지 모드로 시각화
    modes = ["Basic Search", "CLIP Reranking", "Generative"]
    results = []

    print("\n[2] 시각화 및 점수 측정")
    for mode in modes:
        print(f"   Running {mode}...")
        result_src = visualize_and_save(mode, hypernym, clip_components)
        
        # 이미지 로드 및 점수 계산
        if result_src and result_src.startswith("http"):
             response = requests.get(result_src, timeout=10)
             res_img = Image.open(BytesIO(response.content)).convert("RGB")
        elif result_src:
             res_img = Image.open(result_src).convert("RGB")
        else:
             res_img = None

        final_score = calculate_score(hypernym, res_img, processor, model)
        print(f"     -> Score: {final_score} / Source: {result_src}")
        
        results.append({
            "Mode": mode,
            "Hypernym": hypernym,
            "CLIP Score": final_score,
            "Source": result_src
        })

    # 결과 저장
    df = pd.DataFrame(results)
    print("\n=== [이종 도메인 실험 결과] ===")
    print(df)
    df.to_csv("cross_group_results.csv", index=False)

if __name__ == "__main__":
    run_cross_group_experiment()