import os
import sys
import requests
import pandas as pd
import shutil  # 파일 이동/이름변경을 위해 추가
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
# ▶ 1. 실험 데이터셋
# ==========================================
experiment_groups = [
    {
        "name": "Group 1 (Fruit)",
        "inputs": [
            {"text": "apple", "image": "./data/apple.png"},
            {"text": "banana", "image": "./data/banana.png"},
            {"text": "grape", "image": "./data/grape.png"}
        ]
    },
    {
        "name": "Group 2 (Animal)",
        "inputs": [
            {"text": "dog", "image": "./data/dog.png"},
            {"text": "cat", "image": "./data/cat.png"},
            {"text": "tiger", "image": "./data/tiger.png"}
        ]
    },
    {
        "name": "Group 3 (Vehicle)",
        "inputs": [
            {"text": "car", "image": "./data/car.png"},
            {"text": "bus", "image": "./data/bus.png"},
            {"text": "train", "image": "./data/train.png"}
        ]
    },
    {
        "name": "Group 4 (Furniture)",
        "inputs": [
            {"text": "chair", "image": "./data/chair.png"},
            {"text": "sofa", "image": "./data/sofa.png"},
            {"text": "bed", "image": "./data/bed.png"}
        ]
    },
    {
        "name": "Group 5 (Instrument)",
        "inputs": [
            {"text": "guitar", "image": "./data/guitar.png"},
            {"text": "piano", "image": "./data/paino.png"}, # 오타 유지
            {"text": "drum", "image": "./data/drum.png"}
        ]
    }
]

def load_image_content(path_or_url):
    try:
        if path_or_url.startswith("http"):
            response = requests.get(path_or_url, timeout=10)
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            return Image.open(path_or_url).convert("RGB")
    except Exception as e:
        print(f"    [Error] 이미지 로드 실패 ({path_or_url}): {e}")
        return None

def calculate_score(text, image_obj, processor, model):
    if image_obj is None: return 0.0
    try:
        text_feat = clip_processor.get_text_features([text], processor, model)
        img_feat = clip_processor.get_image_features([image_obj], processor, model)
        return round((text_feat @ img_feat.T).item(), 4)
    except:
        return 0.0

# [수정됨] 파일명을 구분하기 위해 suffix(꼬리표) 파라미터 추가
def visualize_by_mode(mode, hypernym, clip_components, suffix=""):
    url_or_path = None
    try:
        if mode == "Basic Search":
            url_or_path = search_api_client.search_image(hypernym, API_CONFIG)
        elif mode == "CLIP Reranking":
            url_or_path = search_with_clip.search_and_rerank_image(
                hypernym, API_CONFIG, clip_components
            )
        elif mode == "Generative":
            # 1. 일단 기본 이름으로 생성
            original_path = sd_generator.generate_image_from_text(hypernym, GENERATED_IMAGES_DIR)
            
            # 2. 파일명 변경 로직 (덮어쓰기 방지)
            if original_path and suffix:
                # 예: furniture_generated.png -> furniture_batch.png
                dir_name = os.path.dirname(original_path)
                file_name = os.path.basename(original_path)
                
                # _generated 부분을 제거하고 suffix 붙이기
                clean_name = file_name.replace("_generated.png", "").replace(".png", "")
                new_filename = f"{clean_name}_{suffix}.png"
                new_path = os.path.join(dir_name, new_filename)
                
                # 파일 이름 변경 (이동)
                shutil.move(original_path, new_path)
                url_or_path = new_path
            else:
                url_or_path = original_path
                
    except Exception as e:
        print(f"      [Visual Error] {mode}: {e}")
        
    return url_or_path

def run_full_sequential_test():
    print(">>> 🔄 [Full Sequential v2] 실험 시작: 파일 덮어쓰기 방지 모드")
    processor, model = clip_processor.load_clip_model()
    clip_components = (processor, model)
    
    results = []

    for group in experiment_groups:
        print(f"\n==================================================")
        print(f" 🧪 Testing Group: {group['name']}")
        print(f"==================================================")
        
        items = group['inputs']
        raw_imgs = [load_image_content(i['image']) for i in items]
        raw_texts = [i['text'] for i in items]
        
        if any(img is None for img in raw_imgs):
            continue

        # -------------------------------------------------
        # 1. 기준값 (Batch): A+B+C
        # -------------------------------------------------
        print(f"  [1] Batch Processing (A+B+C)...")
        batch_hyp, batch_conf = hypernym_extractor.determine_best_hypernym(
            raw_imgs, raw_texts, clip_components
        )
        print(f"   -> Batch Result: '{batch_hyp}'")
        
        # [NEW] 배치 기준 이미지 생성 및 저장 (Generative의 경우 _batch.png로 저장)
        # 비교를 위해 배치 때도 생성을 한 번 수행합니다.
        batch_gen_path = visualize_by_mode("Generative", batch_hyp, clip_components, suffix="batch")
        
        # 배치 생성 이미지 점수 계산
        batch_img_obj = load_image_content(batch_gen_path)
        batch_gen_score = calculate_score(batch_hyp, batch_img_obj, processor, model)
        print(f"   -> Batch Gen Image saved: {batch_gen_path} (Score: {batch_gen_score})")


        # -------------------------------------------------
        # 2. 순차적 처리 시작
        # -------------------------------------------------
        print(f"  [2] Step 1: Processing (A+B)...")
        inter_hyp, inter_conf = hypernym_extractor.determine_best_hypernym(
            raw_imgs[:2], raw_texts[:2], clip_components
        )
        print(f"   -> Intermediate Hypernym: '{inter_hyp}'")

        modes = ["Basic Search", "CLIP Reranking", "Generative"]
        
        for mode in modes:
            print(f"\n    ---- [Pipeline: {mode}] ----")
            
            # (1) 중간 이미지 생성
            print(f"      Visualizing '{inter_hyp}'...")
            inter_img_path = visualize_by_mode(mode, inter_hyp, clip_components) # 중간 이미지는 덮어써도 무방
            inter_img_obj = load_image_content(inter_img_path) if inter_img_path else None
            
            if inter_img_obj is None: continue

            # (2) 최종 추론
            print(f"      Reasoning with {{Intermediate Img + C}}...")
            final_inputs_imgs = [inter_img_obj, raw_imgs[2]]
            final_inputs_texts = [inter_hyp, raw_texts[2]] 
            
            seq_final_hyp, seq_conf = hypernym_extractor.determine_best_hypernym(
                final_inputs_imgs, final_inputs_texts, clip_components
            )
            print(f"      -> Final Hypernym: '{seq_final_hyp}'")
            
            # (3) 최종 시각화 (여기서 suffix="seq" 사용!)
            print(f"      Final Visualization via {mode}...")
            # Generative일 때만 _seq가 붙고, 검색 기반은 URL이라 영향 없음
            final_img_path = visualize_by_mode(mode, seq_final_hyp, clip_components, suffix="seq")
            final_img_obj = load_image_content(final_img_path)
            
            final_score = calculate_score(seq_final_hyp, final_img_obj, processor, model)
            print(f"      -> Final CLIP Score: {final_score}")

            # 결과 저장
            is_match = (batch_hyp == seq_final_hyp)
            
            # CSV 저장을 위해 경로/URL 기록
            saved_path_info = final_img_path if mode == "Generative" else "URL_Image"

            results.append({
                "Group": group['name'],
                "Pipeline Model": mode,
                "Batch Hypernym": batch_hyp,
                "Batch Gen Score": batch_gen_score, # 배치의 생성 모델 점수
                "Sequential Final Hypernym": seq_final_hyp,
                "Consistency Match": "O" if is_match else "X",
                "Final CLIP Score": final_score,
                "Final Image Path": saved_path_info
            })

    if results:
        df = pd.DataFrame(results)
        print("\n\n=== [순차적 실험 최종 결과 (v2)] ===")
        print(df)
        df.to_csv("sequential_full_results_v2.csv", index=False)

if __name__ == "__main__":
    run_full_sequential_test()