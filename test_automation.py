import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
from io import BytesIO
from dotenv import load_dotenv

# 프로젝트 루트 경로 설정 (모듈 임포트를 위해)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 기존 핵심 모듈 임포트
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
# ▶ 1. 실험 데이터셋 설정 (5개 그룹)
# TODO: 실제 테스트할 이미지 경로로 반드시 수정해주세요!
# ==========================================
experiment_groups = [
    {
        "name": "Group 1 (Fruit)",
        "texts": ["apple", "banana", "grape"],
        "images": ["./data/apple.png", "./data/banana.png", "./data/grape.png"] 
    },
    {
        "name": "Group 2 (Animal)",
        "texts": ["dog", "cat", "tiger"],
        "images": ["./data/dog.png", "./data/cat.png", "./data/tiger.png"]
    },
    {
        "name": "Group 3 (Vehicle)",
        "texts": ["car", "bus", "train"],
        "images": ["./data/car.png", "./data/bus.png", "./data/train.png"]
    },
    {
        "name": "Group 4 (Furniture)",
        "texts": ["chair", "sofa", "bed"],
        "images": ["./data/chair.png", "./data/sofa.png", "./data/bed.png"]
    },
    {
        "name": "Group 5 (Instrument)",
        "texts": ["guitar", "piano", "drum"],
        "images": ["./data/guitar.png", "./data/piano.png", "./data/drum.png"]
    }
]

# ==========================================
# ▶ 2. 헬퍼 함수 정의
# ==========================================
def load_image_from_path_or_url(path_or_url):
    """로컬 경로 또는 URL에서 이미지를 로드합니다."""
    try:
        if path_or_url.startswith("http"):
            response = requests.get(path_or_url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(path_or_url).convert("RGB")
        return image
    except Exception as e:
        print(f"이미지 로드 실패: {path_or_url} - Error: {e}")
        return None

def calculate_single_clip_score(text, image, processor, model):
    """단일 이미지와 텍스트 간의 CLIP Score를 계산합니다."""
    if image is None:
        return 0.0
    try:
        text_feat = clip_processor.get_text_features([text], processor, model)
        img_feat = clip_processor.get_image_features([image], processor, model)
        # 코사인 유사도 계산
        similarity = (text_feat @ img_feat.T).item()
        return round(similarity, 4)
    except Exception as e:
        print(f"점수 계산 중 오류 발생: {e}")
        return 0.0

# ==========================================
# ▶ 3. 메인 실험 실행 함수
# ==========================================
def run_experiments():
    print(">>> 🧪 실험 시작: 모델 로딩 중... (시간이 조금 걸립니다)")
    processor, model = clip_processor.load_clip_model()
    clip_components = (processor, model)
    
    results = [] # 결과 저장용 리스트

    for group in experiment_groups:
        print(f"\n--- [Testing Group: {group['name']}] ---")
        
        # 1) 입력 이미지 로드
        input_images = []
        for path in group['images']:
            img = load_image_from_path_or_url(path)
            if img: input_images.append(img)
        
        if len(input_images) < 2:
            print(f"경고: {group['name']} 그룹의 이미지가 부족하여 건너뜁니다.")
            continue

        # 2) 상위어 추론
        print("Step 1: 상위어 추론 중...")
        hypernym, _ = hypernym_extractor.determine_best_hypernym(
            input_images=input_images,
            input_texts=group['texts'],
            clip_model_components=clip_components
        )
        print(f"▶ 결정된 상위어: '{hypernym}'")

        # 3) 3가지 모델 실행 및 평가
        modes = [
            ("Basic Search", search_api_client.search_image, {"api_config": API_CONFIG}),
            ("CLIP Reranking", search_with_clip.search_and_rerank_image, {"api_config": API_CONFIG, "clip_components": clip_components}),
            ("Generative", sd_generator.generate_image_from_text, {"output_dir": GENERATED_IMAGES_DIR})
        ]

        for mode_name, func, kwargs in modes:
            print(f"Step 2: Running mode [{mode_name}]...")
            result_path_or_url = None
            try:
                # 각 모드 함수 실행
                if mode_name == "Generative":
                     result_path_or_url = func(hypernym, **kwargs)
                else:
                     result_path_or_url = func(hypernym, **kwargs)
            except Exception as e:
                print(f"Error in {mode_name}: {e}")

            # 결과 이미지 로드 및 점수 계산
            result_image = load_image_from_path_or_url(result_path_or_url) if result_path_or_url else None
            score = calculate_single_clip_score(hypernym, result_image, processor, model)
            
            print(f"  -> Result Score: {score}")

            # 결과 저장
            results.append({
                "Group": group['name'],
                "Hypernym": hypernym,
                "Model": mode_name,
                "CLIP Score": score
            })

    return results

# ==========================================
# ▶ 4. 결과 시각화 함수
# ==========================================
def visualize_results(df):
    print("\n>>> 📊 결과 시각화 생성 중...")
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(12, 7))
    
    # 막대 그래프 생성 (그룹별, 모델별 비교)
    barplot = sns.barplot(
        data=df,
        x="Group",
        y="CLIP Score",
        hue="Model",
        palette="viridis" # 색상 테마 (deep, muted, pastel, bright, dark, colorblind, viridis 등)
    )

    # 그래프 꾸미기
    plt.title("Quantitative Comparison of CLIP Scores by Group and Model", fontsize=16, fontweight='bold')
    plt.ylabel("CLIP Score (Cosine Similarity)", fontsize=12)
    plt.xlabel("Experiment Groups", fontsize=12)
    plt.xticks(rotation=15)
    plt.legend(title='Model Type', title_fontsize='12', loc='upper right')
    plt.ylim(0.15, 0.35) # Y축 범위 설정 (점수 분포에 따라 조절 가능)

    # 막대 위에 점수 표시
    for container in barplot.containers:
        barplot.bar_label(container, fmt='%.4f', padding=3, fontsize=10)

    plt.tight_layout()
    plt.savefig("experiment_result_graph.png", dpi=300) # 그래프 이미지 저장
    print("그래프가 'experiment_result_graph.png'로 저장되었습니다.")
    plt.show() # 화면에 표시

# ==========================================
# ▶ 메인 실행 블록
# ==========================================
if __name__ == "__main__":
    # 1. 실험 실행
    experiment_results = run_experiments()
    
    if experiment_results:
        # 2. 결과 데이터프레임 생성
        df = pd.DataFrame(experiment_results)
        
        print("\n\n=== [최종 실험 결과 데이터] ===")
        print(df)
        
        # CSV 파일로 저장 (논문 표 작성용)
        df.to_csv("final_experiment_results.csv", index=False)
        print("\n결과 데이터가 'final_experiment_results.csv'로 저장되었습니다.")

        # 3. 평균 점수 계산 및 출력
        print("\n=== [모델별 평균 CLIP Score] ===")
        avg_scores = df.groupby("Model")["CLIP Score"].mean().reset_index()
        print(avg_scores)
        
        # 4. 시각화 실행
        visualize_results(df)
    else:
        print("실험 결과가 없습니다. 이미지 경로를 확인해주세요.")