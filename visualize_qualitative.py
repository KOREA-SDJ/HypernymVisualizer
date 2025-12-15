import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import requests
from io import BytesIO
import os

# ==========================================
# ▶ 데이터 설정 (Group 4: Furniture 예시)
# ※ 실험 후 실제 로그의 URL과 점수로 업데이트하세요!
# ==========================================
data = {
    "title": "Robustness Comparison: Batch vs. Sequential Processing (Group 4: Furniture)",
    
    # -----------------------------------------------------------
    # 1. [Batch Result] A+B+C 배치 처리 결과
    # -----------------------------------------------------------
    "batch_results": [
        {
            "mode": "Basic Search",
            "url": "https://www.snydersfurniture.com/cdn/shop/products/wooden-building-blocks-194003.jpg?v=1686239355",
            "score": 0.2429
        },
        {
            "mode": "CLIP Reranking",
            "url": "https://thumbs.dreamstime.com/b/assorted-cartoon-furniture-elements-isolated-white-background-design-projects-collection-cartoon-style-furniture-items-411838317.jpg",
            "score": 0.3121
        },
        {
            "mode": "Generative",
            # [중요] 배치 실험에서 생성된 이미지 (_batch.png)
            "path": "generated_images/furniture_batch.png",
            "score": 0.2579
        }
    ],

    # -----------------------------------------------------------
    # 2. [Sequential Inputs] 중간 입력 과정
    # -----------------------------------------------------------
    "seq_inputs": [
        {
            "label": "Step 1 Output\n(Intermediate: Seat)", 
            "path": "generated_images/seat_generated.png"
        },
        {
            "label": "New Input\n(Input C: Bed)", 
            "path": "./data/bed.png" 
        }
    ],

    # -----------------------------------------------------------
    # 3. [Sequential Output] 순차 처리 결과
    # -----------------------------------------------------------
    "seq_outputs": [
        {
            "mode": "Basic Search",
            "url": "https://www.snydersfurniture.com/cdn/shop/products/wooden-building-blocks-194003.jpg?v=1686239355",
            "score": 0.2429
        },
        {
            "mode": "CLIP Reranking",
            "url": "https://i5.samsclubimages.com/asr/b83d4992-b268-4b03-9c69-db96e48edce4.fcddd390c359fbe2c4f845020dd3000f.jpeg?odnHeight=612&odnWidth=612&odnBg=FFFFFF",
            "score": 0.2449
        },
        {
            "mode": "Generative",
            # [중요] 순차 실험에서 생성된 이미지 (_seq.png)
            "path": "generated_images/furniture_seq.png",
            "score": 0.2374
        }
    ]
}

# ==========================================
# ▶ 헬퍼 함수
# ==========================================
def load_image(source):
    try:
        if source.startswith("http"):
            response = requests.get(source, timeout=5)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            if os.path.exists(source):
                img = Image.open(source).convert("RGB")
            else:
                print(f"파일 없음: {source}")
                # 파일이 없을 경우 흰색 빈 이미지 반환 (에러 방지)
                return Image.new('RGB', (200, 200), color='#f0f0f0')
        return img
    except Exception as e:
        print(f"이미지 로드 실패 ({source}): {e}")
        return Image.new('RGB', (200, 200), color='#f0f0f0')

# ==========================================
# ▶ 시각화 생성 로직
# ==========================================
def create_full_comparison(data):
    # 세로 길이를 넉넉하게 설정 (14x12)
    fig = plt.figure(figsize=(14, 12)) 
    fig.patch.set_facecolor('white')
    
    # 간격 조정 (hspace=0.6)으로 텍스트 겹침 방지
    gs = gridspec.GridSpec(3, 6, figure=fig, height_ratios=[1, 0.8, 1], hspace=0.6)

    # --- 1. Top Row: Batch Results ---
    plt.figtext(0.5, 0.92, "1. Baseline: Batch Processing (A + B + C)", ha='center', fontsize=14, fontweight='bold', color='#333')
    
    for i, item in enumerate(data['batch_results']):
        ax = fig.add_subplot(gs[0, i*2 : (i+1)*2])
        img = load_image(item.get('url') or item.get('path'))
        ax.imshow(img)
        ax.set_title(f"{item['mode']}\n(Score: {item['score']:.4f})", fontsize=11, pad=10)
        ax.axis('off')

    # --- 2. Middle Row: Sequential Inputs ---
    plt.figtext(0.5, 0.65, "⬇  2. Sequential Step: Intermediate Result + New Input  ⬇", ha='center', fontsize=12, style='italic', color='gray')
    
    # 중앙 정렬을 위해 그리드 인덱스 조정
    ax_mid1 = fig.add_subplot(gs[1, 1:3]) 
    ax_mid2 = fig.add_subplot(gs[1, 3:5])
    
    mid_axes = [ax_mid1, ax_mid2]
    for i, item in enumerate(data['seq_inputs']):
        ax = mid_axes[i]
        img = load_image(item['path'])
        ax.imshow(img)
        ax.set_title(item['label'], fontsize=11, fontweight='bold', color='#444', pad=10)
        ax.axis('off')
        
        if i == 0:
            plt.figtext(0.5, 0.51, "+", ha='center', va='center', fontsize=24, fontweight='bold')

    # --- 3. Bottom Row: Sequential Outputs ---
    plt.figtext(0.5, 0.38, "3. Result: Sequential Processing ((A + B) + C)", ha='center', fontsize=14, fontweight='bold', color='#333')
    
    for i, item in enumerate(data['seq_outputs']):
        ax = fig.add_subplot(gs[2, i*2 : (i+1)*2])
        img = load_image(item.get('url') or item.get('path'))
        ax.imshow(img)
        
        # Reranking 모델 강조 (Bold 폰트 및 테두리)
        title_font = 'bold' if "Reranking" in item['mode'] else 'normal'
        ax.set_title(f"{item['mode']}\n(Score: {item['score']:.4f})", fontsize=11, fontweight=title_font, pad=10)
        
        if "Reranking" in item['mode']:
            for spine in ax.spines.values():
                spine.set_edgecolor('#2ecc71') # 녹색 테두리
                spine.set_linewidth(3)
        else:
            ax.axis('off')

    plt.suptitle(data['title'], fontsize=16, y=0.97, fontweight='bold')
    
    save_path = "full_comparison_result_v2.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 최종 시각화 이미지가 저장되었습니다: {save_path}")
    plt.show()

if __name__ == "__main__":
    create_full_comparison(data)