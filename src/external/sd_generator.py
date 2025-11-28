"""
이 모듈은 Stable Diffusion 모델을 사용하여 텍스트로부터 이미지를 생성합니다.
상위어(Hypernym) 개념을 AI가 해석하여 새로운 이미지를 그려냅니다.
"""

import torch
from diffusers import StableDiffusionPipeline
import os
from typing import Optional

# 전역 변수로 파이프라인을 저장하여 매번 모델을 로드하는 것을 방지합니다.
_SD_PIPELINE = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_sd_model():
    """Stable Diffusion 파이프라인을 로드하고 초기화합니다."""
    global _SD_PIPELINE
    if _SD_PIPELINE is None:
        # Hugging Face Hub에서 사전 학습된 모델 ID (Stable Diffusion v1.5)
        model_id = "runwayml/stable-diffusion-v1-5"
        print(f"Stable Diffusion 모델 로드 중... (장치: {_DEVICE})")
        print("최초 실행 시 수 GB의 모델 다운로드가 진행되므로 시간이 걸릴 수 있습니다.")
        
        try:
            if _DEVICE == "cuda":
                # GPU 사용 시 float16 타입으로 로드하여 메모리 절약 및 속도 향상
                _SD_PIPELINE = StableDiffusionPipeline.from_pretrained(
                    model_id, torch_dtype=torch.float16
                )
                # xformers가 설치되어 있다면 메모리 효율성을 높일 수 있음 (선택 사항)
                # _SD_PIPELINE.enable_xformers_memory_efficient_attention()
            else:
                # CPU 사용 시 기본 float32 타입으로 로드 (속도가 매우 느릴 수 있음)
                _SD_PIPELINE = StableDiffusionPipeline.from_pretrained(model_id)
                
            _SD_PIPELINE.to(_DEVICE)
            
            # 안전 검열기(Safety Checker) 비활성화 (필요에 따라 주석 해제)
            # 모델이 검은 이미지만 출력한다면 이 부분을 해제해보세요.
            # _SD_PIPELINE.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
            
            print("Stable Diffusion 모델 로드 완료.")
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            # GPU 메모리 부족 오류(CUDA out of memory)가 발생하면
            # 1. torch_dtype=torch.float16 확인
            # 2. enable_attention_slicing() 사용 고려
            raise e
    
    return _SD_PIPELINE

def generate_image_from_text(hypernym_text: str, output_dir: str = "generated_images") -> Optional[str]:
    """
    상위어 텍스트를 기반으로 이미지를 생성하고 로컬 파일로 저장합니다.

    Args:
        hypernym_text (str): 모델 1에서 추출된 상위어.
        output_dir (str): 생성된 이미지를 저장할 폴더 경로. 기본값은 'generated_images'.

    Returns:
        Optional[str]: 저장된 이미지 파일의 상대 경로 (예: "generated_images/fruit_gen.png").
                       생성 실패 시 None 반환.
    """
    pipeline = load_sd_model()
    
    # 1. 프롬프트 엔지니어링
    # 상위어의 특징을 잘 살리면서 고품질의 이미지를 얻기 위한 프롬프트 설정
    prompt = (f"a high quality photo representation of {hypernym_text}, "
              f"professional photography, studio lighting, clear focus, 4k, highly detailed, realistic")
    
    # 원하지 않는 요소를 배제하기 위한 부정 프롬프트
    negative_prompt = ("low quality, blurry, distorted, deformed, text, watermark, signature, "
                       "logo, messy, dark, cartoon, illustration, painting")

    print(f"이미지 생성 시작... 프롬프트: '{prompt}'")

    try:
        # 2. 이미지 생성 실행
        # GPU 메모리가 부족할 경우 height와 width를 512 미만으로 줄여보세요.
        image = pipeline(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            height=512, 
            width=512,
            num_inference_steps=30 # 추론 단계 수 (높을수록 고품질이나 느려짐, 기본 50)
        ).images[0]

        # 3. 이미지 파일 저장
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 파일명 생성 (알파벳과 숫자만 남기고 공백 제거)
        safe_name = "".join(x for x in hypernym_text if x.isalnum())
        # 중복 방지를 위해 타임스탬프나 난수를 추가하는 것도 좋음
        file_path = os.path.join(output_dir, f"{safe_name}_generated.png")
        
        image.save(file_path)
        print(f"이미지 생성 및 저장 완료: {file_path}")
        
        # 웹에서 접근 가능한 상대 경로 반환 (백슬래시를 슬래시로 변환)
        return file_path.replace("\\", "/")

    except Exception as e:
        print(f"이미지 생성 중 오류 발생: {e}")
        if "CUDA out of memory" in str(e):
            print("팁: GPU 메모리가 부족합니다. 이미지 크기(height, width)를 줄이거나 실행 중인 다른 GPU 프로그램을 종료하세요.")
        return None