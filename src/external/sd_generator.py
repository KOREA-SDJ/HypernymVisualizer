"""Stable Diffusion 기반 이미지 생성 모듈.

이 모듈은 Stable Diffusion 모델(v1.5)을 사용하여 텍스트 설명(상위어)으로부터
새로운 이미지를 생성하는 기능을 제공합니다.
"""

import os
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline

_SD_PIPELINE = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_sd_model():
    """Stable Diffusion 파이프라인을 로드하고 초기화합니다.

    전역 변수 `_SD_PIPELINE`에 모델이 로드되어 있지 않은 경우,
    'runwayml/stable-diffusion-v1-5' 모델을 다운로드하고 초기화합니다.
    검열 오작동 방지를 위해 안전 필터(Safety Checker)는 비활성화됩니다.

    Returns:
        StableDiffusionPipeline: 초기화된 Stable Diffusion 파이프라인 객체.

    Raises:
        Exception: 모델 로드 중 오류가 발생한 경우.
    """
    global _SD_PIPELINE
    if _SD_PIPELINE is None:
        model_id = "runwayml/stable-diffusion-v1-5"
        print(f"Stable Diffusion 모델 로드 중... (장치: {_DEVICE})")
        
        try:
            if _DEVICE == "cuda":
                _SD_PIPELINE = StableDiffusionPipeline.from_pretrained(
                    model_id, torch_dtype=torch.float16
                )
            else:
                _SD_PIPELINE = StableDiffusionPipeline.from_pretrained(model_id)

            # 안전 필터 비활성화 (Dummy 함수로 대체)
            def dummy_safety_checker(images, clip_input):
                return images, [False] * len(images)

            _SD_PIPELINE.safety_checker = dummy_safety_checker
            _SD_PIPELINE.requires_safety_checker = False

            _SD_PIPELINE.to(_DEVICE)
            print("Stable Diffusion 모델 로드 완료.")
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            raise e
    
    return _SD_PIPELINE


def generate_image_from_text(hypernym_text: str, output_dir: str = "generated_images") -> Optional[str]:
    """상위어 텍스트를 기반으로 이미지를 생성하고 저장합니다.

    입력된 상위어를 고품질 사진 스타일의 프롬프트로 변환하여 이미지를 생성한 후,
    지정된 디렉토리에 PNG 파일로 저장합니다.

    Args:
        hypernym_text (str): 생성할 이미지의 주제(상위어).
        output_dir (str, optional): 이미지가 저장될 디렉토리 경로. 기본값은 "generated_images".

    Returns:
        Optional[str]: 저장된 이미지 파일의 경로. 생성 실패 시 None.
    """
    pipeline = load_sd_model()
    
    # 프롬프트 설정 (고품질, 사실적 사진 스타일 유도)
    prompt = (f"a high quality photo representation of {hypernym_text}, "
              f"professional photography, studio lighting, clear focus, 4k, highly detailed, realistic")
    
    # 부정 프롬프트 설정 (저품질, 왜곡, 텍스트 배제)
    negative_prompt = ("low quality, blurry, distorted, deformed, text, watermark, signature, "
                       "logo, messy, dark, cartoon, illustration, painting")

    print(f"이미지 생성 시작... 프롬프트: '{prompt}'")

    try:
        image = pipeline(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            height=512, 
            width=512,
            num_inference_steps=30
        ).images[0]

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 파일명 생성 (특수문자 제거)
        safe_name = "".join(x for x in hypernym_text if x.isalnum())
        file_path = os.path.join(output_dir, f"{safe_name}_generated.png")
        
        image.save(file_path)
        print(f"이미지 생성 및 저장 완료: {file_path}")
        
        return file_path.replace("\\", "/")

    except Exception as e:
        print(f"이미지 생성 중 오류 발생: {e}")
        if "CUDA out of memory" in str(e):
            print("팁: GPU 메모리가 부족합니다. 이미지 크기(height, width)를 줄이거나 실행 중인 다른 GPU 프로그램을 종료하세요.")
        return None