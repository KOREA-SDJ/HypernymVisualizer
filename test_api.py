import requests
import json
from PIL import Image
from io import BytesIO

# -----------------
# 1. 테스트 데이터 준비 (실제 이미지 파일 경로로 변경해야 함)
# -----------------
image_paths = ['data/banana.png', 'data/apple.png']
texts = ['banana', 'apple']

# (선택) 임시 이미지 생성 (테스트 파일이 없다면)
def create_dummy_image(filename):
    """테스트용 빈 이미지 파일을 생성합니다."""
    img = Image.new('RGB', (10, 10), color = 'yellow')
    img.save(filename)
    print(f"{filename} 생성 완료.")

try:
    for path in image_paths:
        with open(path, 'rb') as f:
            pass # 파일 존재 확인
except FileNotFoundError:
    print("경고: 테스트 이미지를 찾을 수 없습니다. 임시 더미 이미지를 생성합니다.")
    for path in image_paths:
        create_dummy_image(path)


# 2. 요청 데이터 포맷팅
files_data = []
for path in image_paths:
    # 'files' 필드에 튜플 형태로 (필드 이름, 파일 핸들, MIME 타입) 전달
    # FastAPI는 'files'라는 이름으로 여러 파일을 받기를 기대합니다.
    files_data.append(('files', (path.split('/')[-1], open(path, 'rb'), 'image/png')))

# 'texts' 필드를 폼 데이터로 변환 (각 항목은 별도의 튜플로 전달)
# FastAPI는 List[str]을 받기 위해 같은 필드 이름('texts')으로 여러 값을 기대합니다.
texts_data = [('texts', t) for t in texts]

# files_data와 texts_data를 결합
data = texts_data + files_data

# 3. API 호출
API_URL = "http://127.0.0.1:8000/extract_and_visualize/"

print("\nAPI 호출 시작...")
try:
    response = requests.post(API_URL, files=files_data, data=texts_data)
    
    # 파일을 닫습니다.
    for _, file_tuple in files_data:
        file_tuple[1].close()

    # 4. 결과 출력
    response.raise_for_status()
    print("\n--- 서버 응답 성공 (200 OK) ---")
    print(json.dumps(response.json(), indent=4, ensure_ascii=False))

except requests.exceptions.RequestException as e:
    print(f"\n--- 요청 실패 ---")
    print(f"상태 코드: {response.status_code if 'response' in locals() else 'N/A'}")
    print(f"오류 상세: {response.text if 'response' in locals() else e}")
    
# 임시 이미지가 생성되었다면 삭제하는 로직 추가