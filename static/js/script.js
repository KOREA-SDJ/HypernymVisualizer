// 초기 입력 칸 2개 생성
window.onload = function() {
    addInputPair();
    addInputPair();
};

// [수정] 입력 칸 추가 함수 (삭제 버튼 포함)
function addInputPair() {
    const container = document.getElementById('input-list');
    const div = document.createElement('div');
    div.className = 'input-pair';
    
    // HTML 구조에 <button> 태그(삭제 버튼) 추가
    div.innerHTML = `
        <input type="file" accept="image/*" class="file-input">
        <input type="text" placeholder="이미지 설명 (예: apple)" class="text-input">
        <button type="button" class="remove-btn" onclick="removeInputPair(this)">❌</button>
    `;
    container.appendChild(div);
}

// [NEW] 입력 칸 삭제 함수
function removeInputPair(button) {
    // 현재 리스트에 있는 입력 칸 개수 확인
    const container = document.getElementById('input-list');
    
    // (선택 사항) 최소 2개는 남겨두고 싶다면 아래 주석을 해제하세요.
    // if (container.children.length <= 2) {
    //     alert("최소 2개의 입력은 필요합니다.");
    //     return;
    // }

    // 버튼의 부모 요소(div.input-pair)를 찾아서 삭제
    const inputPair = button.parentElement;
    inputPair.remove();
}

async function submitData() {
    const fileInputs = document.querySelectorAll('.file-input');
    const textInputs = document.querySelectorAll('.text-input');
    const mode = document.getElementById('mode-select').value;
    const loadingDiv = document.getElementById('loading');
    const resultDiv = document.getElementById('result-section');

    // FormData 생성
    const formData = new FormData();
    let pairCount = 0;

    // 파일과 텍스트 쌍을 FormData에 추가
    for (let i = 0; i < fileInputs.length; i++) {
        if (fileInputs[i].files[0] && textInputs[i].value) {
            formData.append('files', fileInputs[i].files[0]);
            formData.append('texts', textInputs[i].value);
            pairCount++;
        }
    }

    // [중요] 최소 2쌍 이상인지 확인 (백엔드 로직 때문)
    if (pairCount < 2) {
        alert("최소 2쌍 이상의 이미지와 텍스트를 입력해야 분석이 가능합니다!");
        return;
    }

    // UI 상태 변경
    loadingDiv.classList.remove('hidden');
    resultDiv.classList.add('hidden');

    try {
        // API 호출
        const response = await fetch(`http://127.0.0.1:8000/extract_and_visualize/?visualization_mode=${mode}`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        loadingDiv.classList.add('hidden');

        if (response.ok) {
            // 결과 표시
            document.getElementById('res-hypernym').innerText = data.hypernym;
            document.getElementById('res-score').innerText = data.confidence_score;
            document.getElementById('res-mode').innerText = data.visualization_mode;
            
            const imgElement = document.getElementById('res-image');
            if (data.final_image_url) {
                imgElement.src = data.final_image_url;
                imgElement.style.display = 'block';
            } else {
                imgElement.style.display = 'none';
                alert("이미지를 찾거나 생성하지 못했습니다.");
            }
            
            resultDiv.classList.remove('hidden');
        } else {
            alert("오류 발생: " + (data.detail || "알 수 없는 오류"));
        }

    } catch (error) {
        loadingDiv.classList.add('hidden');
        alert("서버 연결 실패! 백엔드가 실행 중인지 확인하세요.\n" + error);
    }
}