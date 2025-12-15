window.onload = function() {
    addInputPair();
    addInputPair();
};

function addInputPair() {
    const container = document.getElementById('input-list');
    const div = document.createElement('div');
    div.className = 'input-pair';
    
    div.innerHTML = `
        <input type="file" accept="image/*" class="file-input">
        <input type="text" placeholder="이미지 설명 (예: apple)" class="text-input">
        <button type="button" class="remove-btn" onclick="removeInputPair(this)">❌</button>
    `;
    container.appendChild(div);
}

function removeInputPair(button) {
    const inputPair = button.parentElement;
    inputPair.remove();
}

async function submitData() {
    const fileInputs = document.querySelectorAll('.file-input');
    const textInputs = document.querySelectorAll('.text-input');
    const mode = document.getElementById('mode-select').value;
    const loadingDiv = document.getElementById('loading');
    const resultDiv = document.getElementById('result-section');

    const formData = new FormData();
    let pairCount = 0;

    for (let i = 0; i < fileInputs.length; i++) {
        if (fileInputs[i].files[0] && textInputs[i].value) {
            formData.append('files', fileInputs[i].files[0]);
            formData.append('texts', textInputs[i].value);
            pairCount++;
        }
    }

    if (pairCount < 2) {
        alert("최소 2쌍 이상의 이미지와 텍스트를 입력해야 분석이 가능합니다!");
        return;
    }

    loadingDiv.classList.remove('hidden');
    resultDiv.classList.add('hidden');

    try {
        const response = await fetch(`http://127.0.0.1:8000/extract_and_visualize/?visualization_mode=${mode}`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        console.log("서버에서 받은 데이터:", data); // ◀ 이 줄을 추가하고 F12(개발자 도구) 콘솔을 확인하세요!
        loadingDiv.classList.add('hidden');

        if (response.ok) {
            // 결과 데이터 연결
            document.getElementById('res-hypernym').innerText = data.hypernym;
            document.getElementById('res-score').innerText = data.confidence_score;
            
            // ▼▼▼ [새로 추가된 부분] CLIP Score 값 넣기 ▼▼▼
            // 값이 없으면 0.0으로 표시
            document.getElementById('res-clip-score').innerText = data.final_clip_score || "0.0";
            // ▲▲▲ [새로 추가된 부분] 끝 ▲▲▲

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