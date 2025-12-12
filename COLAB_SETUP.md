# Google Colab에서 실행하기 - 빠른 가이드

## 🚀 5분 안에 시작하기

### 1단계: Colab 열기

1. [Google Colab](https://colab.research.google.com) 접속
2. `파일` → `노트북 업로드` 클릭
3. `evaluate_colab.ipynb` 파일 선택

**또는** GitHub에서 직접 열기:

```
https://colab.research.google.com/github/mlnyx/-prompt-firewall/blob/main/evaluate_colab.ipynb
```

### 2단계: GPU 설정

1. 메뉴: `런타임` → `런타임 유형 변경`
2. `하드웨어 가속기`: **GPU** 선택
3. `GPU 유형`: **T4** (무료) 또는 더 강력한 GPU
4. `저장` 클릭

### 3단계: Hugging Face 토큰 준비

Llama 3 모델을 사용하려면 Hugging Face 토큰이 필요합니다:

#### 토큰 생성

1. [Hugging Face 가입](https://huggingface.co/join) (무료)
2. [토큰 생성 페이지](https://huggingface.co/settings/tokens) 이동
3. `New token` 클릭
   - Name: `colab-llama3`
   - Type: `Read` 선택
4. 생성된 토큰 복사 (hf_xxxxxxxxxxxx 형태)

#### Llama 3 접근 권한 요청

1. [Llama 3 8B Instruct 페이지](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) 방문
2. `Request access to this model` 클릭
3. 약관 동의 후 제출
4. **즉시 승인됨** (보통 5분 이내)

### 4단계: 노트북 실행

노트북의 셀을 순서대로 실행:

**셀 1: GPU 확인**

```python
!nvidia-smi
```

출력에서 GPU가 할당되었는지 확인

**셀 2: 저장소 클론**

```python
!git clone https://github.com/mlnyx/-prompt-firewall.git
%cd -prompt-firewall
```

**셀 3: 패키지 설치**

```python
!pip install -q transformers torch sentence-transformers pyyaml pandas tqdm accelerate bitsandbytes
```

약 2-3분 소요

**셀 4: Hugging Face 로그인**

```python
from huggingface_hub import login

# 여기에 토큰 입력
HF_TOKEN = "hf_여기에_복사한_토큰_붙여넣기"
login(token=HF_TOKEN)
```

**셀 5 이후**: 노트북 가이드 따라 실행

## ⚡ 빠른 테스트 (5분)

전체 평가 대신 샘플만 테스트:

```python
# 노트북의 마지막 섹션 "6. 샘플링 테스트" 실행
!python quick_sample_test.py
```

**결과**: 100개 샘플로 Stage 3 성능 검증 (약 5-10분)

## 📊 전체 평가 (선택사항, 2-3시간)

```python
!python evaluate.py
```

**결과**:

- Stage 1: ~1분
- Stage 2: ~10분 (GPU T4 기준)
- Stage 3: ~2-3시간 (5,120개 항목)

## 🔧 문제 해결

### GPU 메모리 부족

```
RuntimeError: CUDA out of memory
```

**해결**: 샘플 크기를 줄이거나 `quick_sample_test.py` 사용

### Llama 3 접근 거부

```
401 Unauthorized: Access to model meta-llama/Meta-Llama-3-8B-Instruct is restricted
```

**해결**:

1. [Llama 3 페이지](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)에서 접근 권한 재요청
2. 승인 이메일 확인 (5-10분)
3. 토큰이 올바른지 확인

### 모델 다운로드 느림

**정상**: Llama 3 8B는 약 4.7GB입니다. Colab 네트워크에서 5-10분 소요

## 💡 유용한 팁

### Colab 세션 유지

- Colab은 90분 idle 시 연결 끊김
- 긴 작업 중에는 가끔 노트북 클릭

### 중간 결과 저장

```python
# stage2_rewrites.txt가 자동 생성됨
from google.colab import files
files.download('stage2_rewrites.txt')
```

### GPU 할당량

- 무료 Colab: 하루 GPU 사용 제한
- 제한 걸리면 다음날 재시도 또는 Colab Pro 사용

## 📈 예상 결과

성공적으로 실행되면:

```
[Stage 1] 실행 중 (20122 seeds)...
  - ALLOW: 3,500개
  - BLOCK: 1,850개
  - ESCALATE: 14,772개

[Stage 2] 실행 중 (14772 seeds)...
  - ALLOW: 8,200개
  - BLOCK: 1,452개
  - REWRITE: 5,120개

[Stage 3] 샘플 테스트 (512 seeds)...
  - 성공: 371개
  - 실패: 141개
```

## 🔄 로컬로 결과 가져오기

Colab에서 결과 다운로드:

```python
from google.colab import files
files.download('stage2_rewrites.txt')
files.download('data/s2_all_scores.csv')
```

로컬에서:

```bash
# 다운로드 폴더 → 프로젝트 폴더
mv ~/Downloads/stage2_rewrites.txt /Users/mlnyx/-prompt-firewall/
```

## 📞 도움이 필요하면

- [GitHub Issues](https://github.com/mlnyx/-prompt-firewall/issues)
- `README_COLAB.md` 참조 (더 자세한 설명)
- `evaluate_colab.ipynb` 주석 확인

---

**행운을 빕니다! 🎉**
