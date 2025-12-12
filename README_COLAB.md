# Google Colab에서 실행하기

맥북의 PyTorch mutex lock 문제를 피하기 위해 Google Colab에서 실행하는 방법입니다.

## 🚀 빠른 시작

1. **Google Colab 열기**

   ```
   https://colab.research.google.com
   ```

2. **노트북 업로드**

   - `evaluate_colab.ipynb` 파일을 Colab에 업로드

3. **GPU 런타임 설정**

   - 메뉴: `런타임` → `런타임 유형 변경` → `하드웨어 가속기: GPU (T4)`

4. **셀 순서대로 실행**
   - ▶️ 버튼을 눌러 각 셀을 실행

## 📋 사전 준비

### Hugging Face 토큰 발급

Llama 3 모델을 사용하려면 Hugging Face 계정과 토큰이 필요합니다:

1. **계정 생성**: https://huggingface.co/join
2. **토큰 생성**: https://huggingface.co/settings/tokens

   - `New token` 클릭
   - Name: `colab-llama3`
   - Type: `Read`
   - 생성된 토큰 복사 (hf\_...)

3. **Llama 3 접근 권한 요청**

   - https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct 방문
   - `Request access to this model` 클릭
   - Meta의 승인 대기 (보통 즉시 승인됨)

4. **노트북에 토큰 입력**
   ```python
   HF_TOKEN = "hf_여기에토큰붙여넣기"
   ```

## 🔧 주요 차이점

### 로컬 (맥북) vs Colab

| 항목        | 로컬                     | Colab                                    |
| ----------- | ------------------------ | ---------------------------------------- |
| LLM 서버    | Ollama (localhost:11434) | Hugging Face Transformers                |
| 모델 로드   | `ollama pull llama3`     | `AutoModelForCausalLM.from_pretrained()` |
| 메모리      | 전체 모델 로드           | 4-bit 양자화 (메모리 절약)               |
| Stage3 코드 | `stage3_rewriter.py`     | `stage3_rewriter_hf.py`                  |
| mutex lock  | ❌ 발생                  | ✅ 없음                                  |

## 📊 실행 옵션

### 1. 전체 평가 (20,122개 항목)

```python
!python evaluate.py
```

**예상 시간**:

- Stage 1: ~1분
- Stage 2: ~10-15분
- Stage 3: ~2-3시간 (GPU T4 기준)

### 2. 샘플링 테스트 (100개)

```python
!python quick_sample_test.py
```

**예상 시간**: ~5-10분

Stage 2에서 REWRITE 판정을 받은 항목 중 100개만 랜덤 샘플링하여 테스트합니다.

## 🐛 문제 해결

### GPU 메모리 부족

```python
RuntimeError: CUDA out of memory
```

**해결책**:

- `quick_sample_test.py`로 샘플 크기 줄이기
- 또는 더 강력한 GPU 사용 (Colab Pro - A100)

### Llama 3 접근 권한 오류

```
401 Unauthorized: Access to model meta-llama/Meta-Llama-3-8B-Instruct is restricted
```

**해결책**:

1. https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct 에서 접근 권한 재요청
2. 토큰이 올바른지 확인
3. 승인 이메일 확인 (5-10분 소요)

### 모델 다운로드 느림

**정상**: Llama 3 8B는 약 4.7GB입니다. Colab의 네트워크 속도에 따라 5-10분 소요될 수 있습니다.

## 💡 팁

1. **Colab 세션 유지**

   - Colab은 90분 idle 시 세션이 끊어집니다
   - 긴 실행 중에는 가끔 노트북 화면을 클릭하세요

2. **중간 결과 저장**

   - `stage2_rewrites.txt`가 자동 생성됩니다
   - 마지막 셀로 로컬에 다운로드할 수 있습니다

3. **GPU 할당량**
   - 무료 Colab은 하루 GPU 사용량 제한이 있습니다
   - 제한에 걸리면 다음날 다시 시도하거나 Colab Pro 사용

## 📈 결과 해석

Stage 3 완료 후:

```
성공: 371/512
실패: 141/512
```

- **성공**: LLM이 안전하게 재작성하고 3단계 검증 통과
- **실패**: 재작성 불가능 또는 검증 실패
  - Phase 1: LLM이 `REWRITE_FAILED` 반환
  - Phase 2: 재작성 텍스트가 여전히 위험
  - Phase 3: 의미 유사도 < 0.85

## 🔄 로컬로 결과 가져오기

Colab에서 실행한 결과를 로컬로 가져오려면:

```python
# Colab 노트북 마지막 셀
from google.colab import files
files.download('stage2_rewrites.txt')
```

그런 다음 로컬에서:

```bash
# 다운로드한 파일을 프로젝트 폴더로 이동
mv ~/Downloads/stage2_rewrites.txt /Users/mlnyx/-prompt-firewall/
```

## 📞 추가 도움

문제가 지속되면:

1. 노트북의 각 셀 출력 확인
2. GPU가 할당되었는지 확인 (`!nvidia-smi`)
3. Hugging Face 토큰 유효성 확인
