# 한국형 실리콘 샘플링 검증

LLM에 인구통계 페르소나를 프롬프트로 부여해 한국 여론조사 문항(KGSS)에
응답시키고, 가상 응답이 실제 응답을 얼마나 재현하는지 정량 분석한다.

## 데이터

원본 파일은 저장소에 포함되지 않는다 — 라이선스 제약이 아니라 **용량과
재생성 가능성** 때문이다. 성균관대학교 서베이리서치센터(SRC)에서 별도
신청 절차 없이 받을 수 있다. 다운로드 후 프로젝트 루트에 둘 것.

- `2003-2025_KGSS_kor_public_v2.sav`
- `2003-2025_KGSS_Codebook_v5.pdf`

## 환경 구성

```powershell
uv venv --python 3.12
uv pip install -r requirements.txt
```

오픈웨이트 모델(EXAONE 3.5, Kanana 1.5)을 다루는 스크립트는 GPU가 필요해
로컬이 아니라 Colab/Kaggle에서 실행한다 — `kgss_contam_colab.ipynb` 참고.

## 실행 순서

```powershell
# 1. 데이터 구조 진단 (결측 분해, eta2, 셀 크기, 문항 후보 선정)
uv run kgss_inventory.py --sav "2003-2025_KGSS_kor_public_v2.sav" `
    --out .\inventory --target-year 2023

# 2. 코드북에서 설문 원문 추출 + A/B 분할표본 순서효과 분석
uv run kgss_codebook.py --pdf "2003-2025_KGSS_Codebook_v5.pdf" `
    --inv .\inventory --out .\selection

# 3. (선택) 사람이 읽는 코드북/회차 데이터 변환
uv run kgss_export.py --sav "2003-2025_KGSS_kor_public_v2.sav" --year 2023

# 4. 선택지 토큰화 점검 (로컬은 토크나이저만, 로짓은 Colab/Kaggle에서)
uv run kgss_token_check.py --tokenizer-only --worded .\selection\variables_worded.csv

# 5. 사전학습 오염 프로브: 생성(로컬) -> 로짓/생성 실행(Colab) -> 채점(로컬)
uv run kgss_contamination.py build --inv .\inventory `
    --worded .\selection\variables_worded.csv --out .\probe
#   Colab에서 kgss_contam_colab.ipynb 실행 -> probe/logit_responses.csv, responses.csv 다운로드 후 여기 probe/ 에 넣기
uv run kgss_contamination.py score --out .\probe
```

## 주요 산출물

| 파일 | 내용 |
|---|---|
| `inventory/kgss_clean.parquet` | 결측 정리 완료 데이터. 이후 모든 분석의 입력 |
| `inventory/diagnostics.md` | 데이터 진단 요약 |
| `selection/variables_worded.csv` | 전 변수 설문 원문 + 선택지. 프롬프트 원천 |
| `selection/ab_pairs.csv` | A/B 분할표본과 인간 응답 순서 효과 |
| `token_check/*.md` | 모델별 선택지 토큰화 · 첫 토큰 로짓 점검 결과 |
| `probe/README.md` | 오염 프로빙 절차 |
| `probe/contamination_result.md` | 오염 프로브 채점 결과 (A/B/C, 모델별) |

## 사전학습 오염 점검

주 회차(2023) 자료가 EXAONE 3.5(2.4B) / Kanana 1.5(2.1B)의 사전학습에
포함됐는지 세 가지 방식으로 점검한다 — 모두 출처 명시/미명시 대조를 통제로
쓴다.

- **A(로짓 기반 분포 재현)**: 문항에 강제로 번호만 답하게 한 뒤 선택지
  숫자 토큰의 첫 토큰 로짓을 뽑아, 모델의 암묵적 분포를 실제 가중 응답
  분포와 Wasserstein거리로 비교
- **B(원문 완성)**: 설문 문장 앞부분을 주고 나머지를 잇게 함 —
  오염 탐지 표준 기법(Golchin & Surdeanu 2023)
- **C(선택지 순서 재현)**: 섞어놓은 선택지를 설문지 원래 순서로 재배열

**2026-09-04 1차 결과**: 모델 2개 × 프로브 3개, 총 6개 검정 중 유의(p<0.05)한
신호 없음. 가장 근접한 값은 B/EXAONE p=0.059(경계, 다중비교 미보정). 2023년
회차에 대해 일관된 오염 증거는 없다는 잠정 결론이며, 최종 판단은 팀 논의로
확정한다. 자세한 수치는 `probe/contamination_result.md` 참고.
