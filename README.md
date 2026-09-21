# 한국형 실리콘 샘플링 검증

LLM에 인구통계 페르소나를 프롬프트로 부여해 한국 여론조사 문항(KGSS)에
응답시키고, 가상 응답이 실제 응답을 얼마나 재현하는지 정량 분석한다.

## 연구 설계

관측되는 건 "LLM 응답 분포가 실제와 다르다"는 사실 하나지만 원인은 여러
개일 수 있다. 아래 네 성분으로 분해해서 각각 측정·개입한다.

| 성분 | 측정 | 개입 |
|---|---|---|
| ① 정보 부족 | 셀 최빈응답 상한 대비 개인예측 정확도 | 응답 이력 조건 추가 |
| ② 척도 표현 | 출력 엔트로피, 셀 내 분산 재현율 | SSR, 디코딩 온도 |
| ③ 내용 편향 | 민감 문항 방향성 편차 | **인칭 × 화계 변형 (헤드라인)** |
| ④ 형식 민감도 | TV거리, 인간 기준선 대비 배율 | 정순/역순 교차 |

**대조군 4수준**: 페르소나 없음(C0) / "한국 성인" / 성별+연령 / 6변수
(SEX·AGE·EDUC·MARITAL·REGION·URBAN). 같은 응답자 표본을 4수준 전부에
재사용해 조건 간 대응비교가 가능하게 한다.

③(인칭×화계 조작)이 이 프로젝트의 고유 기여다 — 한국어 화계 체계는
영어권 선행연구에서 시도할 수 없는 통제된 조작을 허용한다.

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

# 4. 문항 선정 작업지 생성 (3인 독립 태깅용 xlsx)
uv run kgss_shortlist.py --inv .\inventory --out .\selection --year 2023
#   -> 각자 자기 시트에 민감여부 태깅 후 kgss_agreement.py로 합치기
uv run kgss_agreement.py --out .\selection --files `
    주희=selection\문항선정_작업지_2023_정.csv `
    태우=selection\문항선정_작업지_2023_태우라벨링.xlsx `
    다교=selection\문항선정_작업지_2023_HUR.xlsx

# 5. 선택지 토큰화 점검 (로컬은 토크나이저만, 로짓은 Colab/Kaggle에서)
uv run kgss_token_check.py --tokenizer-only --worded .\selection\variables_worded.csv

# 6. 사전학습 오염 프로브: 생성(로컬) -> 로짓/생성 실행(Colab) -> 채점(로컬)
uv run kgss_contamination.py build --inv .\inventory `
    --worded .\selection\variables_worded.csv --out .\probe
#   Colab에서 kgss_contam_colab.ipynb 실행 -> probe/logit_responses.csv, responses.csv 다운로드 후 여기 probe/ 에 넣기
uv run kgss_contamination.py score --out .\probe

# 7. 6변수 페르소나 표본 생성 (FINALWT 가중 복원추출, 대조군 "6변수" 조건용)
uv run kgss_persona_bootstrap.py --inv .\inventory --out .\persona_sample --n 200 --seed 42

# 8. 본실험 프롬프트 생성 (최종 문항 확정 후) + Colab에서 로짓 추출
uv run kgss_experiment_prompt.py --items .\selection\최종문항.csv --out .\experiment
uv run kgss_token_check.py --contam-probes .\experiment\experiment_probes.csv --out .\experiment
```

## 주요 산출물

| 파일 | 내용 |
|---|---|
| `inventory/kgss_clean.parquet` | 결측 정리 완료 데이터. 이후 모든 분석의 입력 |
| `inventory/diagnostics.md` | 데이터 진단 요약 |
| `selection/variables_worded.csv` | 전 변수 설문 원문 + 선택지. 프롬프트 원천 |
| `selection/ab_pairs.csv` | A/B 분할표본과 인간 응답 순서 효과 |
| `selection/문항선정_작업지_2023.xlsx` | 3인 태깅용 문항 선정 작업지 (사실/차별/가치관, 민감여부) |
| `selection/agreement_result.md` | 3인 태깅 일치도(Cohen's/Fleiss' kappa) |
| `selection/문항선정_병합_2023.csv` | 3인 태깅 다수결 병합본 — 최종 문항 뽑을 때 기준 |
| `token_check/*.md` | 모델별 선택지 토큰화 · 첫 토큰 로짓 점검 결과 |
| `probe/README.md` | 오염 프로빙 절차 |
| `probe/contamination_result.md` | 오염 프로브 채점 결과 (A/B/C/E, 모델별) |
| `persona_sample/persona_sample.csv` | FINALWT 가중 복원추출 6변수 페르소나 표본 |

## 진행 상황

- **문항 선정**: 후보 303개(2023년 회차) 중 사실/차별/가치관 자동 분류 완료,
  3인 독립 태깅으로 민감여부 확정 중. 최종 30개(유형별 10개, 배터리당 1개)
  선정이 다음 단계.
- **사전학습 오염 점검**: 로짓 기반(A)·원문완성(B)·선택지순서(C)·직접회상(E)
  4가지 프로브 + 가짜출처 대조군으로 n=140 재실행 완료. EXAONE 3.5의 집계
  응답 비율에 대한 약한 신호(A, 다중비교 미보정 기준)만 남고 나머지는 뚜렷한
  오염 증거 없음. 자세한 수치는 `probe/contamination_result.md` 참고.
- **본실험**: 프롬프트 생성 파이프라인(`kgss_experiment_prompt.py`)까지 준비
  완료, 최종 문항 확정 후 실행 예정.
