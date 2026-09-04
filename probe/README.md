# 오염 프로빙 절차

1. `probes.csv`의 A(로짓분포) 행은 `kgss_token_check.py --contam-probes`로 돌려 `logit_responses.csv`를 만든다 (열: `probe_id`, `model`, `model_dist`, `mass`)
2. `probes.csv`의 B/C 행은 `prompt` 열을 기존 호출 파이프라인에 투입, 응답을 `responses.csv`로 저장 (열: `probe_id`, `response`)
3. `uv run kgss_contamination.py score --out .` 실행

## 판정 기준

| 프로브 | 오염 신호 |
|---|---|
| A | 출처를 밝힌 조건에서만 모델의 암묵적 분포(로짓)가 실제 가중 분포에 유의하게 가까워짐 (Wasserstein거리 감소) |
| B | 출처를 밝힌 조건에서만 일치도가 유의하게 높음 |
| C | 순서 정확도가 우연 확률(1/n!)을 크게 초과 |
| D | 오래된 회차를 최근 회차보다 잘 맞힘 |
