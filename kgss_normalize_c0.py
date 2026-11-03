r"""
운영진 Q3 대응: 모델 정확도를 문항별 상한으로 정규화 (C0 조건)

정의
  - 실제 분포: KGSS 2023, 해당 문항에 실질 응답한 사람(-1, -8 제외)의 선택지별 비율
  - 상한(인간 최빈): 실제 응답의 최빈 선택지가 차지하는 비율 max_c P(c)
  - 모델 C0 예측: 모델 분포의 최빈 선택지 c*
  - 모델 C0 정확도: P(c*)  -- 페르소나가 없어 모든 응답자에게 같은 예측을 한다고 보고,
    그 예측이 실제 응답자와 일치하는 비율의 기댓값
  - 정규화 정확도: P(c*) / max_c P(c)   (1.0 = 상한과 같음)

주의
  - C0는 개인 정보가 없어 셀별 예측이 같다. 따라서 이 값은 "모델이 가장 흔한 답을
    골랐는가"를 재는 것이고, 조건화 능력은 재지 않는다.
  - 개인 수준 정규화(셀 최빈 상한 대비)는 demo/full 실행 후에만 가능하다.
  - 가중치는 쓰지 않는다. inventory/modal_baseline.csv의 상한(0.5385)과 같은 기준.

입력
  - c0_check/probes.csv          (probe_id, var, codes)
  - c0_check/logit_responses.csv (probe_id, model, model_dist)
  - inventory/kgss_clean.parquet

출력
  - c0_check/normalized_c0.csv   문항별 결과
  - 콘솔: 모델별 요약
"""

import io
import json
import sys
from pathlib import Path

import pandas as pd

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent
PROBES = BASE / "c0_check" / "probes.csv"
LOGITS = BASE / "c0_check" / "logit_responses.csv"
CLEAN = BASE / "inventory" / "kgss_clean.parquet"
OUT = BASE / "c0_check" / "normalized_c0.csv"
STRUCTURAL = {-1.0, -8.0}


def real_dist(d2023: pd.DataFrame, var: str, codes: list[int]) -> dict[int, float] | None:
    """선택지 코드별 실제 비율. 선택지 밖 코드와 -1/-8은 분모에서 뺀다."""
    if var not in d2023.columns:
        return None
    s = pd.to_numeric(d2023[var], errors="coerce").dropna()
    s = s[~s.isin(STRUCTURAL)]
    s = s[s.isin(codes)]
    if len(s) == 0:
        return None
    vc = s.value_counts(normalize=True)
    return {c: float(vc.get(float(c), 0.0)) for c in codes}


def main():
    probes = pd.read_csv(PROBES, encoding="utf-8-sig")
    logits = pd.read_csv(LOGITS, encoding="utf-8-sig")
    d = pd.read_parquet(CLEAN, columns=["YEAR"] + [c for c in probes["var"].unique()])
    d2023 = d[d["YEAR"] == 2023]

    meta = probes.set_index("probe_id")
    rows = []
    for _, r in logits.iterrows():
        pid = r["probe_id"]
        if pid not in meta.index:
            continue
        var = meta.at[pid, "var"]
        codes = json.loads(meta.at[pid, "codes"])
        dist = json.loads(r["model_dist"])
        if len(dist) != len(codes):
            continue
        real = real_dist(d2023, var, codes)
        if real is None:
            continue
        top_real = max(real.values())
        if top_real <= 0:
            continue
        k = max(range(len(dist)), key=lambda i: dist[i])
        c_star = codes[k]
        p_hit = real[c_star]
        rows.append({
            "model": r["model"], "var": var, "model_argmax": c_star,
            "model_top_p": dist[k], "real_mode": max(real, key=real.get),
            "real_top_p": top_real, "p_real_at_model_argmax": p_hit,
            "normalized": p_hit / top_real,
            "model_argmax_is_real_mode": c_star == max(real, key=real.get),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT} ({len(out)}행)")
    for m, g in out.groupby("model"):
        print(f"\n{m}")
        print(f"  문항 {len(g)}개")
        print(f"  모델 최빈 = 실제 최빈인 문항 비율: {g['model_argmax_is_real_mode'].mean():.3f}")
        print(f"  평균 정확도(P_real(c*)): {g['p_real_at_model_argmax'].mean():.4f}")
        print(f"  평균 상한(max P_real): {g['real_top_p'].mean():.4f}")
        print(f"  정규화 정확도 평균: {g['normalized'].mean():.4f}  (중앙값 {g['normalized'].median():.4f})")


if __name__ == "__main__":
    main()
