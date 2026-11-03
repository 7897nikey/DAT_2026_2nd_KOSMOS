"""모드 붕괴(성분②) 대응: 디코딩 온도 재정규화 탐색. 네 모델 × 전 수준으로 확장.

이미 계산해 둔 로짓(model_dist = 선택지 확률)만 쓴다. 추가 GPU 연산이 필요 없다.
원리: 제한된 선택지 집합에서 renormalize된 확률 p_i ∝ exp(z_i)이므로,
      p_i^(1/T)를 다시 정규화하면 소프트맥스(z/T)와 수학적으로 동일하다.
      argmax(최빈 선택, 즉 개인 예측 정확도)는 T에 영향받지 않는다 — 온도는 분포의
      '폭'만 바꾸고 순위는 안 바꾸기 때문이다. 그래서 온도 보정은 정확도를 해치지 않는다.

출력: experiment/분석결과/온도_탐색.csv
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent
EXP = BASE / "experiment"
OUT = EXP / "분석결과"
STRUCT = {-1.0, -8.0}
MODELS = {
    "exaone": "logit_exaone.csv", "kanana": "logit_kanana.csv",
    "exaone78b": "logit_exaone78b_fp16.csv", "kanana8b": "logit_kanana8b_fp16.csv",
}
TS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]


def wasserstein(p, q):
    return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))))


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def rescale(p, T):
    """p^(1/T)를 다시 정규화. T>1이면 분포를 넓히고(완화), T<1이면 좁힌다(더 확신)."""
    q = np.power(np.clip(p, 1e-12, 1), 1.0 / T)
    return q / q.sum()


def real_codes(series, codes):
    s = pd.to_numeric(series, errors="coerce")
    s = s[~s.isin(STRUCT)]
    return s[s.isin(codes)]


def main():
    probes = pd.read_csv(EXP / "experiment_probes.csv", encoding="utf-8-sig")
    clean = pd.read_parquet(BASE / "inventory" / "kgss_clean.parquet")
    clean = clean[clean["YEAR"] == 2023].copy()
    wt = clean["FINALWT"].astype(float)

    rows = []
    for tag, fname in MODELS.items():
        path = EXP / fname
        if not path.exists():
            continue
        L = pd.read_csv(path, encoding="utf-8-sig").set_index("probe_id")
        df = probes.set_index("probe_id").join(L[["model_dist"]], how="inner")
        print(f"=== {tag} ===", flush=True)

        for level, gl in df.groupby("level"):
            for var, gv in gl.groupby("var"):
                codes = json.loads(gv["codes"].iloc[0])
                real = real_codes(clean[var] if var in clean.columns else pd.Series(dtype=float), codes)
                if real.empty:
                    continue
                real_w = pd.Series(wt.loc[real.index]).groupby(real).sum()
                real_p = np.array([real_w.get(float(c), 0.0) for c in codes])
                real_p = real_p / real_p.sum()
                uniform = np.ones(len(codes)) / len(codes)
                wd_uniform = wasserstein(uniform, real_p)

                dists = np.array([json.loads(x) for x in gv["model_dist"]])
                for T in TS:
                    scaled = np.array([rescale(p, T) for p in dists])
                    p_model = scaled.mean(axis=0)
                    rows.append(dict(model=tag, level=level, var=var, T=T,
                                     wasserstein=wasserstein(p_model, real_p),
                                     wd_uniform=wd_uniform, entropy_ratio=entropy(p_model) / entropy(real_p)))

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "온도_탐색.csv", index=False, encoding="utf-8-sig")

    # 모델×수준 평균(문항 평균) 기준으로 최적 T 찾기
    agg = res.groupby(["model", "level", "T"]).agg(
        wasserstein=("wasserstein", "mean"), wd_uniform=("wd_uniform", "mean"),
        entropy_ratio=("entropy_ratio", "mean")).reset_index()

    print("\n### 모델×수준별 최적 온도 (문항 평균 Wasserstein 최소 지점)")
    print(f"{'model':<10}{'level':<7}{'T*':>6}{'WD(T*)':>9}{'균등기준선':>11}{'역전':>8}{'엔트로피비율(T*)':>16}")
    for (tag, lev), g in agg.groupby(["model", "level"]):
        best = g.loc[g["wasserstein"].idxmin()]
        reversed_ = "O" if best["wasserstein"] < best["wd_uniform"] else "X"
        print(f"{tag:<10}{lev:<7}{best['T']:>6.2f}{best['wasserstein']:>9.3f}{best['wd_uniform']:>11.3f}{reversed_:>8}{best['entropy_ratio']:>16.3f}")

    print(f"\n저장: {OUT / '온도_탐색.csv'}")


if __name__ == "__main__":
    main()
