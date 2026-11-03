"""온도 보정의 5-fold 문항 교차검증 (Out-of-Fold).

kgss_temperature_sweep.py가 저장한 experiment/분석결과/온도_탐색.csv를 그대로 쓴다.
29개 문항을 5묶음으로 나눠, 각 묶음을 평가용으로 남기고 나머지 문항에서 고른 T*를
그 평가용 묶음에 적용한다. "같은 데이터로 튜닝+평가"하는 문제를 피하기 위함이다.

출력: experiment/분석결과/온도_교차검증.csv
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent
EXP = BASE / "experiment"
OUT = EXP / "분석결과"
N_FOLDS = 5
SEED = 20261013


def main():
    res = pd.read_csv(OUT / "온도_탐색.csv", encoding="utf-8-sig")
    rows = []

    for (tag, level), g in res.groupby(["model", "level"]):
        items = sorted(g["var"].unique())
        rng = np.random.default_rng(SEED)
        order = rng.permutation(len(items))
        folds = np.array_split(order, N_FOLDS)

        oof_T1 = []       # T=1(보정 없음) 기준 Wasserstein
        oof_Tstar = []    # OOF로 고른 T* 적용 Wasserstein
        oof_uniform = []
        chosen_Ts = []

        for k in range(N_FOLDS):
            test_idx = folds[k]
            test_items = [items[i] for i in test_idx]
            train_items = [items[i] for i in range(len(items)) if i not in test_idx]

            train = g[g["var"].isin(train_items)]
            # 학습 묶음에서 Wasserstein 평균이 최소인 T를 고른다
            by_T = train.groupby("T")["wasserstein"].mean()
            T_star = float(by_T.idxmin())
            chosen_Ts.append(T_star)

            test = g[g["var"].isin(test_items)]
            oof_T1.append(test.loc[test["T"] == 1.0, "wasserstein"].mean())
            oof_Tstar.append(test.loc[test["T"] == T_star, "wasserstein"].mean())
            oof_uniform.append(test["wd_uniform"].mean())

        rows.append(dict(model=tag, level=level, n_items=len(items),
                         T_mean=np.mean(chosen_Ts), T_min=min(chosen_Ts), T_max=max(chosen_Ts),
                         wd_raw=np.mean(oof_T1), wd_oof=np.mean(oof_Tstar), wd_uniform=np.mean(oof_uniform)))

    cv = pd.DataFrame(rows)
    cv["역전_보정전"] = cv["wd_raw"] < cv["wd_uniform"]
    cv["역전_OOF보정후"] = cv["wd_oof"] < cv["wd_uniform"]
    cv.to_csv(OUT / "온도_교차검증.csv", index=False, encoding="utf-8-sig")

    print("### 5-fold 문항 교차검증: 온도 보정이 새 문항에도 일반화되는가")
    print(f"{'model':<10}{'level':<7}{'T*(범위)':>14}{'보정전WD':>10}{'OOF WD':>9}{'균등기준선':>11}{'전:역전':>9}{'OOF:역전':>10}")
    for _, r in cv.iterrows():
        trange = f"{r['T_min']:.0f}~{r['T_max']:.0f}"
        print(f"{r['model']:<10}{r['level']:<7}{trange:>14}{r['wd_raw']:>10.3f}{r['wd_oof']:>9.3f}"
              f"{r['wd_uniform']:>11.3f}{str(r['역전_보정전']):>9}{str(r['역전_OOF보정후']):>10}")

    n_ok = cv["역전_OOF보정후"].sum()
    print(f"\nOOF로도 균등 기준선을 역전한 칸: {n_ok} / {len(cv)}")
    print(f"저장: {OUT / '온도_교차검증.csv'}")


if __name__ == "__main__":
    main()
