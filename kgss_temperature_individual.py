r"""
개별 기준 온도 보정 재평가 — 본실험 로짓 재사용, GPU 불필요

기존 kgss_temperature_sweep.py는 예측분포를 먼저 평균 낸 뒤(avg_pop) 온도를 골랐다.
kgss_target_alignment.py에서 avg_pop에는 집계 효과(평균 내면 분포가 매끈해짐)가
섞여 있다는 게 확인됐으므로, 여기서는 개별 예측 단위로 다시 평가한다.

지표 (모두 낮을수록 좋음)
    nll      페르소나의 원래 응답자(RESPID)가 실제로 고른 답의 음의 로그우도.
             개인 예측의 적정 채점규칙(proper scoring rule)이라 온도 보정의 표준 목적함수.
    brier    다범주 Brier 점수
    ind_pop  페르소나별 예측분포와 인구 분포의 Wasserstein 평균 (타겟정렬 스크립트와 동일)
    avg_pop  예측을 먼저 평균 낸 뒤 인구 분포와의 Wasserstein (기존 온도 스캔 방식)

    정확도(argmax)는 온도에 불변이라 T=1 값만 기록.

기준선 (페르소나 200명의 실제 응답에 대한 같은 지표)
    uniform   균등분포
    pop       2023 가중 인구분포 (페르소나 정보 없이 낼 수 있는 최선)
    cell      성별×연령 셀 가중분포 (demo 조건이 받는 정보의 상한)

T* 선택은 문항 단위 leave-one-out 교차검증(28문항으로 고르고 남은 1문항에 적용).

사용법:
    uv run kgss_temperature_individual.py
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
OUT.mkdir(exist_ok=True)
STRUCT = {-1.0, -8.0}
EPS = 1e-12
AGE_BINS = [17, 29, 39, 49, 59, 200]
AGE_LABELS = ["18-29", "30-39", "40-49", "50-59", "60+"]
LEVELS = ["c0", "adult", "demo", "full"]
TS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0,
      16.0, 24.0, 32.0, 64.0, 1000.0]
SCORES = ["nll", "brier", "ind_pop", "avg_pop"]
ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]

MODELS = {
    "exaone": "logit_exaone.csv", "kanana": "logit_kanana.csv",
    "exaone78b": "logit_exaone78b_fp16.csv", "kanana8b": "logit_kanana8b_fp16.csv",
}


def rescale(P, T):
    Q = np.power(np.clip(P, EPS, 1), 1.0 / T)
    return Q / Q.sum(axis=1, keepdims=True)


def w1_rows(P, q):
    return np.abs(np.cumsum(P, axis=1) - np.cumsum(q)).sum(axis=1)


def scores(P, y_idx, tgt):
    n, k = P.shape
    onehot = np.zeros((n, k))
    onehot[np.arange(n), y_idx] = 1
    return dict(
        nll=float(-np.log(np.clip(P[np.arange(n), y_idx], EPS, 1)).mean()),
        brier=float(((P - onehot) ** 2).sum(axis=1).mean()),
        ind_pop=float(w1_rows(P, tgt).mean()),
        avg_pop=float(w1_rows(P.mean(axis=0, keepdims=True), tgt)[0]),
    )


def wdist(sub, var, codes):
    s = pd.to_numeric(sub[var], errors="coerce")
    s = s[s.isin(codes) & ~s.isin(STRUCT)]
    w = sub.loc[s.index, "FINALWT"].astype(float).groupby(s).sum()
    p = np.array([w.get(float(c), 0.0) for c in codes])
    return p / p.sum() if p.sum() > 0 else None


def main():
    probes = pd.read_csv(EXP / "experiment_probes.csv", encoding="utf-8-sig")
    persona = pd.read_csv(BASE / "persona_sample" / "persona_sample.csv", encoding="utf-8-sig")
    clean = pd.read_parquet(BASE / "inventory" / "kgss_clean.parquet")
    clean = clean[clean["YEAR"] == 2023].copy()
    clean["AGE_GRP"] = pd.cut(clean["AGE"], AGE_BINS, labels=AGE_LABELS, right=False)
    persona["AGE_GRP"] = pd.cut(persona["AGE"], AGE_BINS, labels=AGE_LABELS, right=False)

    if clean["RESPID"].duplicated().any():
        raise SystemExit("2023 회차 안에서 RESPID 중복 — 매칭 불가")
    resp = clean.set_index("RESPID")
    persona = persona[persona["RESPID"].isin(resp.index)].copy()
    print(f"페르소나 {len(persona)}명 (RESPID 매칭)")

    item_meta = {}
    for var in probes["var"].unique():
        if var not in clean.columns:
            continue
        codes = json.loads(probes.loc[probes["var"] == var, "codes"].iloc[0])
        tgt = wdist(clean, var, codes)
        if tgt is None:
            continue
        ans = pd.to_numeric(resp.loc[persona["RESPID"], var], errors="coerce").to_numpy()
        valid = np.isin(ans, codes)
        y_idx = np.array([codes.index(int(a)) if v else -1 for a, v in zip(ans, valid)])
        cell_p = {}
        for (sx, ag), sub in clean.groupby(["SEX", "AGE_GRP"], observed=True):
            cell_p[(sx, ag)] = wdist(sub, var, codes)
        item_meta[var] = dict(codes=codes, tgt=tgt, valid=valid, y_idx=y_idx, cell_p=cell_p)

    pids = persona["persona_id"].to_numpy()
    pid_pos = {p: i for i, p in enumerate(pids)}
    cell_keys = list(zip(persona["SEX"], persona["AGE_GRP"]))

    rows, base_rows, acc_rows, shift_rows = [], [], [], []

    for var, m in item_meta.items():
        v = m["valid"]
        k = len(m["codes"])
        y = m["y_idx"][v]
        n = int(v.sum())
        base = {
            "uniform": np.full((n, k), 1.0 / k),
            "pop": np.tile(m["tgt"], (n, 1)),
            "cell": np.array([m["cell_p"][ck] for ck, ok in zip(cell_keys, v) if ok]),
        }
        for name, P in base.items():
            base_rows.append(dict(var=var, baseline=name, n=n, **scores(P, y, m["tgt"])))

    for tag, fname in MODELS.items():
        path = EXP / fname
        if not path.exists():
            print(f"=== {tag}: {fname} 없음, 건너뜀 ===")
            continue
        L = pd.read_csv(path, encoding="utf-8-sig").set_index("probe_id")
        df = probes.set_index("probe_id").join(L[["model_dist"]], how="inner")
        print(f"=== {tag}: 로짓 {len(df)}건 ===", flush=True)

        adult_ref = {}
        for _, r in df[df["level"] == "adult"].iterrows():
            adult_ref[(r["var"], f"{r['addressee']}·{r['register']}")] = np.array(json.loads(r["model_dist"]))

        for (level, var), g in df.groupby(["level", "var"]):
            if var not in item_meta:
                continue
            m = item_meta[var]
            v = m["valid"]
            conds = g["addressee"].fillna("") + "·" + g["register"].fillna("")
            for cond, gc in g.groupby(conds):
                dists = np.array([json.loads(x) for x in gc["model_dist"]])
                if level in ("c0", "adult"):
                    P_all = np.tile(dists[0], (len(pids), 1))
                else:
                    P_all = np.full((len(pids), dists.shape[1]), np.nan)
                    for pid, d in zip(gc["persona_id"].astype(int), dists):
                        if pid in pid_pos:
                            P_all[pid_pos[pid]] = d
                keep = v & ~np.isnan(P_all).any(axis=1)
                P = P_all[keep]
                y = m["y_idx"][keep]
                if len(P) == 0:
                    continue
                acc_rows.append(dict(model=tag, level=level, var=var, cond=cond,
                                     acc=float((P.argmax(axis=1) == y).mean()), n=len(P)))
                for T in TS:
                    rows.append(dict(model=tag, level=level, var=var, cond=cond, T=T,
                                     **scores(rescale(P, T), y, m["tgt"])))
                # 위치 보정: 인구분포 x (페르소나 예측 / 같은 인칭·화계의 adult 예측)^alpha
                # alpha=0이면 pop 기준선과 동일. 0보다 큰 alpha가 나으면 페르소나 조건화가
                # 인구분포 이상의 개인 정보를 담고 있다는 뜻.
                ref = adult_ref.get((var, cond))
                if level in ("demo", "full") and ref is not None:
                    ratio = np.clip(P, EPS, 1) / np.clip(ref, EPS, 1)
                    for a in ALPHAS:
                        Q = m["tgt"] * np.power(ratio, a)
                        Q = Q / Q.sum(axis=1, keepdims=True)
                        Q = np.clip(Q, EPS, 1)
                        shift_rows.append(dict(model=tag, level=level, var=var, cond=cond, alpha=a,
                                               nll=float(-np.log(Q[np.arange(len(y)), y]).mean()),
                                               acc=float((Q.argmax(axis=1) == y).mean())))

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "온도_개별_상세.csv", index=False, encoding="utf-8-sig")
    base = pd.DataFrame(base_rows)
    acc = pd.DataFrame(acc_rows)

    item = res.groupby(["model", "level", "var", "T"])[SCORES].mean().reset_index()
    curve = item.groupby(["model", "level", "T"])[SCORES].mean().reset_index()
    curve.to_csv(OUT / "온도_개별_곡선.csv", index=False, encoding="utf-8-sig")

    # 문항 leave-one-out으로 지표별 T* 선택
    cv_rows = []
    for (tag, level), g in item.groupby(["model", "level"]):
        items = sorted(g["var"].unique())
        for score in SCORES:
            piv = g.pivot(index="var", columns="T", values=score).loc[items]
            oof, chosen = [], []
            for it in items:
                train = piv.drop(index=it).mean(axis=0)
                t_star = float(train.idxmin())
                chosen.append(t_star)
                oof.append(piv.at[it, t_star])
            cv_rows.append(dict(model=tag, level=level, score=score,
                                T1=float(piv[1.0].mean()), oof_Tstar=float(np.mean(oof)),
                                in_sample_best=float(piv.mean(axis=0).min()),
                                T_star_median=float(np.median(chosen)),
                                T_star_range=f"{min(chosen):g}-{max(chosen):g}"))
    cv = pd.DataFrame(cv_rows)
    cv["개선"] = cv["T1"] - cv["oof_Tstar"]
    cv["level"] = pd.Categorical(cv["level"], LEVELS, ordered=True)
    cv = cv.sort_values(["score", "model", "level"])
    cv.to_csv(OUT / "온도_개별_교차검증.csv", index=False, encoding="utf-8-sig")

    bsum = base.groupby("baseline")[SCORES].mean()
    bsum.to_csv(OUT / "온도_개별_기준선.csv", encoding="utf-8-sig")
    asum = acc.groupby(["model", "level", "var"])["acc"].mean().groupby(["model", "level"]).mean()

    print("\n### 기준선 (페르소나 실제 응답 기준, 문항 평균)")
    print(bsum.round(3).to_string())

    for score in ["nll", "brier", "ind_pop", "avg_pop"]:
        print(f"\n### [{score}] T=1 vs LOO 교차검증 T* (문항 평균)")
        sub = cv[cv.score == score].drop(columns="score")
        print(sub.round(3).to_string(index=False))

    print("\n### 정확도 (argmax, 온도 불변) — 기준: pop 최빈 / cell 최빈")
    print(asum.unstack()[LEVELS].round(3).to_string())
    bacc = {}
    for name in ["pop", "cell"]:
        vals = []
        for var, m in item_meta.items():
            v = m["valid"]
            y = m["y_idx"][v]
            if name == "pop":
                pred = np.full(len(y), int(np.argmax(m["tgt"])))
            else:
                pred = np.array([int(np.argmax(m["cell_p"][ck])) for ck, ok in zip(cell_keys, v) if ok])
            vals.append(float((pred == y).mean()))
        bacc[name] = float(np.mean(vals))
    print(f"  pop 최빈 {bacc['pop']:.3f} / cell 최빈 {bacc['cell']:.3f}")

    sh = pd.DataFrame(shift_rows)
    if len(sh):
        sh_item = sh.groupby(["model", "level", "var", "alpha"])[["nll", "acc"]].mean().reset_index()
        sh_rows = []
        for (tag, level), g in sh_item.groupby(["model", "level"]):
            piv = g.pivot(index="var", columns="alpha", values="nll")
            pacc = g.pivot(index="var", columns="alpha", values="acc")
            oof, oacc, chosen = [], [], []
            for it in piv.index:
                a_star = float(piv.drop(index=it).mean(axis=0).idxmin())
                chosen.append(a_star)
                oof.append(piv.at[it, a_star])
                oacc.append(pacc.at[it, a_star])
            sh_rows.append(dict(model=tag, level=level, nll_pop=float(piv[0.0].mean()),
                                nll_oof=float(np.mean(oof)), 개선=float(piv[0.0].mean() - np.mean(oof)),
                                alpha_median=float(np.median(chosen)),
                                acc_pop=float(pacc[0.0].mean()), acc_oof=float(np.mean(oacc))))
        shs = pd.DataFrame(sh_rows)
        shs.to_csv(OUT / "위치보정_교차검증.csv", index=False, encoding="utf-8-sig")
        print("\n### 위치 보정: 인구분포 x (페르소나/adult)^alpha, 문항 LOO로 alpha 선택")
        print("    개선 > 0 이면 페르소나 조건화가 인구분포 이상의 개인 정보를 담고 있음")
        print(f"    참고: cell 기준선 nll {bsum.at['cell', 'nll']:.3f} (성별x연령 실제 분포)")
        print(shs.round(3).to_string(index=False))

    print(f"\n저장 위치: {OUT}")


if __name__ == "__main__":
    main()
