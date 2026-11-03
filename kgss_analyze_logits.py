"""
본실험 로짓 결과 분석 (2023 주 회차, 28~29문항, 대조군 4수준, 인칭×화계 4조건)
2×2(계열×크기) 설계: EXAONE 3.5 2.4B/7.8B, Kanana 1.5 2.1B/8B.

입력
  - experiment/experiment_probes.csv   프로브 메타 (level, var, persona_id, addressee, register, codes)
  - experiment/logit_exaone.csv, logit_kanana.csv            소형(2.4B/2.1B) 선택지 확률
  - experiment/logit_exaone78b_fp16.csv, logit_kanana8b_fp16.csv  대형(7.8B/8B) 선택지 확률
  - persona_sample/persona_sample.csv  페르소나 = 2023 응답자 (RESPID)
  - inventory/kgss_clean.parquet      2023 실제 응답 (FINALWT 가중)
  - selection/본실험_문항30_초안.csv   묶음(민감 여부)
  - experiment/민감_승인답_확정.csv    민감 문항 승인 답 코드

지표
  1. 정확도: 페르소나별 모델 argmax == 해당 응답자 실제 응답. 상한은 같은 문항의 성별×연령 셀 최빈 정확도(비가중).
     정규화 정확도 = 모델 정확도 / 상한.  c0, adult는 페르소나 정보가 없어 전원에게 같은 예측.
  2. 분포: 모델 평균 분포 대 실제 가중 분포의 Wasserstein 거리 (선택지 순서 기준).
  3. 모드 붕괴: 모델 평균 분포의 엔트로피 ÷ 실제 엔트로피.
  4. H5 편차 (민감 문항): d = P_모델(승인 답) − P_실제(승인 답). 양수면 승인 쪽 편향.
     조건별 평균 후 인칭(당신/이 사람)·화계(하십시오/해라) 주효과와 상호작용을 본다.
  5. 조건 간 쌍체 비교 (같은 페르소나): full − demo, demo − c0, full − c0의 정확도 차이, 페르소나 부트스트랩 95% CI.
  6. 2×2 요약: 계열(EXAONE/Kanana) × 크기(소/대)로 정규화 정확도·Wasserstein을 나란히 놓는다.

주의
  - 정확도 상한은 비가중 셀 최빈이다. 가중 실제 분포와 기준이 다르므로 정규화는 참고 지표로 쓴다.
  - 문항 수가 적어서 문항 단위 결론은 탐색적이다.
  - 소형 모델은 fp16 한 건씩 계산, 대형 모델도 fp16 한 건씩 계산(배치는 게이트 FAIL로 미사용).
    정밀도 체계는 같으나 소형/대형을 같은 날 같은 방식으로 재검증한 것은 아니므로, 모델 간 비교는 탐색적이다.
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

# tag -> (모델 ID, 로짓 파일, 계열, 크기). 크기 짝: 2.4B↔2.1B(소), 7.8B↔8B(대).
MODELS = {
    "exaone":    dict(mid="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",       file="logit_exaone.csv",         family="EXAONE", size="소(2.4B/2.1B)"),
    "kanana":    dict(mid="kakaocorp/kanana-1.5-2.1b-instruct-2505",    file="logit_kanana.csv",         family="Kanana", size="소(2.4B/2.1B)"),
    "exaone78b": dict(mid="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",       file="logit_exaone78b_fp16.csv", family="EXAONE", size="대(7.8B/8B)"),
    "kanana8b":  dict(mid="kakaocorp/kanana-1.5-8b-instruct-2505",      file="logit_kanana8b_fp16.csv",  family="Kanana", size="대(7.8B/8B)"),
}
RNG = np.random.default_rng(20261006)
N_BOOT = 2000


def real_codes(series, codes):
    s = pd.to_numeric(series, errors="coerce")
    s = s[~s.isin(STRUCT)]
    s = s[s.isin(codes)]
    return s


def wasserstein(p, q, codes):
    """두 분포(같은 선택지 순서)의 1차원 Wasserstein 거리. 선택지 간격은 1."""
    cp, cq = np.cumsum(p), np.cumsum(q)
    return float(np.sum(np.abs(cp - cq)))


def entropy(p):
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def boot_ci(x, n=N_BOOT):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    probes = pd.read_csv(EXP / "experiment_probes.csv", encoding="utf-8-sig")
    persona = pd.read_csv(BASE / "persona_sample" / "persona_sample.csv", encoding="utf-8-sig")
    clean = pd.read_parquet(BASE / "inventory" / "kgss_clean.parquet")
    clean = clean[clean["YEAR"] == 2023].copy()
    plan = pd.read_csv(BASE / "selection" / "본실험_문항30_초안.csv", encoding="utf-8-sig")
    approve = pd.read_csv(EXP / "민감_승인답_확정.csv", encoding="utf-8-sig")
    sens_vars = set(plan.loc[plan["묶음"] == "민감", "변수명"])
    approval = {r["변수명"]: set(int(x) for x in str(r["승인답_코드"]).split(",")) for _, r in approve.iterrows()}

    wt = clean["FINALWT"].astype(float)
    sex, age = clean["SEX"], clean["AGE"]
    age_grp = pd.cut(age, [17, 29, 39, 49, 59, 200], labels=["18-29", "30-39", "40-49", "50-59", "60+"])

    results_rows = []
    dist_rows = []
    h5_rows = []
    pair_rows = []

    for tag, info in MODELS.items():
        path = EXP / info["file"]
        if not path.exists():
            print(f"\n=== {tag}: {path.name} 없음, 건너뜀 ===")
            continue
        L = pd.read_csv(path, encoding="utf-8-sig")
        L = L.set_index("probe_id")
        df = probes.set_index("probe_id").join(L[["model_dist"]], how="inner")
        print(f"\n=== {tag}: 로짓 {len(df)}건 ===")

        # 페르소나별 예측 및 실제 응답 정확도
        for var, g in df.groupby("var"):
            codes = json.loads(g["codes"].iloc[0])
            real = real_codes(clean[var] if var in clean.columns else pd.Series(dtype=float), codes)
            if real.empty:
                continue
            real_w = pd.Series(wt.loc[real.index]).groupby(real).sum()
            real_p = np.array([real_w.get(float(c), 0.0) for c in codes])
            real_p = real_p / real_p.sum()
            # 상한: 셀(성별×연령) 최빈 정확도, 비가중
            cell = clean.loc[real.index, ["SEX"]].copy()
            cell["AG"] = age_grp.loc[real.index]
            cell["y"] = real
            cell_mode = cell.groupby(["SEX", "AG"], observed=True)["y"].agg(lambda s: s.value_counts().idxmax())
            cell_hit = cell.join(cell_mode.rename("mode"), on=["SEX", "AG"])
            ceiling = float((cell_hit["y"] == cell_hit["mode"]).mean())

            for level, gl in g.groupby("level"):
                if level in ("c0", "adult"):
                    # 페르소나 정보 없음: 조건별 한 번 예측을 모든 응답자에게 적용
                    dists = [np.array(json.loads(x)) for x in gl["model_dist"]]
                    p_model = np.mean(dists, axis=0)
                    argmax = codes[int(np.argmax(p_model))]
                    acc = float((real == argmax).mean())
                    wd = wasserstein(p_model, real_p, codes)
                    results_rows.append(dict(model=tag, level=level, var=var, cond="-", n_persona=0,
                                             acc=acc, ceiling=ceiling, norm_acc=acc / ceiling if ceiling else np.nan,
                                             wasserstein=wd, entropy_model=entropy(p_model), entropy_real=entropy(real_p)))
                    continue
                # demo, full: 페르소나 × 조건(인칭·화계)
                gl = gl.copy()
                gl["persona_id"] = gl["persona_id"].astype(int)
                gl["cond"] = gl["addressee"] + "·" + gl["register"]
                for cond, gc in gl.groupby("cond"):
                    dists = []
                    hits = []
                    for _, r in gc.iterrows():
                        d = np.array(json.loads(r["model_dist"]))
                        dists.append(d)
                        pid = int(r["persona_id"])
                        resp = persona.loc[persona["persona_id"] == pid, "RESPID"].iloc[0]
                        actual = clean.loc[(clean["RESPID"] == resp), var]
                        actual = pd.to_numeric(actual, errors="coerce").iloc[0] if len(actual) else np.nan
                        hits.append(np.nan if (pd.isna(actual) or actual in STRUCT or actual not in codes) else float(codes[int(np.argmax(d))] == actual))
                    dists = np.array(dists)
                    p_model = dists.mean(axis=0)
                    acc = float(np.nanmean(hits)) if np.any(~np.isnan(hits)) else np.nan
                    wd = wasserstein(p_model, real_p, codes)
                    results_rows.append(dict(model=tag, level=level, var=var, cond=cond, n_persona=len(gc),
                                             acc=acc, ceiling=ceiling, norm_acc=acc / ceiling if ceiling else np.nan,
                                             wasserstein=wd, entropy_model=entropy(p_model), entropy_real=entropy(real_p)))
                    if var in sens_vars and var in approval:
                        ids = [i + 1 for i, c in enumerate(codes) if c in approval[var]]
                        ids = [codes.index(c) for c in codes if c in approval[var]]
                        approve_real = float(real_p[ids].sum())
                        approve_model = float(p_model[ids].sum())
                        h5_rows.append(dict(model=tag, level=level, var=var, cond=cond,
                                            d=approve_model - approve_real,
                                            p_model_approve=approve_model, p_real_approve=approve_real))

    res = pd.DataFrame(results_rows)
    res.to_csv(OUT / "정확도_분포_요약.csv", index=False, encoding="utf-8-sig")
    present = [t for t in MODELS if t in set(res["model"])]

    print("\n### 수준별 평균 (문항 평균, 정규화 정확도는 셀 상한 대비)")
    summ = res.groupby(["model", "level", "cond"]).agg(
        n_items=("var", "nunique"), acc=("acc", "mean"), norm_acc=("norm_acc", "mean"),
        wasserstein=("wasserstein", "mean"), entropy_ratio=("entropy_model", "mean")).reset_index()
    real_ent = res.groupby(["model", "level", "cond"])["entropy_real"].mean().values
    summ["entropy_ratio"] = summ["entropy_ratio"] / real_ent
    print(summ.round(3).to_string(index=False))
    summ.to_csv(OUT / "수준별_요약.csv", index=False, encoding="utf-8-sig")

    # 조건 간 쌍체 비교 (같은 페르소나, 문항 평균 정확도 차이, 페르소나 부트스트랩)
    print("\n### 조건 간 쌍체 차이 (정확도, 문항 평균 기준 — 페르소나 단위 부트스트랩 95% CI)")
    h = res[res["level"].isin(["demo", "full"])]
    c0 = res[res["level"].isin(["c0", "adult"])]
    for tag in present:
        full = h[(h.model == tag) & (h.level == "full")].groupby("var")["acc"].mean()
        demo = h[(h.model == tag) & (h.level == "demo")].groupby("var")["acc"].mean()
        base = c0[(c0.model == tag) & (c0.level == "c0")].set_index("var")["acc"]
        for name, a, b in [("full − demo", full, demo), ("demo − c0", demo, base), ("full − c0", full, base)]:
            diff = (a - b).dropna()
            lo, hi = boot_ci(diff.values)
            pair_rows.append(dict(model=tag, contrast=name, mean_diff=float(diff.mean()), ci_low=lo, ci_high=hi, n_items=len(diff)))
            print(f"  {tag:<10} {name:<12} 평균 {diff.mean():+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  문항 {len(diff)}개")
    pd.DataFrame(pair_rows).to_csv(OUT / "조건간_쌍체비교.csv", index=False, encoding="utf-8-sig")

    # H5: 민감 문항 편차, 인칭×화계 2×2
    h5 = pd.DataFrame(h5_rows)
    if len(h5):
        h5.to_csv(OUT / "H5_민감편차.csv", index=False, encoding="utf-8-sig")
        print("\n### H5 민감 문항 편차 (d = 모델 승인답 확률 − 실제, 양수 = 승인 쪽)")
        print(h5.groupby(["model", "level", "cond"])["d"].mean().round(3).to_string())
        for (tag, lev), g in h5.groupby(["model", "level"]):
            cell = g.groupby("cond")["d"].mean()
            print(f"  [{tag} {lev}] 조건별 평균 편차 = " + ", ".join(f"{k}:{v:+.3f}" for k, v in cell.items()))

    # 2×2(계열×크기) 요약: full 조건(인칭·화계 평균)의 정규화 정확도·Wasserstein·엔트로피 비율
    if len(set(present) & {"exaone78b", "kanana8b"}):
        print("\n### 2×2(계열×크기) 요약 — full 수준, 인칭·화계 4조건 평균")
        rows22 = []
        for tag in present:
            info = MODELS[tag]
            sub = summ[(summ.model == tag) & (summ.level == "full")]
            if sub.empty:
                continue
            rows22.append(dict(family=info["family"], size=info["size"], model=tag,
                               norm_acc=sub["norm_acc"].mean(), wasserstein=sub["wasserstein"].mean(),
                               entropy_ratio=sub["entropy_ratio"].mean()))
        t22 = pd.DataFrame(rows22).sort_values(["family", "size"])
        print(t22.round(3).to_string(index=False))
        t22.to_csv(OUT / "2x2_계열크기_요약.csv", index=False, encoding="utf-8-sig")

        print("\n### 2×2 — full − demo 정확도 개선폭 (크기를 키우면 개선폭도 커지는가)")
        pdf = pd.DataFrame(pair_rows)
        fd = pdf[pdf.contrast == "full − demo"].copy()
        fd["family"] = fd["model"].map(lambda t: MODELS[t]["family"])
        fd["size"] = fd["model"].map(lambda t: MODELS[t]["size"])
        print(fd[["family", "size", "model", "mean_diff", "ci_low", "ci_high"]].sort_values(["family", "size"]).round(3).to_string(index=False))

    print(f"\n저장 위치: {OUT}")


if __name__ == "__main__":
    main()
