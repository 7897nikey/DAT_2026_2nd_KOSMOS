r"""
인칭 x 화계 조작 분석 — 본실험 로짓 재사용, GPU 불필요 (탐색적)

1. 조작 점검: 화계·인칭을 바꾸면 출력이 바뀌기는 하는가
   같은 페르소나·문항·수준에서 하십시오체 vs 해라체(인칭 고정), 당신 vs 이사람(화계 고정)
   예측분포 사이 TV거리와 최빈선택 뒤집힘 비율. 비교 척도:
     - 인간 A/B 순서효과 TV 0.02-0.05 (CLAUDE.md 4절)
     - 페르소나 간 TV: 같은 문항·조건에서 무작위로 짝지은 두 페르소나 사이 거리
       (모델이 페르소나 차이에 반응하는 크기)

2. 혼합효과모형
   (a) 민감 8문항 바람직성 치우침 ds의 화계·인칭 효과.
       ds ~ 해라 * 이사람 + level, 문항 무선 절편+기울기(해라, 이사람), 페르소나 분산성분.
       문항이 8개라 고정효과 추론은 결국 문항 수에 묶인다.
   (b) 이중차분: 29문항 전체에서 화계 효과가 민감 문항에서 다른가.
       방향이 정의되지 않는 비민감 문항도 쓰기 위해 종속변수를 "인구분포와의 Wasserstein
       변화"로 둔다. dW = W(해라) - W(하십시오), 음수면 해라체가 실제에 더 가까움.
       dW ~ 민감, 문항 무선 절편. 인칭도 같은 방식.

ds의 바람직한 방향과 MEDIAPOL DK 처리는 kgss_item_direction.py와 동일.

사용법:
    uv run kgss_register_analysis.py
"""

import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from kgss_item_direction import DESIRABLE, keep_codes

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
EXP = BASE / "experiment"
OUT = EXP / "분석결과"
OUT.mkdir(exist_ok=True)
STRUCT = {-1.0, -8.0}
MODELS = {
    "exaone": "logit_exaone.csv", "kanana": "logit_kanana.csv",
    "exaone78b": "logit_exaone78b_fp16.csv", "kanana8b": "logit_kanana8b_fp16.csv",
}
SEED = 20261102


def tv(P, Q):
    return 0.5 * np.abs(P - Q).sum(axis=1)


def w1(P, q):
    return np.abs(np.cumsum(P, axis=1) - np.cumsum(q)).sum(axis=1)


def main():
    probes = pd.read_csv(EXP / "experiment_probes.csv", encoding="utf-8-sig")
    probes = probes[probes["level"].isin(["adult", "demo", "full"])]
    items = pd.read_csv(BASE / "selection" / "본실험_문항30_초안.csv", encoding="utf-8-sig")
    sensitive = set(items.loc[items["묶음"] == "민감", "변수명"])
    vt = pd.read_csv(BASE / "inventory" / "variables.csv").set_index("var")
    clean = pd.read_parquet(BASE / "inventory" / "kgss_clean.parquet")
    clean = clean[clean["YEAR"] == 2023]
    persona = pd.read_csv(BASE / "persona_sample" / "persona_sample.csv", encoding="utf-8-sig")
    age = dict(zip(persona["persona_id"], persona["AGE"]))

    meta = {}
    for var in probes["var"].unique():
        codes = json.loads(probes.loc[probes["var"] == var, "codes"].iloc[0])
        kept = keep_codes(var, codes, vt)
        s = pd.to_numeric(clean[var], errors="coerce")
        s = s[s.isin(kept) & ~s.isin(STRUCT)]
        w = clean.loc[s.index, "FINALWT"].astype(float).groupby(s).sum()
        real = np.array([w.get(float(c), 0.0) for c in kept])
        real = real / real.sum()
        pos = np.linspace(0, 1, len(kept))
        meta[var] = dict(idx=[codes.index(c) for c in kept], real=real, pos=pos, e_real=float(real @ pos))

    rng = np.random.default_rng(SEED)
    pair_rows, between_rows, long_rows = [], [], []

    for tag, fname in MODELS.items():
        path = EXP / fname
        if not path.exists():
            continue
        L = pd.read_csv(path, encoding="utf-8-sig").set_index("probe_id")
        df = probes.set_index("probe_id").join(L[["model_dist"]], how="inner").reset_index()
        df["persona_id"] = df["persona_id"].fillna(-1).astype(int)
        print(f"=== {tag}: 로짓 {len(df)}건 ===", flush=True)

        for (level, var), g in df.groupby(["level", "var"]):
            m = meta[var]
            P = np.array([json.loads(x) for x in g["model_dist"]])[:, m["idx"]]
            P = P / P.sum(axis=1, keepdims=True)
            g = g.assign(row=np.arange(len(g)))
            d = DESIRABLE.get(var, 0)
            e = P @ m["pos"]
            wd = w1(P, m["real"])
            for r, (addr, reg, pid) in enumerate(zip(g["addressee"], g["register"], g["persona_id"])):
                long_rows.append(dict(model=tag, level=level, var=var, persona_id=pid,
                                      age=age.get(pid, np.nan), addressee=addr, register=reg,
                                      ds=d * (e[r] - m["e_real"]) if d else np.nan, wd=wd[r],
                                      sensitive=int(var in sensitive)))

            key = {(a, rg, p): i for i, (a, rg, p) in enumerate(zip(g["addressee"], g["register"], g["persona_id"]))}
            pids = sorted(set(g["persona_id"]))
            for kind, fixed_vals, pair in [("화계", ["당신", "이사람"], ("하십시오체", "해라체")),
                                           ("인칭", ["하십시오체", "해라체"], ("당신", "이사람"))]:
                ia, ib = [], []
                for fv in fixed_vals:
                    for p in pids:
                        ka = (fv, pair[0], p) if kind == "화계" else (pair[0], fv, p)
                        kb = (fv, pair[1], p) if kind == "화계" else (pair[1], fv, p)
                        if ka in key and kb in key:
                            ia.append(key[ka]); ib.append(key[kb])
                if not ia:
                    continue
                A, B = P[ia], P[ib]
                pair_rows.append(dict(model=tag, level=level, var=var, kind=kind,
                                      sensitive=int(var in sensitive), tv=float(tv(A, B).mean()),
                                      flip=float((A.argmax(1) != B.argmax(1)).mean())))

            if level in ("demo", "full") and len(pids) > 1:
                for (addr, reg), gc in g.groupby(["addressee", "register"]):
                    rows_ = gc["row"].to_numpy()
                    perm = rng.permutation(rows_)
                    A, B = P[rows_], P[perm]
                    ok = rows_ != perm
                    between_rows.append(dict(model=tag, level=level, var=var,
                                             sensitive=int(var in sensitive),
                                             tv=float(tv(A[ok], B[ok]).mean()),
                                             flip=float((A[ok].argmax(1) != B[ok].argmax(1)).mean())))

    pr = pd.DataFrame(pair_rows)
    br = pd.DataFrame(between_rows).assign(kind="페르소나간")
    lg = pd.DataFrame(long_rows)
    pr.to_csv(OUT / "조작점검_상세.csv", index=False, encoding="utf-8-sig")
    lg.to_csv(OUT / "인칭화계_long.csv", index=False, encoding="utf-8-sig")

    # ---------- 1. 조작 점검 ----------
    allr = pd.concat([pr, br], ignore_index=True)
    print("\n### 1. 조작 점검 — 평균 TV거리 (문항 평균). 인간 A/B 순서효과 TV 0.02-0.05")
    for lev in ["adult", "demo", "full"]:
        sub = allr[allr.level == lev]
        if sub.empty:
            continue
        print(f"\n[{lev}]")
        print(sub.pivot_table(index="model", columns="kind", values="tv").round(4).to_string())
    print("\n[최빈선택 뒤집힘 비율, demo+full]")
    sub = allr[allr.level.isin(["demo", "full"])]
    print(sub.pivot_table(index="model", columns="kind", values="flip").round(3).to_string())
    print("\n[화계 TV: 민감 vs 비민감, demo+full]")
    print(pr[(pr.kind == "화계") & pr.level.isin(["demo", "full"])]
          .pivot_table(index="model", columns="sensitive", values="tv").round(4).to_string())
    print("\n[화계 TV 상위 문항 (모델 평균, demo+full)]")
    top = (pr[(pr.kind == "화계") & pr.level.isin(["demo", "full"])]
           .groupby("var")["tv"].mean().sort_values(ascending=False).head(8))
    print(top.round(4).to_string())

    # ---------- 2a. 민감 문항 ds 혼합모형 ----------
    print("\n### 2a. 민감 8문항 ds 혼합모형 (demo+full). 계수 음수 = 바람직성 치우침 감소")
    mm_rows = []
    s = lg[lg.sensitive.eq(1) & lg.level.isin(["demo", "full"]) & lg["ds"].notna()].copy()
    s["plain"] = (s["register"] == "해라체").astype(int)
    s["third"] = (s["addressee"] == "이사람").astype(int)
    for tag, g in s.groupby("model"):
        g = g.copy()
        g["pid"] = g["persona_id"].astype(str)
        try:
            fit = smf.mixedlm("ds ~ plain * third + C(level)", g, groups=g["var"],
                              re_formula="~plain + third",
                              vc_formula={"pid": "0 + C(pid)"}).fit(method="lbfgs")
            note = "slope+persona"
        except Exception:
            fit = smf.mixedlm("ds ~ plain * third + C(level)", g, groups=g["var"],
                              re_formula="~plain + third").fit(method="lbfgs")
            note = "slope"
        for term in ["plain", "third", "plain:third"]:
            mm_rows.append(dict(model=tag, term=term, coef=fit.params[term], se=fit.bse[term],
                                p=fit.pvalues[term], spec=note))
    mm = pd.DataFrame(mm_rows)
    print(mm.round(4).to_string(index=False))

    # ---------- 2b. 이중차분: 29문항 Wasserstein 변화 ----------
    print("\n### 2b. 이중차분 — 조건 변화가 실제 분포와의 거리(W)를 바꾸는 정도, 민감 vs 비민감")
    print("    dW 음수 = 실제에 더 가까워짐. sensitive 계수 = 민감 문항에서 추가로 다른 정도")
    d = lg[lg.level.isin(["demo", "full"])]
    wide = d.pivot_table(index=["model", "level", "var", "persona_id", "sensitive"],
                         columns=["addressee", "register"], values="wd").reset_index()
    wide["dW_plain"] = ((wide[("당신", "해라체")] + wide[("이사람", "해라체")])
                        - (wide[("당신", "하십시오체")] + wide[("이사람", "하십시오체")])) / 2
    wide["dW_third"] = ((wide[("이사람", "하십시오체")] + wide[("이사람", "해라체")])
                        - (wide[("당신", "하십시오체")] + wide[("당신", "해라체")])) / 2
    wide.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in wide.columns]
    # 혼합모형은 문항 분산이 0 경계로 수렴해 SE가 안 나오는 경우가 있어, 추론 단위인
    # 문항으로 평균 낸 뒤 순열검정(민감/비민감 라벨 셔플)으로 판단한다.
    did_rows = []
    for tag, g in wide.groupby("model"):
        for dv in ["dW_plain", "dW_third"]:
            per = g.groupby(["var", "sensitive"])[dv].mean().reset_index()
            x = per[dv].to_numpy()
            lab = per["sensitive"].to_numpy()
            obs = x[lab == 1].mean() - x[lab == 0].mean()
            perm = np.array([x[p == 1].mean() - x[p == 0].mean()
                             for p in (rng.permutation(lab) for _ in range(20000))])
            did_rows.append(dict(model=tag, dv=dv, 비민감_평균=x[lab == 0].mean(),
                                 민감_평균=x[lab == 1].mean(), 차이=obs,
                                 perm_p=float((np.abs(perm) >= abs(obs)).mean())))
    did = pd.DataFrame(did_rows)
    print(did.round(4).to_string(index=False))

    mm.to_csv(OUT / "인칭화계_혼합모형.csv", index=False, encoding="utf-8-sig")
    did.to_csv(OUT / "인칭화계_이중차분.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장 위치: {OUT}")


if __name__ == "__main__":
    main()
