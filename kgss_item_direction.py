r"""
문항별 위치 오차와 방향 — 본실험 로짓 재사용, GPU 불필요

kgss_temperature_individual.py에서 모델 문제는 분포의 폭보다 위치(고르는 답 자체)에
있다는 결과가 나왔다. 이 스크립트는 그 위치 오차가 어느 문항에서 어느 방향인지 본다.

지표
    shift     모델 기대 위치 - 실제 기대 위치. 선택지 순번을 0-1로 정규화한 척도
              (첫 선택지 0, 마지막 1). 예측은 페르소나·조건을 모두 평균.
    mode_hit  페르소나별 예측의 최빈 선택이 실제 인구 최빈값과 같은 비율
    ds        민감 문항에서 shift x 방향부호. 양수면 "바람직한 쪽"으로 치우침

바람직한 방향(DESIRABLE)은 정렬 학습이 선호할 쪽(포용·평등·준법)으로 잡은 연구자
가정이다. 한국 사회 규범과 일치한다는 보장은 없고 팀 합의가 필요하다.
HATOPSEX는 어느 쪽이 바람직한지 모호해 제외했다.

MEDIAPOL은 프롬프트에 "8. DK/Refusal"이 선택지로 들어갔다(variables.csv의 legacy_dk에
등록 안 된 라벨이라 real_options() 필터를 통과). 실제 응답엔 8이 없으므로 여기서는
DK 라벨 코드를 빼고 재정규화한다.

인칭 x 화계: demo/full에서 같은 페르소나·같은 문항의 조건 간 ds 차이를 대응비교한다.

사용법:
    uv run kgss_item_direction.py
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent
EXP = BASE / "experiment"
OUT = EXP / "분석결과"
OUT.mkdir(exist_ok=True)
STRUCT = {-1.0, -8.0}
LEVELS = ["c0", "adult", "demo", "full"]
MODELS = {
    "exaone": "logit_exaone.csv", "kanana": "logit_kanana.csv",
    "exaone78b": "logit_exaone78b_fp16.csv", "kanana8b": "logit_kanana8b_fp16.csv",
}
# +1: 큰 코드가 바람직, -1: 작은 코드가 바람직
DESIRABLE = {
    "IMMCRIME": +1,    # 이민자가 범죄율 높인다 -> 반대
    "OPPSEGOV": +1,    # 불복종운동 참가 권리 -> 중요
    "TRUEKOR": -1,     # A(노력하면 한국인) 지지
    "OBEYLAWS": +1,    # 법과 규칙 준수 -> 중요
    "IMMIDEAS": -1,    # 이민자가 사회를 좋게 만든다 -> 찬성
    "OTHERS1": +1,     # 불행한 사람에 동정심 -> 그렇다
    "SEXATT3": +1,     # 동성애 -> 잘못되지 않았다
    "SEXROLE123": -1,  # 아내는 남편 경력 지원 -> 반대
}


def keep_codes(var, codes, vt):
    vl = json.loads(vt.at[var, "value_labels"])
    out = []
    for c in codes:
        lab = str(vl.get(str(float(c)), vl.get(str(c), "")))
        if lab.upper().startswith("DK"):
            continue
        out.append(c)
    return out


def main():
    probes = pd.read_csv(EXP / "experiment_probes.csv", encoding="utf-8-sig")
    items = pd.read_csv(BASE / "selection" / "본실험_문항30_초안.csv", encoding="utf-8-sig")
    group = dict(zip(items["변수명"], items["묶음"]))
    vt = pd.read_csv(BASE / "inventory" / "variables.csv").set_index("var")
    clean = pd.read_parquet(BASE / "inventory" / "kgss_clean.parquet")
    clean = clean[clean["YEAR"] == 2023]

    meta = {}
    for var in probes["var"].unique():
        codes = json.loads(probes.loc[probes["var"] == var, "codes"].iloc[0])
        kept = keep_codes(var, codes, vt)
        idx = [codes.index(c) for c in kept]
        s = pd.to_numeric(clean[var], errors="coerce")
        s = s[s.isin(kept) & ~s.isin(STRUCT)]
        w = clean.loc[s.index, "FINALWT"].astype(float).groupby(s).sum()
        real = np.array([w.get(float(c), 0.0) for c in kept])
        real = real / real.sum()
        pos = np.linspace(0, 1, len(kept))
        if len(kept) != len(codes):
            print(f"  {var}: DK 코드 {sorted(set(codes) - set(kept))} 제외 후 재정규화")
        meta[var] = dict(idx=idx, real=real, pos=pos, e_real=float(real @ pos),
                         real_mode=int(real.argmax()))

    item_rows, cond_rows = [], []
    for tag, fname in MODELS.items():
        path = EXP / fname
        if not path.exists():
            continue
        L = pd.read_csv(path, encoding="utf-8-sig").set_index("probe_id")
        df = probes.set_index("probe_id").join(L[["model_dist"]], how="inner").reset_index()
        print(f"=== {tag}: 로짓 {len(df)}건 ===", flush=True)

        for (level, var), g in df.groupby(["level", "var"]):
            m = meta[var]
            P = np.array([json.loads(x) for x in g["model_dist"]])[:, m["idx"]]
            P = P / P.sum(axis=1, keepdims=True)
            e = P @ m["pos"]
            d = DESIRABLE.get(var, 0)
            mean_p = P.mean(axis=0)
            item_rows.append(dict(model=tag, level=level, var=var, group=group.get(var, ""),
                                  e_model=float(e.mean()), e_real=m["e_real"],
                                  shift=float(e.mean() - m["e_real"]),
                                  ds=float(d * (e.mean() - m["e_real"])) if d else np.nan,
                                  mode_hit=float((P.argmax(axis=1) == m["real_mode"]).mean()),
                                  model_mode=int(mean_p.argmax()) + 1, real_mode=m["real_mode"] + 1))
            if d and level in ("adult", "demo", "full"):
                for (addr, reg, pid), e_i in zip(
                        zip(g["addressee"], g["register"], g["persona_id"].fillna(-1)), e):
                    cond_rows.append(dict(model=tag, level=level, var=var, addressee=addr,
                                          register=reg, persona_id=pid,
                                          ds=d * (e_i - m["e_real"])))

    it = pd.DataFrame(item_rows)
    it["level"] = pd.Categorical(it["level"], LEVELS, ordered=True)
    it.to_csv(OUT / "문항방향_상세.csv", index=False, encoding="utf-8-sig")

    print("\n### 묶음별 위치 오차 (|shift| 평균, mode_hit 평균) — full 조건")
    f = it[it.level == "full"].assign(abs_shift=lambda x: x["shift"].abs())
    print(f.pivot_table(index="group", columns="model", values="abs_shift").round(3).to_string())
    print()
    print(f.pivot_table(index="group", columns="model", values="mode_hit").round(3).to_string())

    print("\n### 문항별 shift (full 조건, 0-1 척도. 양수 = 뒤쪽 선택지로 치우침)")
    pv = f.pivot_table(index=["group", "var"], columns="model", values="shift").round(2)
    pv["실제최빈"] = f.groupby(["group", "var"])["real_mode"].first()
    print(pv.to_string())

    print("\n### 민감 문항 바람직성 치우침 ds (양수 = 바람직한 쪽, 문항 8개)")
    s = it[it["ds"].notna()]
    print(s.pivot_table(index="var", columns=["model"], values="ds",
                        aggfunc=lambda x: x[it.loc[x.index, "level"] == "full"].mean()).round(2).to_string())
    summ = []
    for (tag, level), g in s.groupby(["model", "level"], observed=True):
        v = g["ds"].to_numpy()
        summ.append(dict(model=tag, level=level, mean_ds=v.mean(), n_pos=int((v > 0).sum()), n=len(v),
                         sign_p=stats.binomtest(int((v > 0).sum()), len(v)).pvalue))
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "민감문항_바람직성_요약.csv", index=False, encoding="utf-8-sig")
    print("\n[모델 x 수준 요약] 부호검정은 문항 8개 기준(검정력 낮음)")
    print(summ.round(3).to_string(index=False))

    cr = pd.DataFrame(cond_rows)
    cr.to_csv(OUT / "민감문항_인칭화계_상세.csv", index=False, encoding="utf-8-sig")
    print("\n### 인칭 x 화계별 ds 평균 (민감 8문항, demo+full)")
    pf = cr[cr.level.isin(["demo", "full"])]
    print(pf.pivot_table(index="model", columns=["addressee", "register"], values="ds").round(3).to_string())

    print("\n### 대응비교: 같은 페르소나·문항·수준에서 조건 간 ds 차이")
    eff_rows = []
    for tag, g in pf.groupby("model"):
        w = g.pivot_table(index=["level", "var", "persona_id"], columns=["addressee", "register"], values="ds")
        third = (w[("이사람", "하십시오체")] + w[("이사람", "해라체")]) / 2
        second = (w[("당신", "하십시오체")] + w[("당신", "해라체")]) / 2
        plain = (w[("당신", "해라체")] + w[("이사람", "해라체")]) / 2
        formal = (w[("당신", "하십시오체")] + w[("이사람", "하십시오체")]) / 2
        inter = (w[("이사람", "해라체")] - w[("이사람", "하십시오체")]) - (w[("당신", "해라체")] - w[("당신", "하십시오체")])
        for name, diff in [("인칭(이사람-당신)", third - second), ("화계(해라-하십시오)", plain - formal),
                           ("상호작용", inter)]:
            # 페르소나·수준은 같은 문항 안에서 독립이 아니므로 문항 평균 8개로 검정
            per_item = diff.groupby(level="var").mean()
            t = stats.ttest_1samp(per_item, 0)
            eff_rows.append(dict(model=tag, effect=name, mean=float(diff.mean()),
                                 n_item=len(per_item), n_pos=int((per_item > 0).sum()),
                                 t=float(t.statistic), p=float(t.pvalue)))
    eff = pd.DataFrame(eff_rows)
    eff.to_csv(OUT / "민감문항_인칭화계_효과.csv", index=False, encoding="utf-8-sig")
    print("    가설(Chapala 등): 3인칭·해라체일수록 바람직성 치우침 감소 -> mean < 0")
    print(eff.round(4).to_string(index=False))

    print(f"\n저장 위치: {OUT}")


if __name__ == "__main__":
    main()
