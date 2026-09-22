r"""
C0(페르소나 없음)/adult(한국 성인)/demo(성별+연령)/full(6변수) 조건 분포 재현 타당성 검증

서울 논문(Kim, Park & Suh 2026)이 EXAONE 3.5 7.8B·Qwen3-30B 같은 오픈웨이트에서
subgroup r이 0.05~0.12로 붕괴했다고 보고함 — 우리가 쓰는 EXAONE 2.4B/Kanana 2.1B는
그보다도 작아서, 인칭×화계 헤드라인 분석 전에 "이 모델이 population 수준에서 뭐라도
재현하는지"부터 확인해야 한다.

측정 두 가지:
    population TVD/Wasserstein  모델 예측분포 vs 실제 KGSS 가중분포 (문항별, 전체 평균)
    subgroup r                  성별×연령 10셀의 "실제 셀평균 - 문항 전체평균"(편차) vs
                                 모델의 예측 기댓값. 문항 전체평균을 빼는 이유: C0는 모델
                                 예측이 셀마다 안 달라지므로, 셀 원값을 그대로 상관시키면
                                 문항 자체의 전반적 지지율만 재는 꼴이 된다(인구통계 하위
                                 집단 평균은 보통 문항 전체평균 근처에 몰려 있어서, 모델이
                                 전체평균만 잘 맞혀도 r이 높게 나옴). 편차로 그 효과를
                                 제거해야 "모델이 셀 간 차이를 우연히라도 맞히는지"를 재는
                                 지표가 되고, 페르소나가 없는 C0에서는 이론상 0 근처가
                                 나와야 정상이다(서울 논문 C0 실측 0.03).

네 조건 (CLAUDE.md 3/7절 "대조군 4수준"):
    c0    페르소나 없음. 문항당 프롬프트 1개, 모든 셀에 같은 예측값이 복사됨.
    adult "당신은 한국 성인입니다" 머리말만 추가. 구체적 인구통계는 없음 — c0와 똑같이
          문항당 프롬프트 1개, 모든 셀에 같은 예측값이 복사됨. c0와 비교하면 "막연한
          정체성 지칭"만으로 뭔가 달라지는지를 분리해서 본다.
    demo  "당신은 {연령대 대표나이}세 {성별}입니다" 머리말 추가. 문항×10셀 = 프롬프트
          10개, 셀마다 실제로 다른 예측값이 나옴.
    full  persona_sample.csv(FINALWT 가중 복원추출 응답자)에서 n명을 뽑아 6변수 서사형
          페르소나 문장("당신은 ~" 하십시오체)을 머리말로 붙임. 문항×n명 프롬프트.
          personas.csv의 실제 AGE를 성별×연령 10셀로 다시 묶어 demo/C0와 같은 방식으로
          채점 — 한 셀에 여러 명이 걸리면 그 셀의 모델 예측을 평균해서 문항당 값 하나로
          만든 뒤 상관시킨다.

    네 조건 모두 subgroup r을 같은 방식(편차 기준)으로 계산하므로 직접 비교 가능하다.
    c0에서 벗어나 adult·demo·full 순으로 r이 올라가면 페르소나 정보량이 늘수록 실제로
    셀 구분 능력이 생긴다는 증거고, 안 올라가면 조건화 자체가 이 모델 규모에서는
    무의미하다는 증거(Q1/Q2 대응).

사용법:
    uv run kgss_c0_validity.py build --level c0 --inv ./inventory --out ./c0_check
    uv run kgss_c0_validity.py build --level adult --inv ./inventory --out ./c0_check_adult
    uv run kgss_c0_validity.py build --level demo --inv ./inventory --out ./c0_check_demo
    uv run kgss_c0_validity.py build --level full --inv ./inventory --out ./c0_check_full --n-personas 40
    # Colab: uv run kgss_token_check.py --contam-probes probes.csv --out <out> --no-contam-generate
    uv run kgss_c0_validity.py score --out ./c0_check
    uv run kgss_c0_validity.py score --out ./c0_check_adult
    uv run kgss_c0_validity.py score --out ./c0_check_demo
    uv run kgss_c0_validity.py score --out ./c0_check_full
"""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

STRUCTURAL = {-8.0, -1.0}
DK_COLS = ["legacy_dk", "dk"]
INAP_COLS = ["legacy_inap", "inap"]
AGE_BINS = [18, 30, 40, 50, 60, 200]
AGE_LABELS = ["18-29", "30-39", "40-49", "50-59", "60+"]
AGE_REP = {"18-29": 24, "30-39": 35, "40-49": 45, "50-59": 55, "60+": 70}


def pick_col(dfr, names):
    for n in names:
        if n in dfr.columns:
            return n
    return None


def real_options(var, vt):
    if var not in vt.index:
        return []
    row = vt.loc[var]
    try:
        vl = json.loads(row["value_labels"])
    except Exception:
        return []
    dk_col, inap_col = pick_col(vt, DK_COLS), pick_col(vt, INAP_COLS)
    dk = set(json.loads(row[dk_col])) if dk_col and pd.notna(row[dk_col]) else set()
    inap = set(json.loads(row[inap_col])) if inap_col and pd.notna(row[inap_col]) else set()
    out = []
    for k, lab in sorted(vl.items(), key=lambda kv: float(kv[0])):
        code = float(k)
        if code in STRUCTURAL or code in dk or code in inap:
            continue
        out.append((code, lab))
    return out


def sex_labels(vt):
    vl = json.loads(vt.loc["SEX", "value_labels"])
    return {float(k): v.strip() for k, v in vl.items() if float(k) in (1.0, 2.0)}


def entropy_bits(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def weighted_dist(df, var, codes):
    s = df[[var, "FINALWT"]].dropna()
    s = s[s[var].isin(codes)]
    if len(s) < 100:
        return None
    tot = s["FINALWT"].sum()
    return [round(float(s.loc[s[var] == c, "FINALWT"].sum() / tot), 4) for c in codes]


def build(args):
    inv, outdir = Path(args.inv), Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(inv / "kgss_clean.parquet")
    sub = df[df["YEAR"] == args.year].copy()
    sub["AGE_GRP"] = pd.cut(sub["AGE"], AGE_BINS, labels=AGE_LABELS, right=False)

    wave = pd.read_csv(inv / "wave_items.csv")
    vt = pd.read_csv(inv / "variables.csv").set_index("var")
    worded = pd.read_csv(args.worded).set_index("var")["설문원문"]
    sexlab = sex_labels(vt)

    personas = None
    if args.level == "full":
        personas = pd.read_csv(args.personas).sample(n=args.n_personas, random_state=args.seed)
        personas["AGE_GRP"] = pd.cut(personas["AGE"], AGE_BINS, labels=AGE_LABELS, right=False)

    cand = wave[wave["is_item_cand"]]["var"].tolist()
    rows, cellrows = [], []
    n_skip = 0
    for v in cand:
        opts = real_options(v, vt)
        codes = [int(c) for c, _ in opts]
        if len(opts) < 2 or any(not (1 <= c <= 9) for c in codes) or len(set(codes)) != len(codes):
            n_skip += 1
            continue
        gold = weighted_dist(sub, v, [float(c) for c in codes])
        if gold is None:
            n_skip += 1
            continue
        grand_mean = float(np.sum(np.array(codes) * np.array(gold)))
        txt = str(worded.get(v, "") or "")
        opt_txt = "\n".join(f"{c}. {lab}" for c, (_, lab) in zip(codes, opts))
        item_text = f"문항: {txt}\n\n선택지:\n{opt_txt}\n\n번호로만 답하십시오."

        if args.level == "c0":
            rows.append({
                "probe_id": f"C0_{v}", "var": v, "prompt": item_text,
                "codes": json.dumps(codes), "gold": json.dumps(gold),
            })
        elif args.level == "adult":  # 구체적 인구통계 없이 "한국 성인" 머리말만
            rows.append({
                "probe_id": f"ADULT_{v}", "var": v,
                "prompt": f"당신은 한국 성인입니다.\n\n{item_text}",
                "codes": json.dumps(codes), "gold": json.dumps(gold),
            })
        elif args.level == "demo":  # 셀마다 별도 프롬프트 (성별+연령 머리말)
            for sex_code, sex_lab in sexlab.items():
                for age_grp, age_rep in AGE_REP.items():
                    head = f"당신은 {age_rep}세 {sex_lab}입니다."
                    rows.append({
                        "probe_id": f"DEMO_{v}_{int(sex_code)}_{age_grp}",
                        "var": v, "sex": sex_code, "age_grp": age_grp,
                        "prompt": f"{head}\n\n{item_text}",
                        "codes": json.dumps(codes), "gold": json.dumps(gold),
                    })
        else:  # full: 페르소나 n명마다 별도 프롬프트 (6변수 서사형 머리말)
            for _, p in personas.iterrows():
                rows.append({
                    "probe_id": f"FULL_{v}_{int(p['persona_id'])}",
                    "var": v, "sex": p["SEX"], "age_grp": p["AGE_GRP"],
                    "prompt": f"{p['persona_당신_하십시오체']}\n\n{item_text}",
                    "codes": json.dumps(codes), "gold": json.dumps(gold),
                })

        # 셀(성별x연령)별 실제 가중평균 응답 + 문항 전체평균 (subgroup r용, 전체평균은 편차 계산에 씀)
        for (sex, age_grp), g in sub.groupby(["SEX", "AGE_GRP"], observed=True):
            s = g[[v, "FINALWT"]].dropna()
            s = s[s[v].isin([float(c) for c in codes])]
            if s["FINALWT"].sum() <= 0 or len(s) < 5:
                continue
            mean_code = float((s[v] * s["FINALWT"]).sum() / s["FINALWT"].sum())
            cellrows.append({
                "var": v, "sex": sex, "age_grp": age_grp,
                "real_mean": mean_code, "real_grand": grand_mean,
            })

    print(f"문항 {len(cellrows) and len(set(r['var'] for r in cellrows))}개 확정 "
          f"(제외 {n_skip}개), 프로브 {len(rows)}건")
    pd.DataFrame(rows).to_csv(outdir / "probes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cellrows).to_csv(outdir / "real_cells.csv", index=False, encoding="utf-8-sig")
    print(f"-> {outdir / 'probes.csv'}, {outdir / 'real_cells.csv'}")


def score(args):
    outdir = Path(args.out)
    probes = pd.read_csv(outdir / "probes.csv")
    logits = pd.read_csv(outdir / "logit_responses.csv")
    cells = pd.read_csv(outdir / "real_cells.csv")
    has_cells = "sex" in probes.columns
    level_name = {"C0_": "C0(페르소나 없음)", "ADULT_": "adult(한국 성인)",
                  "DEMO_": "demo(성별+연령)", "FULL_": "full(6변수)"}
    level = next((v for k, v in level_name.items() if probes["probe_id"].iloc[0].startswith(k)), "?")

    d = probes.merge(logits, on="probe_id", how="inner")
    from scipy.stats import wasserstein_distance, pearsonr

    L = [f"# {level} 조건 분포 재현 타당성 검증\n"]
    for model, g in d.groupby("model"):
        L.append(f"\n## {model}\n")
        g = g.copy()
        ws, masses = [], []
        model_pred = []
        ent_model, ent_gold = [], []
        for _, r in g.iterrows():
            gold = np.array(json.loads(r["gold"]))
            pred = np.array(json.loads(r["model_dist"]))
            n = len(gold)
            ws.append(wasserstein_distance(range(n), range(n), u_weights=pred, v_weights=gold))
            masses.append(r["mass"])
            codes = json.loads(r["codes"])
            model_pred.append(float(np.sum(np.array(codes) * pred)))
            ent_model.append(entropy_bits(pred))
            ent_gold.append(entropy_bits(gold))
        g["model_pred"] = model_pred
        L.append(f"- 문항 {g['var'].nunique()}개 / 프로브 {len(g)}건, "
                 f"평균 Wasserstein거리 **{np.mean(ws):.4f}**, 평균 선택지확률질량 {np.mean(masses):.4f}\n")
        L.append(f"- 모드 붕괴 점검: 모델 출력 엔트로피 **{np.mean(ent_model):.3f}bit**"
                 f"(유효 선택지 {2**np.mean(ent_model):.2f}개) vs 실제 응답 엔트로피 "
                 f"{np.mean(ent_gold):.3f}bit(유효 선택지 {2**np.mean(ent_gold):.2f}개) "
                 f"— 비율 {np.mean(ent_model)/np.mean(ent_gold):.2f}\n")

        if has_cells:
            # 셀 하나에 여러 프롬프트(full의 여러 페르소나)가 걸리면 문항당 값 하나로 평균
            cell_pred = g.groupby(["var", "sex", "age_grp"], observed=True)["model_pred"].mean().reset_index()
            merged = cell_pred.merge(cells, on=["var", "sex", "age_grp"], how="inner")
        else:
            preds = dict(zip(g["var"], g["model_pred"]))
            merged = cells.copy()
            merged["model_pred"] = merged["var"].map(preds)
            merged = merged.dropna(subset=["model_pred"])
        merged["real_dev"] = merged["real_mean"] - merged["real_grand"]

        rs = []
        for (sex, age_grp), gc in merged.groupby(["sex", "age_grp"], observed=True):
            if len(gc) < 5:
                continue
            r, p = pearsonr(gc["real_dev"], gc["model_pred"])
            rs.append(r)
        if rs:
            L.append(f"- subgroup r, 편차 기준 (성별×연령 {len(rs)}셀 평균): **{np.mean(rs):.3f}** "
                     f"(범위 {min(rs):.3f}~{max(rs):.3f})\n")
        if has_cells:
            L.append(f"> {level}. C0(같은 방식으로 계산한 subgroup r이 0 근처)과 비교해서 "
                     "여기서 r이 0에서 벗어나 올라가면, 이 조건의 페르소나 정보가 실제로 "
                     "셀 구분 능력을 만든다는 증거.\n")
        else:
            L.append(f"> {level}. 구체적 인구통계가 없어 모델 예측이 셀마다 안 달라지므로, "
                     "이 r은 \"모델이 인구통계 하위집단 차이를 우연히라도 맞히는지\"를 재는 "
                     "것이지 실제 조건화 능력을 재는 게 아니다. 이론상 0 근처가 정상"
                     "(서울 논문 C0 실측 0.03).\n")

    text = "".join(L)
    (outdir / "c0_validity_result.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"-> {outdir / 'c0_validity_result.md'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--level", choices=["c0", "adult", "demo", "full"], default="c0")
    b.add_argument("--inv", default="./inventory")
    b.add_argument("--worded", default="./selection/variables_worded.csv")
    b.add_argument("--year", type=int, default=2023)
    b.add_argument("--out", default="./c0_check")
    b.add_argument("--personas", default="./persona_sample/persona_sample.csv")
    b.add_argument("--n-personas", type=int, default=40)
    b.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("score")
    s.add_argument("--out", default="./c0_check")
    args = ap.parse_args()
    {"build": build, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
