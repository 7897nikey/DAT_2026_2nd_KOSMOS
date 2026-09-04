r"""
KGSS 사전학습 오염 프로빙

주 회차(2023) 자료가 모델의 사전학습에 포함되었는지 점검한다.
포함이 확인되면 재현 성능이 시뮬레이션 능력이 아니라 암기의 결과일 수 있으므로,
실험 설계 자체를 다시 검토해야 한다.

네 가지 프로브를 만든다. 모두 통제 조건을 포함한다.

  A. 로짓 기반 분포 재현
     문항을 주고 "번호로만" 답하게 강제한 뒤, 선택지 숫자 토큰(1~9)의 첫 토큰
     로짓을 softmax하여 모델이 암묵적으로 갖는 응답 분포를 얻는다. 텍스트 생성
     + 문자열 파싱이 아니라 forward pass 한 번으로 끝나므로 포맷팅 실패에 흔들리지
     않고, B와 동일한 통제 축(출처 명시/미명시)을 그대로 쓸 수 있다.
     통제: 같은 문항을 "KGSS 2023년 조사에 사용된 문항입니다"로 밝힌 조건과
           밝히지 않은 조건으로 나눈다. 실제 가중 분포(FINALWT)와의 Wasserstein
           거리가 출처를 밝혔을 때만 유의하게 줄어들면, 일반적 사회 통념이 아니라
           해당 조사의 실측치를 기억하고 있다는 뜻이다.
     실행: 이 스크립트의 build로 프로브를 만든 뒤, 로짓 추출은
           `kgss_token_check.py --contam-probes`로 수행한다 (모델 로드가 필요해
           GPU 환경에서 돌리며, 그 스크립트에 이미 있는 토크나이저/모델 로딩과
           첫 토큰 로짓 추출 인프라를 재사용한다).

  B. 원문 완성 (guided prompting)
     설문 문장의 앞부분을 주고 나머지를 생성하게 한다.
     통제: 출처를 밝힌 조건과 밝히지 않은 조건을 대조한다.
           출처를 밝혔을 때만 일치도가 오르면 해당 문서를 기억하고 있다는 뜻이다.
           이것이 오염 탐지의 표준 설계다 (Golchin & Surdeanu 2023).

  C. 선택지 배열 재현
     선택지를 설문지에 제시된 순서대로 나열하게 한다.
     통제: 우연히 맞을 확률(1/n!)과 비교한다.
           순서까지 맞으면 코드북 자체가 학습되었을 가능성이 높다.

  D. 회차별 성능 곡선 (별도 실행)
     같은 문항을 여러 회차에 대해 실행한다. 오래된 회차를 더 잘 맞히면
     재현이 아니라 암기 신호다. 기존 응답 수집 파이프라인을 그대로 쓰면 되므로
     이 스크립트는 대상 문항 목록만 제공한다.

사용법
    # 1) 프로브 생성 (A/B/C 프롬프트 + D 대상 목록)
    uv run kgss_contamination.py build --inv .\inventory --worded .\selection\variables_worded.csv --out .\probe

    # 2-A) A(로짓): probe/probes.csv를 token_check로 넘겨 로짓 추출
    uv run kgss_token_check.py --contam-probes .\probe\probes.csv --out .\probe --device cuda
    #     -> probe/logit_responses.csv 생성 (probe_id, model, model_dist, mass)

    # 2-B) B/C(생성): probe/probes.csv의 prompt 열을 기존 호출 파이프라인에 넣어 응답 수집
    #      결과를 probe/responses.csv로 저장 (열: probe_id, response)

    # 3) 채점
    uv run kgss_contamination.py score --out .\probe
"""

import argparse
import difflib
import io
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SURVEY = "한국종합사회조사(KGSS)"


# ---------------------------------------------------------------- 유틸
def parse_options(s: str) -> list[tuple[float, str]]:
    """'1=매우 그렇다 | 2=약간 그렇다' -> [(1.0, '매우 그렇다'), ...]"""
    out = []
    for part in str(s).split("|"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out.append((float(k.strip()), v.strip()))
        except ValueError:
            continue
    return out


def weighted_dist(df: pd.DataFrame, var: str, codes: list[float]) -> list[float] | None:
    """FINALWT 가중 응답 분포. 실제값(정답)으로 쓴다."""
    s = df[[var, "FINALWT"]].dropna()
    s = s[s[var].isin(codes)]
    if len(s) < 100:
        return None
    tot = s["FINALWT"].sum()
    return [round(float(s.loc[s[var] == c, "FINALWT"].sum() / tot), 4) for c in codes]


# ---------------------------------------------------------------- 생성
def build(args):
    inv, outdir = Path(args.inv), Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    df = pd.read_parquet(inv / "kgss_clean.parquet")
    wave = pd.read_csv(inv / "wave_items.csv")
    worded = pd.read_csv(args.worded).set_index("var")

    cand = wave[wave["is_item_cand"]]["var"].tolist()
    cand = [v for v in cand if v in worded.index and v in df.columns]
    rng.shuffle(cand)

    sub = df[df["YEAR"] == args.year]
    rows = []

    print(f"[1/3] 문항 선택 (후보 {len(cand)}개)")
    picked = []
    for v in cand:
        txt = str(worded.at[v, "설문원문"] or "")
        opts = parse_options(worded.at[v, "선택지"])
        if len(txt) < 40 or not (2 <= len(opts) <= 7):
            continue
        codes = [c for c, _ in opts]
        dist = weighted_dist(sub, v, codes)
        if dist is None:
            continue
        picked.append((v, txt, opts, dist))
        if len(picked) >= args.n_items:
            break
    print(f"      {len(picked)}개 확정")

    # ---------- A. 로짓 기반 분포 재현 (출처 명시 / 미명시)
    print("[2/3] 프로브 생성")
    n_skip_digit = 0
    for v, txt, opts, dist in picked:
        codes = [int(c) for c, _ in opts]
        # 첫 토큰 로짓으로 판별하려면 선택지가 1~9 범위의 단일 숫자 토큰이어야 한다
        # (kgss_token_check.py 실측: 숫자 1~9는 모두 단일 토큰).
        if any(not (1 <= c <= 9) for c in codes) or len(set(codes)) != len(codes):
            n_skip_digit += 1
            continue
        opt_txt = "\n".join(f"{c}. {lab}" for c, (_, lab) in zip(codes, opts))
        for cond, head in [
            ("출처명시", f"다음은 {SURVEY} {args.year}년 조사에 사용된 문항입니다. "
                        f"당신이 이 조사에 응답하는 대한민국 성인이라고 가정하고 답하십시오."),
            ("출처미명시", "다음 문항에 답하십시오. "
                         "당신이 대한민국 성인이라고 가정하고 답하십시오."),
        ]:
            rows.append({
                "probe_id": f"A_{v}_{cond}",
                "type": "A_로짓분포",
                "condition": cond,
                "var": v, "year": args.year,
                "prompt": (f"{head}\n\n문항: {txt}\n\n선택지:\n{opt_txt}\n\n"
                          f"번호로만 답하십시오."),
                "gold": json.dumps(dist),
                "codes": json.dumps(codes),
            })
    if n_skip_digit:
        print(f"      A: 선택지 코드가 1~9 범위를 벗어나 {n_skip_digit}개 문항 제외")

    # ---------- B. 원문 완성 (출처 명시 / 미명시)
    for v, txt, opts, dist in picked:
        cut = max(20, int(len(txt) * args.prefix_ratio))
        prefix, suffix = txt[:cut], txt[cut:]
        if len(suffix) < 15:
            continue
        for cond, head in [
            ("출처명시", f"다음은 {SURVEY} {args.year}년 설문지에 실린 문항입니다. "
                        f"이어지는 부분을 원문 그대로 완성하십시오."),
            ("출처미명시", "다음 문장에 이어지는 부분을 자연스럽게 완성하십시오."),
        ]:
            rows.append({
                "probe_id": f"B_{v}_{cond}",
                "type": "B_원문완성",
                "condition": cond,
                "var": v, "year": args.year,
                "prompt": f"{head}\n\n{prefix}",
                "gold": suffix,
            })

    # ---------- C. 선택지 배열 재현
    for v, txt, opts, dist in picked:
        labels = [lab for _, lab in opts]
        shuffled = labels[:]
        rng.shuffle(shuffled)
        if shuffled == labels:
            shuffled = labels[::-1]
        rows.append({
            "probe_id": f"C_{v}",
            "type": "C_선택지순서",
            "condition": "실제연도",
            "var": v, "year": args.year,
            "prompt": (
                f"{SURVEY} {args.year}년 조사의 다음 문항에서, 설문지에 제시된 "
                f"선택지 순서를 답하십시오.\n\n문항: {txt}\n\n"
                f"아래 선택지를 설문지에 나온 순서대로 나열하십시오.\n"
                + "\n".join(f"- {s}" for s in shuffled)
                + "\n\n순서대로 한 줄에 하나씩만 적으십시오."
            ),
            "gold": json.dumps(labels, ensure_ascii=False),
        })

    p = pd.DataFrame(rows)
    p.to_csv(outdir / "probes.csv", index=False, encoding="utf-8-sig")
    print(f"      프로브 {len(p)}건")
    print(p.groupby(["type", "condition"]).size().to_string())

    # ---------- D. 회차별 성능용 문항 목록
    print("[3/3] 회차별 성능 곡선 대상 목록")
    common = pd.read_csv(inv / "common_items.csv")
    tgt = common[common["is_item_cand"]]["var"].tolist()[:args.n_items]
    pd.DataFrame({"var": tgt}).to_csv(
        outdir / "wave_curve_items.csv", index=False, encoding="utf-8-sig")
    print(f"      {len(tgt)}개 (기존 응답 수집 파이프라인으로 여러 회차 실행)")

    (outdir / "README.md").write_text(
        "# 오염 프로빙 절차\n\n"
        "1. `probes.csv`의 A(로짓분포) 행은 `kgss_token_check.py --contam-probes`로 "
        "돌려 `logit_responses.csv`를 만든다 (열: `probe_id`, `model`, `model_dist`, `mass`)\n"
        "2. `probes.csv`의 B/C 행은 `prompt` 열을 기존 호출 파이프라인에 투입, "
        "응답을 `responses.csv`로 저장 (열: `probe_id`, `response`)\n"
        "3. `uv run kgss_contamination.py score --out .` 실행\n\n"
        "## 판정 기준\n\n"
        "| 프로브 | 오염 신호 |\n|---|---|\n"
        "| A | 출처를 밝힌 조건에서만 모델의 암묵적 분포(로짓)가 실제 가중 분포에 "
        "유의하게 가까워짐 (Wasserstein거리 감소) |\n"
        "| B | 출처를 밝힌 조건에서만 일치도가 유의하게 높음 |\n"
        "| C | 순서 정확도가 우연 확률(1/n!)을 크게 초과 |\n"
        "| D | 오래된 회차를 최근 회차보다 잘 맞힘 |\n",
        encoding="utf-8")
    print(f"\n완료 -> {outdir.resolve()}")


# ---------------------------------------------------------------- 채점
def score(args):
    outdir = Path(args.out)
    probes = pd.read_csv(outdir / "probes.csv")

    L = ["# 오염 프로빙 결과\n"]

    # ---------- A (로짓 기반)
    a = probes[probes["type"] == "A_로짓분포"].copy()
    lg_path = outdir / "logit_responses.csv"
    if len(a) and lg_path.exists():
        lg = pd.read_csv(lg_path)
        a = a.merge(lg, on="probe_id", how="inner")
        recs = []
        for _, r in a.iterrows():
            gold = np.array(json.loads(r["gold"]), dtype=float)
            pred = np.array(json.loads(r["model_dist"]), dtype=float)
            n = len(gold)
            try:
                from scipy.stats import wasserstein_distance
                w = float(wasserstein_distance(range(n), range(n),
                                               u_weights=pred, v_weights=gold))
            except Exception:
                w = float(np.sum(np.abs(np.cumsum(pred) - np.cumsum(gold))))
            recs.append({"model": r.get("model", "unknown"), "var": r["var"],
                        "condition": r["condition"], "wasserstein": w,
                        "mass": r.get("mass", np.nan)})
        ar = pd.DataFrame(recs)
        L.append("## A. 로짓 기반 분포 재현\n")
        L.append("| 모델 | 조건 | n | 평균 Wasserstein거리 | 평균 선택지확률질량 |\n"
                 "|---|---|---|---|---|")
        for (model, cond), g in ar.groupby(["model", "condition"]):
            L.append(f"| {model} | {cond} | {len(g)} | {g['wasserstein'].mean():.4f} "
                     f"| {g['mass'].mean():.4f} |")
        for model, gm in ar.groupby("model"):
            piv = gm.pivot_table(index="var", columns="condition", values="wasserstein")
            if {"출처명시", "출처미명시"} <= set(piv.columns):
                # 양수 = 출처를 밝혔을 때 실제 분포에 더 가까워짐
                diff = (piv["출처미명시"] - piv["출처명시"]).dropna()
                L.append(f"\n### {model}\n- 쌍별 차이(미명시-명시) 평균 **{diff.mean():+.4f}** "
                         f"(명시가 더 가까움 {int((diff > 0).sum())}/{len(diff)})")
                try:
                    from scipy import stats
                    t, p = stats.ttest_rel(piv["출처미명시"].dropna(), piv["출처명시"].dropna())
                    L.append(f"- 대응표본 t검정: t={t:.3f}, p={p:.4f}")
                except Exception:
                    pass
        L.append("\n> **이것이 핵심 지표입니다.** 출처를 밝힌 조건에서만 모델의 암묵적 분포가 "
                 "실제 응답 분포에 유의하게 가까워지면(Wasserstein거리 감소) 해당 조사의 실측치를 "
                 "기억하고 있다는 뜻입니다. 두 조건이 비슷하면 일반적 사회 통념으로 추정한 것입니다. "
                 "`mass`(선택지 토큰에 실린 확률질량)가 낮은 행은 로짓이 선택지 밖으로 새어나간 "
                 "것이므로 해당 관측의 신뢰도가 낮습니다.\n")
    elif len(a):
        L.append("## A. 로짓 기반 분포 재현\n\n`logit_responses.csv`가 없어 건너뜁니다. "
                 "`uv run kgss_token_check.py --contam-probes probe\\probes.csv --out probe`"
                 "로 먼저 생성하십시오.\n")

    resp_path = outdir / "responses.csv"
    if resp_path.exists():
        resp = pd.read_csv(resp_path)
        d = probes.merge(resp, on="probe_id", how="inner")
        print(f"응답 {len(d)}건 / 프로브 {len(probes)}건\n")
    else:
        d = probes.iloc[0:0].assign(response=[])
        print("responses.csv가 없어 B/C는 건너뜁니다.\n")

    # ---------- B
    b = d[d["type"] == "B_원문완성"].copy()
    if len(b):
        b["sim"] = [
            difflib.SequenceMatcher(
                None, str(r["gold"]), str(r["response"])[:len(str(r["gold"])) * 2]
            ).ratio()
            for _, r in b.iterrows()
        ]
        L.append("## B. 원문 완성\n")
        L.append("| 모델 | 조건 | n | 평균 일치도 | 중앙값 |\n|---|---|---|---|---|")
        for (model, cond), g in b.groupby(["model", "condition"]):
            L.append(f"| {model} | {cond} | {len(g)} | {g['sim'].mean():.3f} "
                     f"| {g['sim'].median():.3f} |")
        for model, gm in b.groupby("model"):
            piv = gm.pivot_table(index="var", columns="condition", values="sim")
            if {"출처명시", "출처미명시"} <= set(piv.columns):
                diff = (piv["출처명시"] - piv["출처미명시"]).dropna()
                L.append(f"\n### {model}\n- 쌍별 차이 평균 **{diff.mean():+.3f}** "
                         f"(양수 {int((diff > 0).sum())}/{len(diff)})")
                try:
                    from scipy import stats
                    t, p = stats.ttest_rel(piv["출처명시"].dropna(),
                                           piv["출처미명시"].dropna())
                    L.append(f"- 대응표본 t검정: t={t:.3f}, p={p:.4f}")
                except Exception:
                    pass
        L.append("\n> **이것이 핵심 지표입니다.** 출처를 밝힌 조건에서만 일치도가 유의하게 "
                 "높으면 해당 문서를 기억하고 있다는 뜻입니다. 두 조건이 비슷하면 "
                 "단순히 한국어 설문 문체를 재현한 것입니다.\n")

    # ---------- C
    c = d[d["type"] == "C_선택지순서"].copy()
    if len(c):
        recs = []
        for _, r in c.iterrows():
            gold = json.loads(r["gold"])
            lines = [x.strip(" -·*0123456789.") for x in str(r["response"]).split("\n")]
            lines = [x for x in lines if x]
            hit = lines[:len(gold)] == gold
            chance = 1 / math.factorial(len(gold))
            recs.append({"model": r.get("model", "unknown"), "n_opt": len(gold),
                        "exact": hit, "chance": chance})
        cr = pd.DataFrame(recs)
        L.append("## C. 선택지 배열 재현\n")
        L.append("| 모델 | n | 정확 일치율 | 우연 기대값 |\n|---|---|---|---|")
        for model, g in cr.groupby("model"):
            L.append(f"| {model} | {len(g)} | {g['exact'].mean()*100:.1f}% "
                     f"| {g['chance'].mean()*100:.1f}% |")
        L.append("> 우연 확률을 크게 초과하면 코드북 또는 설문지 자체가 학습된 신호입니다. "
                 "다만 선택지가 논리적 순서를 가지면(긍정→부정) 추론으로도 맞출 수 있으므로 "
                 "보조 근거로만 사용합니다.\n")

    (outdir / "contamination_result.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n완료 -> {outdir / 'contamination_result.md'}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    b = sp.add_parser("build")
    b.add_argument("--inv", default="./inventory")
    b.add_argument("--worded", default="./selection/variables_worded.csv")
    b.add_argument("--out", default="./probe")
    b.add_argument("--year", type=int, default=2023)
    b.add_argument("--n-items", type=int, default=20)
    b.add_argument("--prefix-ratio", type=float, default=0.5)
    b.add_argument("--seed", type=int, default=42)
    b.set_defaults(func=build)

    s = sp.add_parser("score")
    s.add_argument("--out", default="./probe")
    s.set_defaults(func=score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
