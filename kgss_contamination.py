r"""
KGSS 사전학습 오염 프로빙

주 회차(2023) 자료가 모델의 사전학습에 포함되었는지 점검한다.
포함이 확인되면 재현 성능이 시뮬레이션 능력이 아니라 암기의 결과일 수 있으므로,
실험 설계 자체를 다시 검토해야 한다.

다섯 가지 프로브를 만든다. 모두 통제 조건을 포함한다.

  A. 로짓 기반 분포 재현
     문항을 주고 "번호로만" 답하게 강제한 뒤, 선택지 숫자 토큰(1~9)의 첫 토큰
     로짓을 softmax하여 모델이 암묵적으로 갖는 응답 분포를 얻는다. 텍스트 생성
     + 문자열 파싱이 아니라 forward pass 한 번으로 끝나므로 포맷팅 실패에 흔들리지
     않고, B와 동일한 통제 축(출처 명시/미명시/가짜출처)을 그대로 쓸 수 있다.
     통제: 같은 문항을 "KGSS 2023년 조사에 사용된 문항입니다"로 밝힌 조건,
           밝히지 않은 조건, **존재하지 않는 가짜 조사명으로 밝힌 조건** 셋으로
           나눈다. 실제 가중 분포(FINALWT)와의 Wasserstein거리가 진짜 출처를
           밝혔을 때만 유의하게 줄고 가짜 출처에서는 안 줄면, 일반적인
           "공식조사 프레이밍" 효과가 아니라 이 조사 자체를 특정해서 기억하고
           있다는 뜻이다. 가짜 출처에서도 똑같이 줄면 "공식성 프레이밍" 효과일
           뿐 오염 증거가 아니다 (2026-09 팀 논의 — 실제 관측된 유의값이
           프레이밍 효과와 구분 안 된다는 지적 반영).
     실행: 이 스크립트의 build로 프로브를 만든 뒤, 로짓 추출은
           `kgss_token_check.py --contam-probes`로 수행한다 (모델 로드가 필요해
           GPU 환경에서 돌리며, 그 스크립트에 이미 있는 토크나이저/모델 로딩과
           첫 토큰 로짓 추출 인프라를 재사용한다).

  B. 원문 완성 (guided prompting)
     설문 문장의 앞부분을 주고 나머지를 생성하게 한다.
     통제: 출처 명시/미명시/가짜출처 셋을 대조한다(A와 동일한 논리).
           진짜 출처에서만 일치도가 오르고 가짜 출처에서는 안 오르면 이 설문
           문서 자체를 기억하고 있다는 뜻이다. 이것이 오염 탐지의 표준 설계다
           (Golchin & Surdeanu 2023) — 가짜출처 대조는 그 표준 설계에 우리가
           추가한 것.

  C. 선택지 배열 재현
     선택지를 설문지에 제시된 순서대로 나열하게 한다.
     통제: 우연히 맞을 확률(1/n!)과 비교한다.
           순서까지 맞으면 코드북 자체가 학습되었을 가능성이 높다.

  D. 회차별 성능 곡선 (별도 실행)
     같은 문항을 여러 회차에 대해 실행한다. 오래된 회차를 더 잘 맞히면
     재현이 아니라 암기 신호다. 기존 응답 수집 파이프라인을 그대로 쓰면 되므로
     이 스크립트는 대상 문항 목록만 제공한다.

  E. 직접 회상 (Silicon Sampling in Seoul, Kim/Park/Suh 2026 방식)
     "이 문항의 실제 응답 분포를 기억하는가?"를 직접 묻고, 정말로 기억하는
     경우에만 백분율로 답하고 그렇지 않으면 "모름"이라고 답하도록 명시적으로
     지시한다. 진짜 출처/가짜 출처 두 조건으로 나눈다.
     통제: 가짜 출처에서도 자신 있게(회피 안 하고) 답하면 모델이 "모른다"고
           말해야 할 때 말 못 하는 것 — 이 채널 자체가 신뢰 못 할 신호라는 뜻.
           진짜 출처에서만 회피율이 낮고 정확도가 높으면 오염 쪽에 힘을 싣는다.
     우리 A/B/C(모델의 실제 계산·행동 측정)와 다른 층위 — 모델의 메타인지
     (자기가 아는지 모르는지 얼마나 정확히 아는지)를 본다. 회상정확도와
     시뮬레이션정확도의 상관(서울 논문의 핵심 지표)은 본실험 데이터가 있어야
     계산 가능하므로 여기서는 회피율·정확도까지만 낸다.

사용법
    # 1) 프로브 생성 (A/B/C/E 프롬프트 + D 대상 목록)
    uv run kgss_contamination.py build --inv .\inventory --worded .\selection\variables_worded.csv --out .\probe

    # 2-A) A(로짓): probe/probes.csv를 token_check로 넘겨 로짓 추출
    uv run kgss_token_check.py --contam-probes .\probe\probes.csv --out .\probe --device cuda
    #     -> probe/logit_responses.csv 생성 (probe_id, model, model_dist, mass)

    # 2-B) B/C/E(생성): probe/probes.csv의 prompt 열을 기존 호출 파이프라인에 넣어 응답 수집
    #      결과를 probe/responses.csv로 저장 (열: probe_id, response)
    #      (kgss_token_check.py --contam-probes를 쓰면 이것도 같이 됨)

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

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":  # 다른 kgss_*.py가 import할 때 이중 래핑 방지
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SURVEY = "한국종합사회조사(KGSS)"
# 가짜출처 대조군용 존재하지 않는 조사명. 실제 한국 사회조사와 이름이
# 겹치지 않도록 지었다 — "공식조사 프레이밍" 효과와 "이 조사를 특정해서
# 기억함"을 분리하는 게 목적이라, 그럴듯하되 검색해도 안 나와야 한다.
FAKE_SURVEY = "한국사회인식조사(KSPS)"


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


def parse_percents(text: str, n: int) -> list[float] | None:
    """'12.3, 45.6, 42.1' -> [12.3, 45.6, 42.1] (100 근처로 재정규화).
    숫자가 n개 미만이면 파싱 실패로 본다."""
    nums = re.findall(r"\d+\.?\d*", str(text))
    if len(nums) < n:
        return None
    vals = [float(x) for x in nums[:n]]
    s = sum(vals)
    if s <= 0:
        return None
    return [v / s * 100 for v in vals]


# ---------------------------------------------------------------- 생성
def build(args):
    inv, outdir = Path(args.inv), Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    df = pd.read_parquet(inv / "kgss_clean.parquet")
    wave = pd.read_csv(inv / "wave_items.csv")
    worded = pd.read_csv(args.worded).set_index("var")
    battery = pd.read_csv(inv / "variables.csv").set_index("var")["battery"]

    cand = wave[wave["is_item_cand"]]["var"].tolist()
    cand = [v for v in cand if v in worded.index and v in df.columns]
    rng.shuffle(cand)

    sub = df[df["YEAR"] == args.year]
    rows = []

    print(f"[1/3] 문항 선택 (후보 {len(cand)}개)")
    picked = []
    seen_battery = set()
    n_skip_battery = 0
    for v in cand:
        txt = str(worded.at[v, "설문원문"] or "")
        opts = parse_options(worded.at[v, "선택지"])
        if len(txt) < 40 or not (2 <= len(opts) <= 7):
            continue
        codes = [c for c, _ in opts]
        dist = weighted_dist(sub, v, codes)
        if dist is None:
            continue
        # 배터리당 1개만 채택 (CLAUDE.md 6절 함정 — 배터리 중복이면 그 문항들끼리
        # 표현이 비슷해 대응표본 t검정의 독립 관측치 가정이 약해진다).
        bat = battery.get(v, v)
        if bat in seen_battery:
            n_skip_battery += 1
            continue
        seen_battery.add(bat)
        picked.append((v, txt, opts, dist))
        if len(picked) >= args.n_items:
            break
    if n_skip_battery:
        print(f"      배터리 중복으로 {n_skip_battery}개 건너뜀")
    print(f"      {len(picked)}개 확정 (고유 배터리 {len(seen_battery)}개)")

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
            ("가짜출처", f"다음은 {FAKE_SURVEY} {args.year}년 조사에 사용된 문항입니다. "
                       f"당신이 이 조사에 응답하는 대한민국 성인이라고 가정하고 답하십시오."),
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
            ("가짜출처", f"다음은 {FAKE_SURVEY} {args.year}년 설문지에 실린 문항입니다. "
                       f"이어지는 부분을 원문 그대로 완성하십시오."),
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

    # ---------- E. 직접 회상 (진짜 출처 / 가짜 출처)
    for v, txt, opts, dist in picked:
        opt_txt = "\n".join(f"{int(c)}. {lab}" for c, lab in opts)
        for cond, survey in [("실제", SURVEY), ("가짜", FAKE_SURVEY)]:
            rows.append({
                "probe_id": f"E_{v}_{cond}",
                "type": "E_직접회상",
                "condition": cond,
                "var": v, "year": args.year,
                "prompt": (
                    f"{survey} {args.year}년 조사에서 다음 문항에 대한 응답 분포를 "
                    f"정확히 기억하십니까?\n\n문항: {txt}\n\n선택지:\n{opt_txt}\n\n"
                    f"정말로 정확한 수치를 기억하는 경우에만 각 선택지의 백분율을 "
                    f"순서대로 쉼표로 구분해 답하십시오. 예: 12.3, 45.6, 42.1\n"
                    f"확실히 기억하지 못하면 반드시 '모름'이라고만 답하십시오. "
                    f"추측해서 답하지 마십시오."
                ),
                "gold": json.dumps(dist),
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
        "2. `probes.csv`의 B/C/E 행은 `prompt` 열을 기존 호출 파이프라인에 투입, "
        "응답을 `responses.csv`로 저장 (열: `probe_id`, `response`) — "
        "`kgss_token_check.py --contam-probes`를 쓰면 A와 함께 자동으로 됨\n"
        "3. `uv run kgss_contamination.py score --out .` 실행\n\n"
        "## 판정 기준\n\n"
        "| 프로브 | 오염 신호 | 프레이밍 효과와 구분 |\n|---|---|---|\n"
        "| A | 진짜 출처에서만 모델의 암묵적 분포(로짓)가 실제 가중 분포에 "
        "유의하게 가까워짐(Wasserstein거리 감소) | 가짜 출처에서도 똑같이 가까워지면 "
        "\"공식조사 프레이밍\" 효과일 뿐, 오염 아님 |\n"
        "| B | 진짜 출처에서만 일치도가 유의하게 높음 | 가짜 출처에서도 똑같이 오르면 "
        "\"원문 그대로 재현하라\"는 지시가 정형화된 문체를 유도한 것일 뿐, "
        "이 문서 자체를 기억하는 게 아님 |\n"
        "| C | 순서 정확도가 우연 확률(1/n!)을 크게 초과 | (대조 없음, 보조 근거) |\n"
        "| D | 오래된 회차를 최근 회차보다 잘 맞힘 | (별도 실행) |\n"
        "| E | 진짜 출처에서만 회피율이 낮고 정확도가 높음 | 가짜 출처에서도 자신 있게 "
        "답하면(회피 안 함) 이 채널 자체가 신뢰 못 할 신호 — 진짜/가짜 둘 다 대부분 "
        "회피해야 정상 |\n\n"
        "**A/B는 이제 진짜출처/미명시/가짜출처 3조건**이다. 진짜출처 vs 미명시 차이가 "
        "유의해도, 진짜출처 vs 가짜출처 차이가 유의하지 않으면(둘 다 비슷하게 좋아지면) "
        "오염이 아니라 프레이밍 효과로 봐야 한다 — 이게 이번에 추가한 판정 기준의 핵심.\n",
        encoding="utf-8")
    print(f"\n완료 -> {outdir.resolve()}")


def tost_paired(x: pd.Series, y: pd.Series, d_bound: float = 0.2) -> dict | None:
    """대응표본 동등성검정(TOST, Two One-Sided Tests).

    일반 t검정은 "차이가 있다"를 기각 못 하는 것뿐이지 "차이가 없다"를
    증명하지 않는다(2026-09 팀 논의: "이 정도면 오염 없다고 해도 되지 않나" →
    통계적으로는 그렇게 못 말한다는 점을 명확히 하기 위해 추가). TOST는
    반대로 "진짜-가짜 차이가 무시할 만한 범위(±Δ) 안에 있다"를 적극적으로
    검정한다.

    Δ(동등성 한계)는 diff의 표준편차에 Cohen's dz 단위 기준치(기본 0.2,
    관례적 "작은 효과")를 곱해 원 단위로 환산한다 — 문항별 절대 척도가
    다른 A(Wasserstein거리)와 B(문자열 유사도)에 동일한 기준을 쓸 수 있게
    하는 방법. 양쪽 단측검정(diff > -Δ, diff < +Δ)이 모두 p<0.05여야
    "0.2 표준편차 이내로 동등하다"고 결론 내릴 수 있다 — 표본이 작아
    SE가 크면 동등성도 입증하기 어려워진다(그만큼 큰 표본이 필요하다는 뜻이지,
    작은 표본에서 유의하지 않다고 자동으로 동등한 게 아니다).
    """
    diff = (x - y).dropna()
    n = len(diff)
    if n < 3:
        return None
    sd = diff.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return None
    se = sd / math.sqrt(n)
    bound = d_bound * sd
    mean = diff.mean()
    df = n - 1
    from scipy import stats
    t_low = (mean - (-bound)) / se
    p_low = 1 - stats.t.cdf(t_low, df)   # H0: 참값 <= -Δ
    t_high = (mean - bound) / se
    p_high = stats.t.cdf(t_high, df)     # H0: 참값 >= +Δ
    p_tost = max(p_low, p_high)
    return {"n": n, "mean": mean, "sd": sd, "bound": bound, "d_bound": d_bound,
            "p_low": p_low, "p_high": p_high, "p_tost": p_tost,
            "equivalent": p_tost < 0.05}


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
            L.append(f"\n### {model}")
            if {"출처명시", "출처미명시"} <= set(piv.columns):
                # 양수 = 출처를 밝혔을 때(미명시 대비) 실제 분포에 더 가까워짐
                diff = (piv["출처미명시"] - piv["출처명시"]).dropna()
                L.append(f"- 진짜출처 vs 미명시: 쌍별 차이(미명시-명시) 평균 "
                         f"**{diff.mean():+.4f}** (명시가 더 가까움 {int((diff > 0).sum())}/{len(diff)})")
                try:
                    from scipy import stats
                    t, p = stats.ttest_rel(piv["출처미명시"].dropna(), piv["출처명시"].dropna())
                    L.append(f"  대응표본 t검정: t={t:.3f}, p={p:.4f}")
                except Exception:
                    pass
            if {"출처명시", "가짜출처"} <= set(piv.columns):
                # 이게 핵심 대조: 양수면 진짜 출처가 가짜 출처보다 더 가까움
                # (프레이밍 효과라면 이 차이가 0에 가까워야 함)
                diff2 = (piv["가짜출처"] - piv["출처명시"]).dropna()
                L.append(f"- **진짜출처 vs 가짜출처(프레이밍 효과 통제)**: 쌍별 차이(가짜-진짜) 평균 "
                         f"**{diff2.mean():+.4f}** (진짜가 더 가까움 {int((diff2 > 0).sum())}/{len(diff2)})")
                try:
                    from scipy import stats
                    t2, p2 = stats.ttest_rel(piv["가짜출처"].dropna(), piv["출처명시"].dropna())
                    L.append(f"  대응표본 t검정: t={t2:.3f}, p={p2:.4f}")
                except Exception:
                    pass
                tost = tost_paired(piv["가짜출처"], piv["출처명시"])
                if tost:
                    verdict = ("**동등(무시할 만한 차이) — dz 0.2 이내로 동등성 확인됨**"
                              if tost["equivalent"] else
                              "판정 불가(표본이 더 커야 동등성도 입증 가능)")
                    L.append(f"  동등성검정(TOST, 한계=±{tost['bound']:.4f}, dz={tost['d_bound']}): "
                             f"p={tost['p_tost']:.4f} → {verdict}")
        L.append("\n> **핵심은 두 번째 비교(진짜 vs 가짜출처)입니다.** 진짜출처가 가짜출처보다도 "
                 "유의하게 가까우면 이 조사 자체를 특정해서 기억하고 있다는 뜻입니다. 진짜와 가짜가 "
                 "비슷하게(둘 다 미명시보다) 좋아지면 \"공식조사 프레이밍\" 효과일 뿐 오염이 아닙니다. "
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
            L.append(f"\n### {model}")
            if {"출처명시", "출처미명시"} <= set(piv.columns):
                diff = (piv["출처명시"] - piv["출처미명시"]).dropna()
                L.append(f"- 진짜출처 vs 미명시: 쌍별 차이 평균 **{diff.mean():+.3f}** "
                         f"(양수 {int((diff > 0).sum())}/{len(diff)})")
                try:
                    from scipy import stats
                    t, p = stats.ttest_rel(piv["출처명시"].dropna(),
                                           piv["출처미명시"].dropna())
                    L.append(f"  대응표본 t검정: t={t:.3f}, p={p:.4f}")
                except Exception:
                    pass
            if {"출처명시", "가짜출처"} <= set(piv.columns):
                diff2 = (piv["출처명시"] - piv["가짜출처"]).dropna()
                L.append(f"- **진짜출처 vs 가짜출처(프레이밍 효과 통제)**: 쌍별 차이 평균 "
                         f"**{diff2.mean():+.3f}** (진짜가 더 높음 {int((diff2 > 0).sum())}/{len(diff2)})")
                try:
                    from scipy import stats
                    t2, p2 = stats.ttest_rel(piv["출처명시"].dropna(),
                                            piv["가짜출처"].dropna())
                    L.append(f"  대응표본 t검정: t={t2:.3f}, p={p2:.4f}")
                except Exception:
                    pass
                tost = tost_paired(piv["출처명시"], piv["가짜출처"])
                if tost:
                    verdict = ("**동등(무시할 만한 차이) — dz 0.2 이내로 동등성 확인됨**"
                              if tost["equivalent"] else
                              "판정 불가(표본이 더 커야 동등성도 입증 가능)")
                    L.append(f"  동등성검정(TOST, 한계=±{tost['bound']:.3f}, dz={tost['d_bound']}): "
                             f"p={tost['p_tost']:.4f} → {verdict}")
        L.append("\n> **핵심은 두 번째 비교(진짜 vs 가짜출처)입니다.** \"원문 그대로 재현하라\"는 "
                 "지시 자체가 정형화된 한국어 설문 문체를 유도해서 일치도를 올릴 수 있습니다 — "
                 "가짜출처에서도 똑같이 오르면 그 효과이지 이 문서를 기억하는 게 아닙니다. "
                 "진짜출처가 가짜출처보다 유의하게 높아야 문서 자체를 기억한다고 말할 수 있습니다.\n")

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

    # ---------- E (직접 회상, 서울 논문 방식 + 가짜출처 대조)
    e = d[d["type"] == "E_직접회상"].copy()
    if len(e):
        recs = []
        for _, r in e.iterrows():
            gold = json.loads(r["gold"])
            txt = str(r["response"])
            refused = bool(re.search(r"모름|모르|알 수 없|없습니다", txt))
            pred = parse_percents(txt, len(gold)) if not refused else None
            mae = (float(np.mean(np.abs(np.array(pred) - np.array(gold) * 100)))
                   if pred else np.nan)
            recs.append({"model": r.get("model", "unknown"), "condition": r["condition"],
                        "refused": refused, "mae": mae})
        er = pd.DataFrame(recs)
        L.append("## E. 직접 회상\n")
        L.append("| 모델 | 조건 | n | 회피율 | 회피 안 한 경우 평균 절대오차(%p) |\n"
                 "|---|---|---|---|---|")
        for (model, cond), g in er.groupby(["model", "condition"]):
            L.append(f"| {model} | {cond} | {len(g)} | {g['refused'].mean()*100:.1f}% "
                     f"| {g['mae'].mean():.2f} |")
        for model, gm in er.groupby("model"):
            real_ref = gm[gm["condition"] == "실제"]["refused"]
            fake_ref = gm[gm["condition"] == "가짜"]["refused"]
            if len(real_ref) and len(fake_ref):
                try:
                    from scipy import stats
                    chi2, p, *_ = stats.chi2_contingency([
                        [real_ref.sum(), (~real_ref).sum()],
                        [fake_ref.sum(), (~fake_ref).sum()],
                    ])
                    L.append(f"\n### {model}\n- 회피율 차이(실제 vs 가짜) 카이제곱검정: "
                             f"chi2={chi2:.3f}, p={p:.4f}")
                except Exception:
                    pass
        L.append("\n> **판정**: 진짜 출처에서만 회피율이 낮고(자신 있게 답함) 오차도 작으면 "
                 "오염 쪽에 힘을 싣습니다. **가짜 출처에서도 자신 있게 답하면(회피 안 함) 이 채널 "
                 "자체가 신뢰 못 할 신호입니다** — 존재하지 않는 조사인데도 '모른다'고 말 못 하고 "
                 "있다는 뜻이라, 진짜 출처에서의 자신감도 곧이곧대로 믿기 어려워집니다.\n")

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
