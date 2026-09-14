r"""
인칭×화계 조작 파일럿

성분③(내용 편향/사회적 바람직성)을 언어적으로 조작하는 실험의 소규모 파일럿.
민감 문항(SEXROLE 배터리)에 대해 페르소나를 2인칭(당신)으로 부르는지 3인칭
(이 사람)으로 서술하는지, 문장 종결을 하십시오체로 하는지 해라체로 하는지를
교차한 4조건 프롬프트를 만든다.

    A  당신 × 하십시오체   (기존 표준 방식과 가장 가까운 기준 조건)
    B  당신 × 해라체
    C  이 사람 × 하십시오체
    D  이 사람 × 해라체     (이론적으로 가장 "거리를 둔" 조건)

인칭축을 화자(나는~)가 아니라 "페르소나를 청자로 부르는지 vs 관찰 대상으로
서술하는지"로 잡은 이유: 최종 응답이 숫자 하나뿐이라 화자 인칭은 출력에
드러나지 않는다. Chapala et al. (2025)의 "삼인칭 재구성"도 같은 방식(응답자를
2인칭이 아니라 3인칭으로 지칭)이다.

로짓 추출은 새로 안 만들고 kgss_token_check.py --contam-probes를 그대로
재사용한다 — probes.csv에 codes/gold 열만 있으면 오염 프로브인지 이 파일럿
인지 안 가리고 첫 토큰 로짓을 뽑는다(part_d_contam이 type 값과 무관하게
codes가 있는 행을 전부 처리하도록 이미 일반화돼 있음).

사용법
    uv run kgss_register_pilot.py build --inv .\inventory \
        --worded .\selection\variables_worded.csv --out .\register_pilot

    # Colab: 로짓 추출 (오염 점검과 같은 인프라, 출력 폴더만 다르게)
    !python kgss_token_check.py --contam-probes register_pilot/probes.csv \
        --out register_pilot --device cuda --no-contam-generate

    uv run kgss_register_pilot.py score --out .\register_pilot
"""

import argparse
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":  # 다른 kgss_*.py가 import할 때 이중 래핑 방지
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from kgss_contamination import parse_options, weighted_dist  # noqa: E402

# 항상 같은 페르소나로 고정 (kgss_token_check.py의 진단 예시와 동일 — 비교 가능성)
PERSONA_YOU_FORMAL = "당신은 45세 남성입니다. 고등학교를 졸업했고 인천에 거주하며 기혼입니다."
PERSONA_YOU_PLAIN = "당신은 45세 남성이다. 고등학교를 졸업했고 인천에 거주하며 기혼이다."
PERSONA_HE_FORMAL = "다음은 45세 남성에 대한 설명입니다. 고등학교를 졸업했고 인천에 거주하며 기혼입니다."
PERSONA_HE_PLAIN = "다음은 45세 남성에 대한 설명이다. 고등학교를 졸업했고 인천에 거주하며 기혼이다."

# 문항마다 판단 동사쌍이 다르다(찬성/반대, 동의/반대 등) — "찬성 또는 반대"로
# 하드코딩하면 SEXATT3처럼 "옳다/옳지 않다"로 묻는 문항에서 지시문과 선택지가
# 안 맞는 문제가 생긴다(실제로 한 번 겪음). 원문에서 동사쌍을 직접 뽑는다.
VERB_PAT = re.compile(r"(\S+?)\s*또는\s*(\S+?)하(?:십니까|시는지)")


def extract_verbs(txt: str) -> tuple[str, str]:
    """'...어느 정도 찬성 또는 반대하십니까?' -> ('찬성', '반대').
    패턴이 없는 문항(예: SEXATT3처럼 '옳다/옳지 않다'로 묻는 경우)은 이 함수로
    못 돌리므로 DEFAULT_ITEMS에서 애초에 뺐다 — 못 찾으면 폴백 대신 예외를
    던져서 조용히 틀린 프롬프트가 만들어지는 걸 막는다.
    """
    m = VERB_PAT.search(txt)
    if not m:
        raise ValueError(f"판단 동사쌍을 못 찾음(문항 지시문 확인 필요): {txt[:60]!r}")
    return m.group(1), m.group(2)


CONDITIONS = [
    ("A_당신_하십시오체", PERSONA_YOU_FORMAL,
     lambda v1, v2: f"다음 의견에 대해 얼마나 {v1} 또는 {v2}하십니까?", "번호로만 답하십시오."),
    ("B_당신_해라체", PERSONA_YOU_PLAIN,
     lambda v1, v2: f"다음 의견에 대해 얼마나 {v1} 또는 {v2}하는가?", "번호로만 답하라."),
    ("C_이사람_하십시오체", PERSONA_HE_FORMAL,
     lambda v1, v2: f"이 사람은 다음 의견에 대해 얼마나 {v1} 또는 {v2}하겠습니까?", "번호로만 답하십시오."),
    ("D_이사람_해라체", PERSONA_HE_PLAIN,
     lambda v1, v2: f"이 사람은 다음 의견에 대해 얼마나 {v1} 또는 {v2}하겠는가?", "번호로만 답하라."),
]

# SEXROLE 배터리는 2023년엔 조사 안 됨(다른 회차 문항) — 2023 후보 중 민감
# 유형으로 실제 조사된 문항으로 바꿨다. 넷 다 "한국에서 살려고 온 외국인에
# 대하여... 어느 정도 [찬성/동의] 또는 반대하십니까?... (실제 진술)" 구조가
# 동일해서 item_statement()의 "..." 분리와 extract_verbs가 안전하게 먹힌다
# (NOIMMIGR는 진술이 "..."로 안 분리되고 질문 안에 embedded돼 있어 지시문이
# 중복되는 걸 확인하고 뺐다). 전부 이민자 태도 문항(주제 일관성).
DEFAULT_ITEMS = ["IMMCRIME", "IMMECON", "IMMJOBS", "KORORFOR"]

# 값 라벨에 이 패턴이 있으면 DK/비해당 취급 — 실질 선택지가 아니므로 제외.
# 문항마다 표현이 달라서("선택할 수 없음", "DK/Refusal" 등) 넓게 잡는다.
DK_LABEL_PAT = ("모르겠다", "무응답", "비해당", "모름", "선택할 수 없음",
               "선택 불가", "dk", "refusal", "n/a")


def is_dk_label(lab: str) -> bool:
    low = lab.lower()
    return any(p in low for p in DK_LABEL_PAT)


def clean_options(raw: str) -> list[tuple[int, str]]:
    """parse_options 결과에서 DK/비해당 라벨을 뺀다.

    선택지 값이 8=모르겠다/무응답, 9=비해당처럼 라벨로 명시돼 있는 경우만
    걸러낸다 — CLAUDE.md 6절 함정("라벨 어휘만으로 DK 판정"은 위험)과 달리
    여기는 코드값이 아니라 라벨 문자열 자체가 명시적으로 DK/비해당이라고
    말하는 경우만 제외하므로 안전하다.
    """
    opts = parse_options(raw)
    return [(int(c), lab) for c, lab in opts if not is_dk_label(lab)]


def item_statement(txt: str) -> str:
    """'귀하는 ...하십니까?... 실제 진술' -> '실제 진술'만 뽑는다."""
    if "..." in txt:
        return txt.split("...", 1)[1].strip()
    return txt.strip()


def build(args):
    inv, outdir = Path(args.inv), Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(inv / "kgss_clean.parquet")
    worded = pd.read_csv(args.worded).set_index("var")
    sub = df[df["YEAR"] == args.year]

    rows = []
    print(f"[1/2] 문항 {len(args.items)}개 x 조건 {len(CONDITIONS)}개")
    for v in args.items:
        if v not in worded.index:
            print(f"      {v}: variables_worded.csv에 없음, 건너뜀")
            continue
        raw_txt = str(worded.at[v, "설문원문"] or "")
        txt = item_statement(raw_txt)
        if txt.rstrip().endswith(("?", "까", "니까")):
            # "..." 구분자가 없어 원문 전체(질문까지)가 그대로 반환된 경우 —
            # 그대로 쓰면 우리 지시문 뒤에 원문 질문이 또 붙어 중복된다.
            print(f"      {v}: 진술을 못 분리함(질문 전체가 반환됨), 건너뜀 — {txt[:50]!r}")
            continue
        try:
            v1, v2 = extract_verbs(raw_txt)
        except ValueError as e:
            print(f"      {v}: {e}, 건너뜀")
            continue
        opts = clean_options(worded.at[v, "선택지"])
        codes = [c for c, _ in opts]
        if any(not (1 <= c <= 9) for c in codes) or len(set(codes)) != len(codes):
            print(f"      {v}: 선택지 코드가 1~9 범위를 벗어나 건너뜀 ({codes})")
            continue
        dist = weighted_dist(sub, v, codes)
        if dist is None:
            print(f"      {v}: 실제 가중 분포 계산 불가(표본 부족), 건너뜀")
            continue
        opt_txt = "\n".join(f"{c}. {lab}" for c, lab in opts)

        for cond, persona, ask_fn, instr in CONDITIONS:
            ask = ask_fn(v1, v2)
            rows.append({
                "probe_id": f"R_{v}_{cond}",
                "type": "R_인칭화계",
                "condition": cond,
                "var": v, "year": args.year,
                "prompt": f"{persona}\n{ask}\n\n{txt}\n\n{opt_txt}\n\n{instr}",
                "codes": json.dumps(codes),
                "gold": json.dumps(dist),
            })
        print(f"      {v}: {txt[:40]}...")

    p = pd.DataFrame(rows)
    p.to_csv(outdir / "probes.csv", index=False, encoding="utf-8-sig")
    print(f"[2/2] 프로브 {len(p)}건 -> {outdir / 'probes.csv'}")
    if len(p):
        print(p.groupby(["var", "condition"]).size().unstack(fill_value=0))
    else:
        print("      확정된 문항이 없습니다 — --items로 2023년에 실제 조사된 "
              "문항을 지정하세요.")


def score(args):
    outdir = Path(args.out)
    probes = pd.read_csv(outdir / "probes.csv")
    lg_path = outdir / "logit_responses.csv"
    if not lg_path.exists():
        print(f"{lg_path}가 없습니다. "
              f"kgss_token_check.py --contam-probes {outdir}\\probes.csv --out {outdir}"
              f" 로 먼저 로짓을 뽑으세요.")
        return
    lg = pd.read_csv(lg_path)
    d = probes.merge(lg, on="probe_id", how="inner")
    print(f"응답 {len(d)}건 / 프로브 {len(probes)}건\n")

    recs = []
    for _, r in d.iterrows():
        codes = np.array(json.loads(r["codes"]), dtype=float)
        gold = np.array(json.loads(r["gold"]), dtype=float)
        pred = np.array(json.loads(r["model_dist"]), dtype=float)
        n = len(gold)
        try:
            from scipy.stats import wasserstein_distance
            w = float(wasserstein_distance(range(n), range(n), u_weights=pred, v_weights=gold))
        except Exception:
            w = float(np.sum(np.abs(np.cumsum(pred) - np.cumsum(gold))))
        recs.append({
            "model": r.get("model", "unknown"), "var": r["var"], "condition": r["condition"],
            "wasserstein": w,
            "기댓값_모델": float((pred * codes).sum()),
            "기댓값_실제": float((gold * codes).sum()),
            "mass": r.get("mass", np.nan),
        })
    rr = pd.DataFrame(recs)

    L = ["# 인칭×화계 파일럿 결과\n",
         "> 표본이 문항 4개뿐이라 유의성 검정은 안 하고, 방향성만 본다. "
         "기댓값은 코드 가중 평균(코드가 클수록 '반대' 쪽 — 문항별 선택지 참고).\n"]
    for model, gm in rr.groupby("model"):
        L.append(f"## {model}\n")
        piv = gm.pivot_table(index="var", columns="condition",
                             values="기댓값_모델", aggfunc="mean")
        cond_order = [c for c, *_ in CONDITIONS if c in piv.columns]
        piv = piv[cond_order]
        L.append("### 조건별 기댓값 (모델이 암묵적으로 갖는 평균 응답)\n")
        L.append(piv.round(3).to_markdown())
        L.append("")
        real = gm.groupby("var")["기댓값_실제"].mean().round(3)
        L.append(f"실제 KGSS 기댓값: {real.to_dict()}\n")

        wpiv = gm.pivot_table(index="var", columns="condition",
                              values="wasserstein", aggfunc="mean")[cond_order]
        L.append("### 조건별 Wasserstein거리 (실제 분포 대비, 낮을수록 가까움)\n")
        L.append(wpiv.round(3).to_markdown())
        L.append("")

        if "A_당신_하십시오체" in piv.columns and "D_이사람_해라체" in piv.columns:
            shift = (piv["D_이사람_해라체"] - piv["A_당신_하십시오체"]).round(3)
            L.append(f"**A→D 기댓값 변화(양수=반대 쪽으로, 음수=찬성 쪽으로 이동)**: "
                     f"{shift.to_dict()}\n")

    text = "\n".join(L)
    (outdir / "register_pilot_result.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\n완료 -> {outdir / 'register_pilot_result.md'}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    b = sp.add_parser("build")
    b.add_argument("--inv", default="./inventory")
    b.add_argument("--worded", default="./selection/variables_worded.csv")
    b.add_argument("--out", default="./register_pilot")
    b.add_argument("--year", type=int, default=2023)
    b.add_argument("--items", nargs="+", default=DEFAULT_ITEMS)
    b.set_defaults(func=build)

    s = sp.add_parser("score")
    s.add_argument("--out", default="./register_pilot")
    s.set_defaults(func=score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
