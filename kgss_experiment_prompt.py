r"""
본실험 프롬프트 생성 — 대조군 4수준 x 인칭·화계 4조건

대조군 4수준 (CLAUDE.md 7절):
    c0     페르소나 없음 — 아무 설명 없이 문항만
    adult  "한국 성인" — 개인정보 없는 고정 문구
    demo   성별+연령만 (persona_sample.csv에서 SEX/AGE만 사용)
    full   6변수 전부 (persona_sample.csv의 기존 persona_당신_하십시오체 등 컬럼 재사용)

c0/adult는 페르소나별 내용이 똑같으므로 문항당 한 번(c0)/화계조합당 한 번(adult)만
만든다. demo/full은 페르소나 표본(같은 응답자 집합)을 그대로 재사용해 조건 간
대응비교(within-persona)가 성립하게 한다.

출력은 kgss_contamination.py의 probes.csv와 같은 형태(probe_id/prompt/codes)라
kgss_token_check.py --contam-probes로 그대로 로짓을 뽑을 수 있다.

사용법:
    uv run kgss_experiment_prompt.py --items selection/final_items.csv --out ./experiment
    (또는 --vars VAR1 VAR2 ... 로 직접 지정, --items보다 우선)
"""

import argparse
import io
import json
import sys
from pathlib import Path

import pandas as pd

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REGISTERS = [("formal", "하십시오체", "입니다"), ("plain", "해라체", "이다")]
ADDRESSEES = ["당신", "이사람"]

STRUCTURAL = {-8.0, -1.0}
DK_COLS = ["legacy_dk", "dk"]
INAP_COLS = ["legacy_inap", "inap"]


def pick_col(dfr: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in dfr.columns:
            return n
    return None


def real_options(var: str, vt: pd.DataFrame) -> list[tuple[float, str]]:
    """variables_worded.csv의 '선택지' 문자열은 DK/비해당 코드를 안 거른
    원본이라(kgss_codebook.py가 그렇게 만듦) 쓰지 않는다. inventory/variables.csv의
    value_labels를 kgss_shortlist.py의 fmt_options()와 같은 기준으로 직접 걸러서
    쓴다 — CLAUDE.md 3절 "모르겠다는 선택지로 제공하지 않음" 원칙."""
    import json as _json
    if var not in vt.index:
        return []
    row = vt.loc[var]
    try:
        vl = _json.loads(row["value_labels"])
    except Exception:
        return []
    dk_col, inap_col = pick_col(vt, DK_COLS), pick_col(vt, INAP_COLS)
    dk = set(_json.loads(row[dk_col])) if dk_col and pd.notna(row[dk_col]) else set()
    inap = set(_json.loads(row[inap_col])) if inap_col and pd.notna(row[inap_col]) else set()
    out = []
    for k, lab in sorted(vl.items(), key=lambda kv: float(kv[0])):
        code = float(k)
        if code in STRUCTURAL or code in dk or code in inap:
            continue
        out.append((code, lab))
    return out


def head_text(addressee: str, subj: str, register: str, ending: str) -> str:
    if addressee == "당신":
        return f"당신은 {subj}{ending}."
    return f"다음은 {subj}에 대한 설명{ending}."


def build_item_text(txt: str, opts: list[tuple[float, str]]):
    codes = [int(c) for c, _ in opts]
    if any(not (1 <= c <= 9) for c in codes) or len(set(codes)) != len(codes):
        return None
    opt_txt = "\n".join(f"{c}. {lab}" for c, (_, lab) in zip(codes, opts))
    return codes, f"문항: {txt}\n\n선택지:\n{opt_txt}\n\n번호로만 답하십시오."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inv", default="./inventory")
    ap.add_argument("--worded", default="./selection/variables_worded.csv")
    ap.add_argument("--persona", default="./persona_sample/persona_sample.csv")
    ap.add_argument("--items", default=None, help="변수명 열이 있는 CSV. 없으면 --vars 필요")
    ap.add_argument("--vars", nargs="+", default=None)
    ap.add_argument("--levels", nargs="+", default=["c0", "adult", "demo", "full"],
                    choices=["c0", "adult", "demo", "full"])
    ap.add_argument("--n-personas", type=int, default=None, help="demo/full에 쓸 페르소나 수 제한 (기본 전체)")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--out", default="./experiment")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.vars:
        var_list = args.vars
    elif args.items:
        var_list = pd.read_csv(args.items)["변수명"].tolist()
    else:
        raise SystemExit("--items 또는 --vars 중 하나는 필요합니다.")

    worded = pd.read_csv(args.worded).set_index("var")
    vt = pd.read_csv(Path(args.inv) / "variables.csv").set_index("var")
    persona = pd.read_csv(args.persona)
    if args.n_personas:
        persona = persona.head(args.n_personas)
    print(f"문항 {len(var_list)}개, 페르소나 {len(persona)}명, 수준 {args.levels}")

    items = {}
    n_skip = 0
    for v in var_list:
        if v not in worded.index:
            print(f"  경고: {v} variables_worded.csv에 없음, 건너뜀")
            continue
        txt = str(worded.at[v, "설문원문"] or "")
        opts = real_options(v, vt)
        built = build_item_text(txt, opts)
        if built is None:
            n_skip += 1
            continue
        items[v] = built
    if n_skip:
        print(f"  선택지 코드가 1~9 범위를 벗어나 {n_skip}개 문항 제외")
    print(f"  사용 가능 문항 {len(items)}개")

    rows = []

    if "c0" in args.levels:
        for v, (codes, item_text) in items.items():
            rows.append({"probe_id": f"C0_{v}", "level": "c0", "var": v, "year": args.year,
                        "addressee": "", "register": "", "persona_id": "",
                        "prompt": item_text, "codes": json.dumps(codes)})

    if "adult" in args.levels:
        for v, (codes, item_text) in items.items():
            for addressee in ADDRESSEES:
                for register, tag, ending in REGISTERS:
                    head = head_text(addressee, "한국 성인", register, ending)
                    rows.append({"probe_id": f"ADULT_{v}_{addressee}_{tag}", "level": "adult",
                                "var": v, "year": args.year, "addressee": addressee,
                                "register": tag, "persona_id": "",
                                "prompt": f"{head}\n\n{item_text}", "codes": json.dumps(codes)})

    if "demo" in args.levels:
        for _, p in persona.iterrows():
            subj = f"{int(p['AGE'])}세 {p['SEX_label']}"
            for v, (codes, item_text) in items.items():
                for addressee in ADDRESSEES:
                    for register, tag, ending in REGISTERS:
                        head = head_text(addressee, subj, register, ending)
                        rows.append({"probe_id": f"DEMO_{v}_{addressee}_{tag}_{p['persona_id']}",
                                    "level": "demo", "var": v, "year": args.year,
                                    "addressee": addressee, "register": tag,
                                    "persona_id": p["persona_id"],
                                    "prompt": f"{head}\n\n{item_text}", "codes": json.dumps(codes)})

    if "full" in args.levels:
        for _, p in persona.iterrows():
            for v, (codes, item_text) in items.items():
                for addressee in ADDRESSEES:
                    for register, tag, ending in REGISTERS:
                        head = p[f"persona_{addressee}_{tag}"]
                        rows.append({"probe_id": f"FULL_{v}_{addressee}_{tag}_{p['persona_id']}",
                                    "level": "full", "var": v, "year": args.year,
                                    "addressee": addressee, "register": tag,
                                    "persona_id": p["persona_id"],
                                    "prompt": f"{head}\n\n{item_text}", "codes": json.dumps(codes)})

    out = pd.DataFrame(rows)
    out_path = outdir / "experiment_probes.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n총 {len(out)}행 -> {out_path}")
    print(out["level"].value_counts().to_string())
    print("\n다음: uv run kgss_token_check.py --contam-probes", out_path, "--out ./experiment --device cuda")


if __name__ == "__main__":
    main()
