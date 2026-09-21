r"""
3인 태깅 결과 병합 및 일치도 계산

문항유형은 자동분류를 공유한 값이라 참고용 diff만 보여준다.
민감여부(Y/N)가 진짜 독립 판단 대상이라 Cohen's kappa(쌍별) + Fleiss' kappa(3인 전체)를 계산한다.

사용법:
    uv run kgss_agreement.py --files 주희=selection/문항선정_작업지_2023_정.csv \
        태우=selection/문항선정_작업지_2023_태우라벨링.xlsx:태우 \
        다교=selection/문항선정_작업지_2023_HUR.xlsx:다교 \
        --out ./selection

    시트 이름을 안 주면(:태우 생략) xlsx 안에서 민감여부가 가장 많이 채워진 시트를 자동으로 고른다.
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

COLS = ["변수명", "문항내용", "문항유형", "민감여부", "포함"]


def load_tagger(spec: str) -> pd.DataFrame:
    path_part, _, sheet = spec.partition(":")
    path = Path(path_part)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        xl = pd.ExcelFile(path)
        if sheet:
            df = xl.parse(sheet)
        else:
            candidates = [s for s in xl.sheet_names if s != "태깅기준"]
            counts = {s: xl.parse(s)["민감여부"].notna().sum() for s in candidates}
            best = max(counts, key=counts.get)
            print(f"  {path.name}: 시트 자동 선택 -> {best} (민감여부 {counts[best]}행)")
            df = xl.parse(best)
    return df.set_index("변수명")[["문항내용", "문항유형", "민감여부", "포함"]]


def cohen_kappa(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    both = pd.concat([a, b], axis=1).dropna()
    n = len(both)
    if n == 0:
        return float("nan"), 0
    po = (both.iloc[:, 0] == both.iloc[:, 1]).mean()
    pe = sum(
        (both.iloc[:, 0] == c).mean() * (both.iloc[:, 1] == c).mean()
        for c in set(both.iloc[:, 0]) | set(both.iloc[:, 1])
    )
    kappa = 1.0 if pe == 1 else (po - pe) / (1 - pe)
    return kappa, n


def fleiss_kappa(mat: pd.DataFrame) -> tuple[float, int]:
    """mat: 문항 x 태거, 각 셀이 카테고리. 전원 응답한 행만 사용."""
    full = mat.dropna()
    if len(full) == 0:
        return float("nan"), 0
    n_raters = full.shape[1]
    cats = pd.unique(full.values.ravel())
    counts = pd.DataFrame(
        {c: (full == c).sum(axis=1) for c in cats}
    )
    N = len(full)
    P_i = (counts.pow(2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    p_j = counts.sum(axis=0) / (N * n_raters)
    Pe_bar = (p_j ** 2).sum()
    kappa = 1.0 if Pe_bar == 1 else (P_bar - Pe_bar) / (1 - Pe_bar)
    return kappa, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True,
                    help="이름=경로[:시트] 형식. 예: 주희=selection/x.csv")
    ap.add_argument("--out", default="./selection")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    taggers = {}
    print("[1/3] 파일 로드")
    for f in args.files:
        name, _, spec = f.partition("=")
        taggers[name] = load_tagger(spec)
        print(f"  {name}: {len(taggers[name])}행")

    names = list(taggers)
    text = taggers[names[0]]["문항내용"]

    type_df = pd.concat({n: taggers[n]["문항유형"] for n in names}, axis=1)
    sens_df = pd.concat({n: taggers[n]["민감여부"] for n in names}, axis=1)
    incl_df = pd.concat({n: taggers[n]["포함"] for n in names}, axis=1)

    print("\n[2/3] 문항유형 일치도 (자동분류를 다들 그대로 안 썼다면 독립 태깅으로 취급)")
    type_diff = type_df[type_df.nunique(axis=1, dropna=True) > 1]
    print(f"  {len(type_diff)}개 문항에서 문항유형 불일치")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            k, n = cohen_kappa(type_df[names[i]], type_df[names[j]])
            print(f"  {names[i]} vs {names[j]}: kappa={k:.3f} (n={n})")
    type_fk, type_fn = fleiss_kappa(type_df)
    print(f"  Fleiss kappa={type_fk:.3f} (n={type_fn})")

    print("\n[3/3] 민감여부 일치도")
    lines = ["# 3인 태깅 일치도\n"]
    lines.append(f"\n태거: {', '.join(names)}\n")

    lines.append("\n## 문항유형 일치도\n")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            k, n = cohen_kappa(type_df[names[i]], type_df[names[j]])
            lines.append(f"- {names[i]} vs {names[j]}: kappa={k:.3f} (n={n})\n")
    lines.append(f"- Fleiss' kappa(3인): {type_fk:.3f} (n={type_fn})\n")
    lines.append(f"\n불일치 {len(type_diff)}개\n")
    if len(type_diff):
        lines.append("\n| 변수명 | " + " | ".join(names) + " |\n")
        lines.append("|---|" + "---|" * len(names) + "\n")
        for v, row in type_diff.iterrows():
            lines.append(f"| {v} | " + " | ".join(str(row[n]) for n in names) + " |\n")

    lines.append("\n## 민감여부 쌍별 Cohen's kappa\n")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            k, n = cohen_kappa(sens_df[names[i]], sens_df[names[j]])
            line = f"- {names[i]} vs {names[j]}: kappa={k:.3f} (n={n})"
            print(f"  {line}")
            lines.append(line + "\n")

    fk, fn = fleiss_kappa(sens_df)
    line = f"\n## 민감여부 Fleiss' kappa (3인 전원 응답한 {fn}문항)\n\nkappa={fk:.3f}\n"
    print(f"\n  Fleiss kappa={fk:.3f} (n={fn})")
    lines.append(line)

    # 불일치 문항 상세 (조율용)
    disagree = sens_df[sens_df.nunique(axis=1, dropna=True) > 1].copy()
    disagree.insert(0, "문항내용", text.reindex(disagree.index))
    disagree.to_csv(outdir / "agreement_민감여부_불일치.csv", encoding="utf-8-sig")
    lines.append(f"\n## 민감여부 불일치 문항: {len(disagree)}개 -> agreement_민감여부_불일치.csv 참고\n")

    incl_agree = incl_df.fillna("N").apply(lambda r: r.nunique() == 1, axis=1)
    lines.append(f"\n## 포함(최종 30개 후보) 3인 일치: {incl_agree.sum()}개 문항\n")

    # 다수결 병합 — 2인 이상 동의한 값. 전원 갈리면(2인만 응답+불일치 등) 빈 칸으로 남김
    def majority(row):
        vc = row.dropna().value_counts()
        if len(vc) == 0:
            return np.nan
        top = vc[vc == vc.max()].index
        return top[0] if len(top) == 1 else "혼동"

    sens_majority = sens_df.apply(majority, axis=1)
    n_split = (sens_df.nunique(axis=1, dropna=True) > 1).sum()
    lines.append(f"\n## 민감여부 다수결 병합\n"
                 f"2-1 스플릿 {n_split}개 포함 전부 다수결로 병합 (동률은 '혼동')\n"
                 f"병합 결과: Y {(sens_majority=='Y').sum()}개 / N {(sens_majority=='N').sum()}개\n")

    merged = pd.DataFrame({"문항내용": text, "문항유형": taggers[names[0]]["문항유형"]})
    for n in names:
        merged[f"민감여부_{n}"] = sens_df[n]
    merged["민감여부_다수결"] = sens_majority
    merged.to_csv(outdir / "문항선정_병합_2023.csv", encoding="utf-8-sig")

    (outdir / "agreement_result.md").write_text("".join(lines), encoding="utf-8")
    print(f"\n-> {outdir / 'agreement_result.md'}")
    print(f"-> {outdir / 'agreement_민감여부_불일치.csv'}")
    print(f"-> {outdir / '문항선정_병합_2023.csv'} (다수결 병합본, 최종 30개 고를 때 이걸 기준으로)")


if __name__ == "__main__":
    main()
