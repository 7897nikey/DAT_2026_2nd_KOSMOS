r"""
문항 선정 작업지 생성

305개 후보를 3인이 독립 태깅할 수 있는 xlsx로 정리한다.
자동으로 채우는 정보와 사람이 판단할 열을 분리한다.

자동 채움:
    변수명 / 문항내용 / 선택지 / 응답수 / eta2 / eta2_구간 / 배터리 / AB분할표본
사람 판단 (빈 열, Seoul Table 3 방식 — 문항유형을 먼저 나누고 그 안에서
민감성을 정의한다. CLAUDE.md 3절 미결 "민감 문항 선정 기준 문서화",
운영진 Q6 대응. 2026-09-11 개정: 기존 "유형(사실/태도/민감)" 1열 구조를
2열(문항유형 + 민감여부)로 분리, 2026-09-11 개정: 타이핑 부담을 줄이려고
"패턴화된 태도·차별"/"가치관·규범"을 각각 "차별"/"가치관"으로 축약 — 전체
정의는 "태깅기준" 시트에 그대로 있다. 드롭다운(데이터 유효성 검사)으로
정확히 이 값들만 고를 수 있게 해서 철자가 갈리는 걸 막는다 — 나중에
kgss_agreement.py로 3인 일치도를 계산하려면 값이 정확히 같아야 한다):
    문항유형    사실 / 차별 / 가치관   (드롭다운)
    민감여부    Y / N                  (드롭다운, 문항유형이 "사실"이면 비워둠)
    포함        Y / N                  (드롭다운)
    메모

배터리 탐지: 변수명 끝의 숫자를 떼어낸 어간이 같으면 같은 문항군으로 본다.
    CESD1..CESD11 -> CESD,  PERTRT1..PERTRT17 -> PERTRT
    같은 배터리에서 여러 문항을 뽑으면 조건-타깃 누출이 생기므로 표시해 둔다.

AB 분할표본 탐지: 어간이 같고 접미가 A/B인 쌍.
    SAMPTHOUA / SAMPTHOUB23,  ELEFRAUDA / ELEFRAUDB
    문구 효과의 인간 기준값이 데이터에 이미 있으므로 별도 축으로 쓴다.

사용법:
    uv pip install openpyxl
    uv run kgss_shortlist.py --inv .\inventory --out .\selection --year 2023
"""

import argparse
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment
from openpyxl.worksheet.datavalidation import DataValidation

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":  # 다른 kgss_*.py가 import할 때 이중 래핑 방지
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

STRUCTURAL = {-8.0, -1.0}
TRAILING_NUM = re.compile(r"\d+$")
AB_SUFFIX = re.compile(r"^(?P<stem>.+?)(?P<ab>[AB])(?P<yr>\d{2})?$")

# variables.csv 버전별 열 이름 (kgss_codebook.py와 동일한 패턴 —
# CLAUDE.md 6절 "두 스크립트 간 열 이름 불일치" 함정 참고)
DK_COLS = ["legacy_dk", "dk"]
INAP_COLS = ["legacy_inap", "inap"]


def pick_col(dfr: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in dfr.columns:
            return n
    return None


def battery_stem(var: str) -> str:
    """CESD11 -> CESD, GOVSPD3 -> GOVSPD, POSGEN10 -> POSGEN"""
    s = TRAILING_NUM.sub("", var)
    return s if len(s) >= 3 else var


def fmt_options(vl_json: str, dk_json: str, inap_json: str) -> str:
    try:
        vl = json.loads(vl_json)
    except Exception:
        return ""
    dk = set(json.loads(dk_json or "[]"))
    inap = set(json.loads(inap_json or "[]"))
    parts = []
    for k, lab in sorted(vl.items(), key=lambda kv: float(kv[0])):
        code = float(k)
        if code in STRUCTURAL or code in dk or code in inap:
            continue
        parts.append(f"{int(code) if code == int(code) else code}={lab}")
    return " | ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inv", default="./inventory")
    ap.add_argument("--out", default="./selection")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--worded", default="./selection/variables_worded.csv",
                    help="kgss_codebook.py가 만든 설문 원문 파일. wave_items.csv의 "
                         "label은 .sav 변수라벨이라 문항별로 축약 정도가 들쭉날쭉해서 "
                         "(어떤 건 전체 질문, 어떤 건 3~4단어 태그) 태깅용 문항내용은 "
                         "여기서 우선 가져온다.")
    ap.add_argument("--taggers", nargs="+", default=["주희", "태우", "다교"])
    ap.add_argument("--type-classification", default="./selection/문항유형_classification.csv",
                    help="문항유형(사실/차별/가치관) 사전 분류 CSV. 있으면 자동으로 채우고 "
                         "민감여부만 사람이 판단한다.")
    args = ap.parse_args()

    inv, outdir = Path(args.inv), Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    wave = pd.read_csv(inv / "wave_items.csv")
    vt = pd.read_csv(inv / "variables.csv")
    eta_path = inv / "eta2.csv"
    eta = pd.read_csv(eta_path) if eta_path.exists() else pd.DataFrame(columns=["item", "eta2"])
    worded_path = Path(args.worded)
    worded = (pd.read_csv(worded_path).set_index("var")["설문원문"]
             if worded_path.exists() else pd.Series(dtype=str))

    cand = wave[wave["is_item_cand"]].copy()
    print(f"후보 {len(cand)}개")

    # 문항내용: 설문원문(코드북 PDF에서 추출한 실제 질문 전문)을 우선 쓰고,
    # 코드북에 없는 변수(가구 구성 등 조사원 기록 항목)만 .sav 라벨로 대체한다.
    fallback = []
    def pick_text(v: str, label: str) -> str:
        w = worded.get(v)
        if isinstance(w, str) and w.strip():
            return w.strip()
        fallback.append(v)
        return label
    cand["문항내용"] = [pick_text(v, lab) for v, lab in zip(cand["var"], cand["label"])]
    if fallback:
        print(f"설문원문 없어 라벨로 대체한 문항 {len(fallback)}개: {fallback}")

    info = vt.set_index("var")
    dk_col = pick_col(vt, DK_COLS)
    inap_col = pick_col(vt, INAP_COLS)
    cand["선택지"] = [
        fmt_options(
            info.at[v, "value_labels"] if v in info.index else "{}",
            info.at[v, dk_col] if dk_col and v in info.index else "[]",
            info.at[v, inap_col] if inap_col and v in info.index else "[]",
        )
        for v in cand["var"]
    ]

    cand = cand.merge(eta[["item", "eta2"]], left_on="var", right_on="item", how="left")
    cand.drop(columns=["item"], inplace=True, errors="ignore")

    # eta2 3분위
    q = cand["eta2"].dropna()
    if len(q) >= 10:
        lo, hi = q.quantile([1 / 3, 2 / 3])
        cand["eta2_구간"] = pd.cut(
            cand["eta2"], [-1, lo, hi, 2], labels=["저", "중", "고"]
        )
    else:
        cand["eta2_구간"] = np.nan

    # 배터리
    cand["배터리"] = cand["var"].map(battery_stem)
    size = cand["배터리"].value_counts()
    cand["배터리크기"] = cand["배터리"].map(size)

    # AB 분할표본 쌍
    stems = {}
    for v in cand["var"]:
        m = AB_SUFFIX.match(v)
        if m:
            stems.setdefault(m.group("stem"), []).append((m.group("ab"), v))
    ab_pairs = {}
    for stem, lst in stems.items():
        sides = {ab for ab, _ in lst}
        if {"A", "B"} <= sides:
            for _, v in lst:
                ab_pairs[v] = stem
    cand["AB분할표본"] = cand["var"].map(ab_pairs).notna().map({True: "Y", False: ""})
    n_ab = int((cand["AB분할표본"] == "Y").sum())
    print(f"AB 분할표본 문항 {n_ab}개")

    # wave_items.csv는 --target-year 시점에 이미 단일 연도로 계산돼 있어
    # 연도별 열(y2023 등)이 아니라 answer_rate 하나뿐이다.
    # label(.sav 변수라벨)도 참고용으로 남겨둔다 — 축약 정도가 문항마다
    # 달라서(8절 참고), 문항내용(설문원문)과 대조해서 봐야 할 때가 있다.
    sheet = cand[
        ["var", "문항내용", "label", "선택지", "n_options", "answer_rate", "eta2",
         "eta2_구간", "배터리", "배터리크기", "AB분할표본", "ordinal_guess"]
    ].rename(
        columns={
            "var": "변수명",
            "label": "라벨(참고)",
            "n_options": "선택지수",
            "answer_rate": "응답률",
            "ordinal_guess": "순서형",
        }
    )
    sheet["응답률"] = sheet["응답률"].round(3)
    type_path = Path(args.type_classification)
    if type_path.exists():
        type_map = pd.read_csv(type_path).set_index("변수명")["문항유형"]
        sheet["문항유형"] = sheet["변수명"].map(type_map).fillna("")
        print(f"문항유형 사전 분류 적용: {type_path} ({len(type_map)}개)")
    else:
        sheet["문항유형"] = ""   # 사실 / 차별 / 가치관 (드롭다운, 태깅기준 시트에 전체 정의)
    sheet["민감여부"] = ""   # Y / N (문항유형이 "사실"이면 비워둘 것)
    sheet["포함"] = ""       # Y / N
    sheet["메모"] = ""
    sheet = sheet.sort_values(["배터리크기", "배터리", "변수명"], ascending=[False, True, True])

    xlsx = outdir / f"문항선정_작업지_{args.year}.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        for name in args.taggers:
            sheet.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            widths = {"A": 14, "B": 60, "C": 30, "D": 55, "E": 9, "F": 9,
                      "G": 9, "H": 9, "I": 14, "J": 10, "K": 12, "L": 9,
                      "M": 20, "N": 10, "O": 8, "P": 30}
            for col, w in widths.items():
                ws.column_dimensions[col].width = w
            for row in ws.iter_rows(min_row=2, max_row=ws.max_row,
                                    min_col=2, max_col=4):
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
            ws.freeze_panes = "B2"

            # 드롭다운 — 직접 타이핑하면 "차별"/"차별문항"처럼 3인이 철자를
            # 다르게 쓸 수 있어서, 정확히 이 값만 고르게 강제한다.
            n_rows = len(sheet)
            dv_type = DataValidation(type="list", formula1='"사실,차별,가치관"',
                                     allow_blank=True, showDropDown=False)
            dv_sens = DataValidation(type="list", formula1='"Y,N"',
                                     allow_blank=True, showDropDown=False)
            dv_incl = DataValidation(type="list", formula1='"Y,N"',
                                     allow_blank=True, showDropDown=False)
            ws.add_data_validation(dv_type)
            ws.add_data_validation(dv_sens)
            ws.add_data_validation(dv_incl)
            dv_type.add(f"M2:M{n_rows + 1}")
            dv_sens.add(f"N2:N{n_rows + 1}")
            dv_incl.add(f"O2:O{n_rows + 1}")

        guide = pd.DataFrame(
            {
                "항목": [
                    "① 문항유형 입력값: 사실",
                    "① 문항유형 입력값: 차별  (전체 정의: 패턴화된 태도·차별 문항)",
                    "① 문항유형 입력값: 가치관  (전체 정의: 가치관·규범 문항)",
                    "② 민감여부 (문항유형이 ①에서 정해진 뒤에만 판단)",
                    "  판정 기준 1",
                    "  판정 기준 2",
                    "  판정 기준 3",
                    "  판정 예시 (민감=Y)",
                    "  판정 예시 (민감=N)",
                    "포함", "배터리크기", "AB분할표본", "eta2_구간",
                ],
                "정의": [
                    "본인의 객관적 사실·행동을 묻는 문항 (투표 여부, 병원 방문 횟수, 가입 단체). "
                    "민감여부는 판단하지 않고 비워둔다 — 사회적 바람직성이 개입할 태도 표현 자체가 없음.",
                    "특정 사회집단(이민자·여성/남성·성소수자·장애인·특정 정당 지지자 등)에 대한 태도나 "
                    "그 집단과 관련된 차별적·배제적 판단을 묻는 문항. 서울 논문(Kim, Park & Suh 2026) "
                    "Table 3의 첫 축과 동일한 구분.",
                    "특정 집단을 지목하지 않는 일반적 가치·신념·규범을 묻는 문항 (가족관, 정부 역할에 "
                    "대한 일반적 믿음, 삶의 만족도, 사회 신뢰 등).",
                    "①에서 '차별' 또는 '가치관'으로 분류된 문항만 Y/N 판단. "
                    "아래 기준 중 하나라도 해당하면 Y.",
                    "문항 문구만 보고도 '사회적으로 더 바람직한' 응답 방향이 추론 가능한가",
                    "특정 집단(성별·연령·이민자·장애인·성소수자·종교·정치성향 등)에 대한 차별적/배제적 "
                    "태도를 직접 묻는가",
                    "응답이 공개적으로 밝혀질 경우 사회적 비난·평판 손상 가능성이 있는가",
                    "'이민자가 범죄를 늘린다는 데 동의하십니까' — 부정 응답이 사회적으로 바람직해 보임 → Y",
                    "'여가 시간에 주로 무엇을 하십니까' — 바람직한 방향이 없는 개인 선호 → N",
                    "최종 30개에 넣을 후보면 Y. 각자 40개 내외로 표시할 것",
                    "같은 어간을 공유하는 문항 수. 같은 배터리에서 2개 이상 뽑으면 조건-타깃 누출 위험",
                    "같은 내용을 다른 문구로 물은 A/B형이 존재. 문구 효과의 인간 기준값이 있음",
                    "인구통계 설명력 3분위. 고=예측 가능성 높음, 저=인구통계만으로는 불가",
                ],
            }
        )
        guide.to_excel(xw, sheet_name="태깅기준", index=False)
        ws = xw.sheets["태깅기준"]
        ws.column_dimensions["A"].width = 30
        ws.column_dimensions["B"].width = 95
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")

    print(f"-> {xlsx}")
    print("\n문항유형은 자동 분류돼 있음 — 이상하면 드롭다운으로 고치세요.")
    print("민감여부만 각자 독립적으로 태깅한 뒤 kgss_agreement.py로 일치도를 계산하세요.")

    # 요약 통계
    print("\n[배터리 상위 10]")
    print(size.head(10).to_string())
    if n_ab:
        print("\n[AB 분할표본 쌍]")
        for v, stem in sorted(ab_pairs.items()):
            lab = cand.loc[cand["var"] == v, "label"].iloc[0]
            print(f"  {v:16s} {lab[:50]}")


if __name__ == "__main__":
    main()