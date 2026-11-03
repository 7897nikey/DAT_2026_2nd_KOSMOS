r"""
확장 라운드 문항 목록 — 팀 3인 태깅 303문항에서 페르소나에게 물을 수 없는 문항 제외

입력: selection/문항_확정_유형민감_2023.csv (문항유형_다수결, 민감여부_확정)
출력: experiment_expand/items.csv

제외
    ADMIN     조사 관리·가구주 정보 변수 (설문원문 없음, 응답자 태도와 무관)
    PERSONA   페르소나 문장에 이미 들어가는 변수 — 물으면 답이 프롬프트에 있음

사용법:
    uv run kgss_expand_items.py
    uv run kgss_experiment_prompt.py --items experiment_expand/items.csv --levels c0 adult demo \
        --age-grid 22 30 40 50 60 70 80 --out ./experiment_expand
"""

import io
import sys
from pathlib import Path

import pandas as pd

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent
ADMIN = {"HDHEAD", "HHAWAY", "HHEMPLY", "HHMARI", "HHSEX", "INTDATM", "SAMPLEAB"}
PERSONA = {"SEX", "AGE", "EDUC", "MARITAL", "REGION", "URBAN"}


def main():
    f = pd.read_csv(BASE / "selection" / "문항_확정_유형민감_2023.csv", encoding="utf-8-sig")
    drop = f["변수명"].isin(ADMIN | PERSONA)
    print(f"303문항 중 제외 {drop.sum()}개: {sorted(f.loc[drop, '변수명'])}")
    out = f.loc[~drop, ["변수명", "문항유형_다수결", "민감여부_확정"]]
    d = BASE / "experiment_expand"
    d.mkdir(exist_ok=True)
    out.to_csv(d / "items.csv", index=False, encoding="utf-8-sig")
    print(pd.crosstab(out["문항유형_다수결"], out["민감여부_확정"].fillna("-"), margins=True).to_string())
    print(f"-> {d / 'items.csv'}")


if __name__ == "__main__":
    main()
