r"""
타겟 정렬(Target Alignment) 재채점 — 본실험 로짓 재사용, GPU 불필요

배경 (CLAUDE.md 7.5절): kgss_analyze_logits.py는 c0/adult/demo/full 네 조건
모두에서 Wasserstein 거리를 2023 전체 인구 분포 하나에 대해서만 계산한다.
demo는 성별+연령, full은 6변수 페르소나 정보를 모델에 주는데도 비교 대상은
똑같이 "전체 인구"다(태우님 재실험 보고서
KOSMOS_REEXPERIMENT_BENCHMARK_REPORT_20261010.pdf가 지적한 Target Mismatch).

첫 버전은 타겟과 집계 순서를 동시에 바꿔서 두 효과가 섞였다. 이 버전은 2x2로
분리한다.

                       인구 전체 타겟          매칭 셀 타겟
    평균 먼저(avg)      avg_pop (기존 방식)     avg_aligned
    개별 먼저(ind)      ind_pop                 ind_aligned

    avg_*  예측분포를 먼저 평균 낸 뒤 타겟과 비교. avg_aligned는 같은 매칭 셀에
           속한 페르소나끼리 예측을 평균 내 그 셀 타겟과 비교하고, 셀 크기(페르소나
           수)로 가중평균한다.
    ind_*  페르소나 예측 하나하나를 타겟과 비교한 뒤 거리를 평균.

    타겟 효과 = *_aligned − *_pop  (음수면 매칭 셀 타겟에 더 가까움 = 페르소나
                                     조건화가 맞는 방향으로 움직였다는 뜻)
    집계 효과 = ind_* − avg_*

    집계 효과는 측정 결과가 아니라 수학적으로 정해져 있다. 1차원 Wasserstein은
    첫 인자에 대해 볼록이라, 타겟이 고정이면 평균 예측의 거리가 개별 거리의
    평균보다 항상 작거나 같다(젠슨 부등식). 그래서 첫 버전의 "정렬하니 더
    나빠졌다"는 결과는 집계 순서만으로도 나올 수 있는 결과였다.

    조건 간 비교에는 ind_*를 쓴다. avg_aligned는 매칭 셀 하나에 들어가는 페르소나
    수가 조건마다 다르다(demo는 10셀에 200명, full은 수백 셀에 200명이 흩어짐).
    셀당 페르소나가 많을수록 평균으로 인한 평활이 커지므로 avg_aligned는 조건 간
    공정 비교가 아니다.

매칭 셀 구성 규칙
    c0/adult  페르소나 정보가 없으므로 전체 인구 분포가 곧 매칭 타겟.
    demo      성별×연령. CLAUDE.md 4절 표에서 성별×연령 셀은 전부 n>=20.
    full      성별×연령×학력×혼인×도시성×지역에서 시작해, 그 페르소나의 실제 셀
              인원이 20명 미만이면 지역 → 도시성 → 혼인 → 학력 순으로 변수를 하나씩
              빼며 20명 이상이 될 때까지 백오프(성별×연령은 항상 유지).
              빼는 순서는 CLAUDE.md 4절 셀크기표(학력을 더할 때 12→33셀, 도시성
              33→57, 지역 57→155)를 거꾸로 적용한 설계 선택이다. 혼인은 표에 없어
              학력 다음으로 추정 배치했다. 유일한 정답은 아니다.

사용법:
    uv run kgss_target_alignment.py
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent
EXP = BASE / "experiment"
OUT = EXP / "분석결과"
OUT.mkdir(exist_ok=True)
STRUCT = {-1.0, -8.0}
MIN_CELL_N = 20
AGE_BINS = [17, 29, 39, 49, 59, 200]
AGE_LABELS = ["18-29", "30-39", "40-49", "50-59", "60+"]
FULL_EXTRA = ["EDUC", "MARITAL", "URBAN", "REGION"]
DROP_ORDER = ["REGION", "URBAN", "MARITAL", "EDUC"]  # 먼저 빠지는 변수부터
LEVELS = ["c0", "adult", "demo", "full"]
METRICS = ["avg_pop", "avg_aligned", "ind_pop", "ind_aligned"]

MODELS = {
    "exaone":    dict(mid="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",    file="logit_exaone.csv"),
    "kanana":    dict(mid="kakaocorp/kanana-1.5-2.1b-instruct-2505", file="logit_kanana.csv"),
    "exaone78b": dict(mid="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",    file="logit_exaone78b_fp16.csv"),
    "kanana8b":  dict(mid="kakaocorp/kanana-1.5-8b-instruct-2505",   file="logit_kanana8b_fp16.csv"),
}


def wasserstein(p, q):
    """1차원 Wasserstein 거리(선택지 순서 기준, 간격 1). kgss_analyze_logits.py와 동일 정의."""
    cp, cq = np.cumsum(p), np.cumsum(q)
    return float(np.sum(np.abs(cp - cq)))


def real_codes(series, codes):
    s = pd.to_numeric(series, errors="coerce")
    s = s[~s.isin(STRUCT)]
    return s[s.isin(codes)]


def weighted_dist(sub, s, codes):
    w = sub.loc[s.index, "FINALWT"]
    tot = w.sum()
    if tot <= 0:
        return None
    return np.array([float(w[s == c].sum() / tot) for c in codes])


def matched_real_dist(clean, var, codes, sex, age_grp, extra_vals, level):
    """level에 맞는 변수 집합에서 시작해 20명 미만이면 DROP_ORDER 순서로 백오프.

    반환: (dist 또는 None, 실제 사용한 추가변수 리스트, 셀 raw 인원수)
    """
    active = list(FULL_EXTRA) if level == "full" else []
    while True:
        mask = (clean["SEX"] == sex) & (clean["AGE_GRP"] == age_grp)
        for vn in active:
            mask &= (clean[vn] == extra_vals[vn])
        sub = clean.loc[mask]
        s = real_codes(sub[var], codes) if var in sub.columns else pd.Series(dtype=float)
        n = len(s)
        if n >= MIN_CELL_N or not active:
            if n == 0:
                return None, active, n
            return weighted_dist(sub, s, codes), list(active), n
        drop = next(v for v in DROP_ORDER if v in active)
        active.remove(drop)


def main():
    probes = pd.read_csv(EXP / "experiment_probes.csv", encoding="utf-8-sig")
    persona = pd.read_csv(BASE / "persona_sample" / "persona_sample.csv", encoding="utf-8-sig")
    clean = pd.read_parquet(BASE / "inventory" / "kgss_clean.parquet")
    clean = clean[clean["YEAR"] == 2023].copy()
    clean["AGE_GRP"] = pd.cut(clean["AGE"], AGE_BINS, labels=AGE_LABELS, right=False)
    persona["AGE_GRP"] = pd.cut(persona["AGE"], AGE_BINS, labels=AGE_LABELS, right=False)
    persona_idx = persona.set_index("persona_id")

    # 문항별 전체 인구 타겟 (kgss_analyze_logits.py와 동일 방식)
    wt = clean["FINALWT"].astype(float)
    pop_target = {}
    for var in probes["var"].unique():
        if var not in clean.columns:
            continue
        codes = json.loads(probes.loc[probes["var"] == var, "codes"].iloc[0])
        real = real_codes(clean[var], codes)
        if real.empty:
            continue
        w = wt.loc[real.index].groupby(real).sum()
        p = np.array([w.get(float(c), 0.0) for c in codes])
        pop_target[var] = p / p.sum()

    cache = {}  # (level, persona_id, var) -> (dist, used_vars, n, cell_key)
    backoff_rows = []

    def get_matched(level, pid, var, codes):
        key = (level, pid, var)
        if key in cache:
            return cache[key]
        p = persona_idx.loc[pid]
        extra_vals = {vn: p[vn] for vn in FULL_EXTRA}
        dist, used, n = matched_real_dist(clean, var, codes, p["SEX"], p["AGE_GRP"], extra_vals, level)
        cell_key = (p["SEX"], p["AGE_GRP"], tuple((vn, extra_vals[vn]) for vn in used))
        cache[key] = (dist, used, n, cell_key)
        backoff_rows.append(dict(level=level, persona_id=pid, var=var, n_vars_used=len(used) + 2, cell_n=n))
        return cache[key]

    rows = []
    for tag, info in MODELS.items():
        path = EXP / info["file"]
        if not path.exists():
            print(f"=== {tag}: {path.name} 없음, 건너뜀 ===")
            continue
        L = pd.read_csv(path, encoding="utf-8-sig").set_index("probe_id")
        df = probes.set_index("probe_id").join(L[["model_dist"]], how="inner")
        df["d"] = [np.array(json.loads(x)) for x in df["model_dist"]]
        print(f"=== {tag}: 로짓 {len(df)}건 ===")

        for var, g in df.groupby("var"):
            if var not in pop_target:
                continue
            codes = json.loads(g["codes"].iloc[0])
            tgt = pop_target[var]

            for level, gl in g.groupby("level"):
                if level in ("c0", "adult"):
                    ds = list(gl["d"])
                    avg_pop = wasserstein(np.mean(ds, axis=0), tgt)
                    ind_pop = float(np.mean([wasserstein(d, tgt) for d in ds]))
                    rows.append(dict(model=tag, level=level, var=var, cond="-", n_persona=0,
                                     avg_pop=avg_pop, avg_aligned=avg_pop,
                                     ind_pop=ind_pop, ind_aligned=ind_pop,
                                     n_vars_matched=2.0, n_cells=1, avg_cell_n=np.nan))
                    continue

                gl = gl.copy()
                gl["persona_id"] = gl["persona_id"].astype(int)
                gl["cond"] = gl["addressee"] + "·" + gl["register"]
                for cond, gc in gl.groupby("cond"):
                    ind_pop, ind_al, n_vars, cell_ns = [], [], [], []
                    by_cell = {}  # cell_key -> (매칭 타겟, [예측...])
                    for pid, d in zip(gc["persona_id"], gc["d"]):
                        dist, used, n, cell_key = get_matched(level, int(pid), var, codes)
                        if dist is None:
                            continue
                        ind_pop.append(wasserstein(d, tgt))
                        ind_al.append(wasserstein(d, dist))
                        n_vars.append(len(used) + 2)
                        cell_ns.append(n)
                        by_cell.setdefault(cell_key, (dist, []))[1].append(d)

                    avg_pop = wasserstein(np.mean(list(gc["d"]), axis=0), tgt)
                    if by_cell:
                        parts = [(len(ds), wasserstein(np.mean(ds, axis=0), t)) for t, ds in by_cell.values()]
                        avg_al = sum(k * w for k, w in parts) / sum(k for k, _ in parts)
                    else:
                        avg_al = np.nan
                    rows.append(dict(model=tag, level=level, var=var, cond=cond, n_persona=len(gc),
                                     avg_pop=avg_pop, avg_aligned=avg_al,
                                     ind_pop=float(np.mean(ind_pop)) if ind_pop else np.nan,
                                     ind_aligned=float(np.mean(ind_al)) if ind_al else np.nan,
                                     n_vars_matched=float(np.mean(n_vars)) if n_vars else np.nan,
                                     n_cells=len(by_cell),
                                     avg_cell_n=float(np.mean(cell_ns)) if cell_ns else np.nan))

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "타겟정렬_상세.csv", index=False, encoding="utf-8-sig")

    summ = res.groupby(["model", "level"]).agg(
        **{m: (m, "mean") for m in METRICS},
        n_vars_matched=("n_vars_matched", "mean"),
        n_cells=("n_cells", "mean"),
        avg_cell_n=("avg_cell_n", "mean")).reset_index()
    summ["타겟효과_개별"] = summ["ind_aligned"] - summ["ind_pop"]
    summ["타겟효과_평균"] = summ["avg_aligned"] - summ["avg_pop"]
    summ["집계효과_인구타겟"] = summ["ind_pop"] - summ["avg_pop"]
    summ["level"] = pd.Categorical(summ["level"], LEVELS, ordered=True)
    summ = summ.sort_values(["model", "level"])
    summ.to_csv(OUT / "타겟정렬_요약.csv", index=False, encoding="utf-8-sig")

    print("\n### 2x2 분해 (문항 평균, 낮을수록 실제 분포에 가까움)")
    print("    타겟효과 음수 = 매칭 셀 타겟에 더 가까움 / 집계효과는 수학적으로 항상 0 이상")
    print(summ.round(3).to_string(index=False))

    print("\n### 조건 간 비교 — 개별 먼저(ind) 기준. 조건 간 공정 비교는 이 표로 한다")
    for m in ["ind_pop", "ind_aligned"]:
        pv = summ.pivot(index="model", columns="level", values=m)[LEVELS]
        print(f"\n[{m}]")
        print(pv.round(3).to_string())

    bl = pd.DataFrame(backoff_rows).drop_duplicates(subset=["level", "persona_id", "var"])
    if len(bl):
        print("\n### full 조건 백오프 분포 (매칭에 쓴 변수 수: 2=성별+연령만 ... 6=전부)")
        print(bl[bl.level == "full"]["n_vars_used"].value_counts().sort_index().to_string())

    print(f"\n저장 위치: {OUT}")


if __name__ == "__main__":
    main()
