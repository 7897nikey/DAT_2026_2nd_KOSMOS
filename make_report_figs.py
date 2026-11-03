"""중간보고서 그래프 생성. 입력은 experiment/ 아래 분석 결과 CSV와 배치 게이트 실행 출력(고정값)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "보고서_그림"
OUT.mkdir(exist_ok=True)

FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=FONT_PATH).get_name(),
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
})
C_EX, C_KA = "#1f4e79", "#c55a11"
CONDS = ["당신·하십시오체", "당신·해라체", "이사람·하십시오체", "이사람·해라체"]
SHORT = ["당신/하십시오", "당신/해라", "이사람/하십시오", "이사람/해라"]

# 그림 1: 쌍체 대비 (정규화 정확도 차이, 페르소나 부트스트랩 95% CI)
pair = pd.read_csv(BASE / "experiment/분석결과/조건간_쌍체비교.csv", encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(6.2, 3.0))
labels, ys, y = [], [], 0
for model, col, name in [("exaone", C_EX, "EXAONE"), ("kanana", C_KA, "Kanana")]:
    for _, r in pair[pair.model == model].iterrows():
        ax.errorbar(r.mean_diff, y,
                    xerr=[[r.mean_diff - r.ci_low], [r.ci_high - r.mean_diff]],
                    fmt="o", color=col, capsize=3, ms=4)
        labels.append(f"{name}  {r.contrast.replace(chr(0x2212), chr(45))}")
        ys.append(y)
        y += 1
    y += 0.6
ax.axvline(0, color="#888888", lw=0.8, ls="--")
ax.set_yticks(ys)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("정확도 차이 (문항 평균, 페르소나 부트스트랩 95% CI)")
fig.tight_layout()
fig.savefig(OUT / "fig1_paired_contrasts.png", dpi=220)
plt.close(fig)

# 그림 2: 정규화 정확도 (조건별)
summ = pd.read_csv(BASE / "experiment/분석결과/수준별_요약.csv", encoding="utf-8-sig")
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
for ax, model, col, title in [(axes[0], "exaone", C_EX, "EXAONE 3.5 2.4B"),
                              (axes[1], "kanana", C_KA, "Kanana 1.5 2.1B")]:
    sub = summ[summ.model == model]
    ax.axhline(sub[sub.level == "c0"].norm_acc.iloc[0], color="#777777", lw=0.9, ls=":", label="C0")
    ax.axhline(sub[sub.level == "adult"].norm_acc.iloc[0], color="#777777", lw=0.9, ls="-.", label="adult")
    for k, lev in enumerate(["demo", "full"]):
        vals = [sub[(sub.level == lev) & (sub.cond == c)].norm_acc.iloc[0] for c in CONDS]
        xs = [i + (k - 0.5) * 0.36 for i in range(4)]
        ax.bar(xs, vals, width=0.34, color=col, alpha=0.45 if lev == "demo" else 1.0, label=lev)
    ax.axhline(1.0, color="#c00000", lw=0.8, ls="--")
    ax.set_xticks(range(4))
    ax.set_xticklabels(SHORT, fontsize=7)
    ax.set_title(title, fontsize=9)
axes[0].set_ylabel("정규화 정확도 (셀 최빈 상한 대비)")
axes[0].legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
fig.tight_layout()
fig.savefig(OUT / "fig2_norm_accuracy.png", dpi=220)
plt.close(fig)

# 그림 3: 선택지 순서 효과 - 모델 TV거리 ÷ 인간 TV거리
op = pd.read_csv(BASE / "experiment/order_probe_results.csv", encoding="utf-8-sig")
op["m"] = op.model.str.contains("EXAONE").map({True: "EXAONE", False: "Kanana"})
piv = op.pivot_table(index="pair", columns="m", values="ratio").sort_values("EXAONE")
fig, ax = plt.subplots(figsize=(6.2, 3.6))
yy = list(range(len(piv)))
ax.scatter(piv["EXAONE"], yy, color=C_EX, s=22, label="EXAONE", zorder=3)
ax.scatter(piv["Kanana"], yy, color=C_KA, s=22, marker="s", label="Kanana", zorder=3)
ax.axvline(1.0, color="#333333", lw=0.9)
ax.set_xscale("log")
ax.set_yticks(yy)
ax.set_yticklabels(piv.index, fontsize=7.5)
ax.set_xlabel("모델 TV거리 ÷ 인간 TV거리 (로그 척도, 1 = 인간과 같은 크기)")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "fig3_order_ratio.png", dpi=220)
plt.close(fig)

# 그림 4: 배치 게이트 (처리 시간과 한 건씩 대비 최대 차이). 2026-10-06 편집기 세션 1차 실행 값
bs = ["한 건씩", "배치 8", "배치 16", "배치 32"]
ex_t = [46.0, 41.8, 46.1, 47.2]
ka_t = [45.6, 40.5, 42.1, 43.4]
ex_d = [1.60e-5, 1.63e-5, 1.81e-5]
ka_d = [2.76e-5, 3.38e-5, 6.02e-5]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.7))
x = list(range(4))
a1.bar([i - 0.2 for i in x], ex_t, width=0.4, color=C_EX, label="EXAONE")
a1.bar([i + 0.2 for i in x], ka_t, width=0.4, color=C_KA, label="Kanana")
a1.set_xticks(x)
a1.set_xticklabels(bs, fontsize=7.5)
a1.set_ylabel("200건 처리 시간 (초)")
a1.set_ylim(0, 60)
a1.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=2)
a2.plot([1, 2, 3], ex_d, "o-", color=C_EX, label="EXAONE")
a2.plot([1, 2, 3], ka_d, "s-", color=C_KA, label="Kanana")
a2.axhline(1e-3, color="#c00000", lw=0.9, ls="--")
a2.text(1.0, 1.2e-3, "게이트 기준 1e-3", color="#c00000", fontsize=7)
a2.set_yscale("log")
a2.set_ylim(1.5e-5, 1.5e-3)
a2.set_yticks([2e-5, 5e-5, 1e-4, 1e-3])
a2.set_yticklabels(["2e-5", "5e-5", "1e-4", "1e-3"])
a2.minorticks_off()
a2.set_xticks([1, 2, 3])
a2.set_xticklabels(["배치 8", "배치 16", "배치 32"], fontsize=7.5)
a2.set_ylabel("한 건씩 대비 최대 차이 (로그)")
a2.legend(frameon=False, fontsize=7.5, loc="center right")
fig.tight_layout()
fig.savefig(OUT / "fig4_batch_gate.png", dpi=220)
plt.close(fig)

print("saved:", sorted(p.name for p in OUT.glob("*.png")))
