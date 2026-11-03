r"""
성분④ 형식 민감도: 선택지 순서 효과의 모델 쪽 측정 (C0 조건)

인간 기준선(selection/ab_pairs.csv)은 같은 문항을 선택지 배열만 바꿔 두 분할표본에
물은 결과다. 이 스크립트는 같은 설정을 모델에 재현한다.

  - 대상: ab_pairs.csv에서 순서역전=True인 쌍 (문구 동일, 선택지 집합 동일, 배열만 역전)
  - 정순(forward): A 변수의 선택지를 원래 배열대로 1..k 번호 부여
  - 역순(reverse): 같은 선택지를 거꾸로 1..k 번호 부여
  - 두 배열 모두 같은 문구, C0(페르소나 없음), 같은 prefill 프롬프트
  - 각 배열의 번호별 첫 토큰 확률을 선택지 토큰끼리만 재정규화하고,
    선택지 내용 기준으로 되돌려 비교

지표
  - TV거리(정순 대 역순, 내용 기준): 0.5 * sum |P_fwd(c) - P_rev(c)|
    인간 쪽 TV거리(ab_pairs.csv 효과_TV거리)와 같은 정의
  - 배율 = 모델 TV / 인간 TV
  - 위치 선호 지수: 제시 위치를 0(첫 번째)~1(마지막)로 정규화한 평균 선택 위치.
    두 배열 평균이 0.5보다 크면 뒤쪽 선택지 선호, 작으면 앞쪽 선호
  - mass: 선택지 번호 토큰에 실린 확률질량(낮으면 형식 이탈이므로 신뢰도 낮음)

사용법
  uv run --with torch --with transformers==4.48.3 --with accelerate kgss_order_probe.py --device cpu

모델 고정
  - EXAONE: --revision 8e6fc27 (원격 코드 버전 고정)
  - Kanana: 최신 저장소에 해당 해시가 없으므로 리비전 없이 로드, transformers 4.48.3 고정
  - 두 모델은 transformers 버전이 같아 한 번에 돌려도 된다

출력
  - experiment/order_probe_results.csv   쌍 × 모델 단위
  - experiment/order_probe_detail.csv    번호별 확률 원자료
"""

import argparse
import io
import json
import sys
from pathlib import Path

import pandas as pd

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kgss_token_check import contam_prompt_text, get_tokenizer, load_model, part_b  # noqa: E402

BASE = Path(__file__).resolve().parent
AB = BASE / "selection" / "ab_pairs.csv"
MODELS = {
    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct": "8e6fc27",
    "kakaocorp/kanana-1.5-2.1b-instruct-2505": None,
}


def parse_options(s: str) -> list[str]:
    """'1=... | 2=...' 또는 '1= ... | 2= ...' 형식을 라벨 리스트로."""
    labels = []
    for part in str(s).split("|"):
        part = part.strip()
        if "=" not in part:
            continue
        labels.append(part.split("=", 1)[1].strip())
    return labels


def make_prompt(wording: str, labels: list[str]) -> str:
    opt_txt = "\n".join(f"{i}. {lab}" for i, lab in enumerate(labels, 1))
    return f"문항: {wording}\n\n선택지:\n{opt_txt}\n\n번호로만 답하십시오."


def digit_dist(model, tok, digit_ids, prompt: str, k: int, device: str):
    import torch
    text = contam_prompt_text(tok, prompt)
    ids = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**ids).logits[0, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    targets = [digit_ids.get(str(i)) for i in range(1, k + 1)]
    if any(t is None for t in targets):
        return None, None
    raw = [probs[t].item() for t in targets]
    mass = sum(raw)
    dist = [x / mass for x in raw] if mass > 0 else [1 / k] * k
    return dist, mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(BASE / "experiment"))
    args = ap.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA를 쓸 수 없습니다. --device cpu로 실행하세요.")

    ab = pd.read_csv(AB, encoding="utf-8-sig")
    ab = ab[ab["순서역전"] == True].copy()  # noqa: E712
    print(f"대상 쌍 {len(ab)}개: {', '.join(ab['쌍'])}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    results, details = [], []
    for mid, rev in MODELS.items():
        print(f"\n=== {mid} ===")
        tok, how, err = get_tokenizer(mid, rev)
        if tok is None:
            print(f"  토크나이저 실패: {err}")
            continue
        model, how_m, merr = load_model(mid, args.device, rev)
        if model is None:
            print(f"  모델 로드 실패: {merr[:400]}")
            continue
        L: list[str] = []
        digit_ids = part_b(tok, L)

        for _, r in ab.iterrows():
            labels = parse_options(r["A선택지"])
            k = len(labels)
            wording = str(r["A원문"])
            fwd_p = make_prompt(wording, labels)
            rev_labels = list(reversed(labels))
            rev_p = make_prompt(wording, rev_labels)

            d_f, m_f = digit_dist(model, tok, digit_ids, fwd_p, k, args.device)
            d_r, m_r = digit_dist(model, tok, digit_ids, rev_p, k, args.device)
            if d_f is None or d_r is None:
                print(f"  {r['쌍']}: 번호 토큰 누락, 건너뜀")
                continue

            # 내용 기준으로 되돌리기. 정순은 labels[i]가 위치 i, 역순은 rev_labels[i]가 위치 i
            p_fwd = {labels[i]: d_f[i] for i in range(k)}
            p_rev = {labels[k - 1 - i]: d_r[i] for i in range(k)}
            tv = 0.5 * sum(abs(p_fwd[c] - p_rev[c]) for c in labels)
            # 위치 선호: 정순 위치 i/(k-1), 역순 위치 i/(k-1)을 각각 평균
            pos_f = sum(d_f[i] * i / (k - 1) for i in range(k)) if k > 1 else 0.5
            pos_r = sum(d_r[i] * i / (k - 1) for i in range(k)) if k > 1 else 0.5
            human_tv = float(r["효과_TV거리"])
            results.append({
                "model": mid, "pair": r["쌍"], "k": k,
                "model_tv": round(tv, 4), "human_tv": round(human_tv, 4),
                "ratio": round(tv / human_tv, 2) if human_tv > 0 else None,
                "pos_pref_fwd": round(pos_f, 3), "pos_pref_rev": round(pos_r, 3),
                "pos_pref_mean": round((pos_f + pos_r) / 2, 3),
                "mass_fwd": round(m_f, 4), "mass_rev": round(m_r, 4),
            })
            for c in labels:
                details.append({"model": mid, "pair": r["쌍"], "option": c,
                                "p_fwd": round(p_fwd[c], 5), "p_rev": round(p_rev[c], 5)})
            print(f"  {r['쌍']:<12} TV={tv:.4f} 인간={human_tv:.4f} 배율={tv/human_tv:.2f} "
                  f"위치선호={(pos_f+pos_r)/2:.2f} mass={min(m_f, m_r):.3f}")

        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    res = pd.DataFrame(results)
    det = pd.DataFrame(details)
    res.to_csv(outdir / "order_probe_results.csv", index=False, encoding="utf-8-sig")
    det.to_csv(outdir / "order_probe_detail.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {outdir / 'order_probe_results.csv'}")
    if len(res):
        for m, g in res.groupby("model"):
            print(f"\n{m}")
            print(f"  쌍 {len(g)}개, 중앙 TV {g['model_tv'].median():.4f}, "
                  f"인간 중앙 TV {g['human_tv'].median():.4f}, 중앙 배율 {g['ratio'].median():.2f}")
            print(f"  위치 선호 평균 {g['pos_pref_mean'].mean():.3f} (0.5 = 중립)")
            print(f"  뒤쪽 선호 쌍 {int((g['pos_pref_mean'] > 0.5).sum())}/{len(g)}")


if __name__ == "__main__":
    main()
