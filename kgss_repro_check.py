r"""
로짓 재현성 검증 (본 실행 전 게이트)

확인하는 두 가지
  A. 같은 방식 재실행 일치: 저장된 로짓(logit_<model>.csv)을 같은 조건(한 건씩, 원래 코드)으로 다시 계산해 비교.
     차이가 거의 0이면 같은 환경에서는 결과가 재현된다.
  B. 배치 일치: 같은 프롬프트를 왼쪽 패딩 배치로 계산해 한 건씩 결과와 비교.
     최대 절대 차이와 argmax(최빈 선택) 뒤집힘 수를 보고한다.

게이트 기준 (사전에 정함)
  - A: 최대 절대 차이 < 1e-4 이면 통과
  - B: 최대 절대 차이 < 1e-3 이고 argmax 뒤집힘이 0건이면 배치 실행 가능

사용법 (Colab, GPU)
  python kgss_repro_check.py --model LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct --revision 8e6fc27d1910b526b5d48a2aa129b08a0293df5e --tag exaone
  python kgss_repro_check.py --model kakaocorp/kanana-1.5-2.1b-instruct-2505 --tag kanana

입력: probes.csv (experiment_probes.csv 압축 해제본), logit_<tag>.csv (저장된 로짓)
출력: repro_<tag>.csv (프로브별 차이), 콘솔 요약
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STRUCT = None


def wrap(tok, user):
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True) + "정답: "
        except Exception:
            pass
    return user + "\n\n정답: "


def digit_ids(tok):
    d = {}
    for k in "123456789":
        ids = tok.encode(k, add_special_tokens=False)
        if len(ids) == 1:
            d[k] = ids[0]
    return d


def dist_from_logits(logits, codes, digit):
    p = torch.softmax(logits.float(), dim=-1)
    raw = [p[digit[str(c)]].item() for c in codes]
    m = sum(raw)
    return [x / m for x in raw]


def run_single(model, tok, digit, rows):
    out = {}
    for _, r in rows.iterrows():
        codes = json.loads(r["codes"])
        ids = tok(wrap(tok, r["prompt"]), return_tensors="pt").to("cuda")
        with torch.no_grad():
            lg = model(**ids).logits[0, -1]
        out[r["probe_id"]] = dist_from_logits(lg, codes, digit)
    return out


def run_batched(model, tok, digit, rows, bs):
    out = {}
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    items = list(rows.itertuples(index=False))
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        texts = [wrap(tok, r.prompt) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        mask = enc["attention_mask"]
        pos = (mask.cumsum(dim=1) - 1).clamp(min=0)
        with torch.no_grad():
            lg = model(input_ids=enc["input_ids"], attention_mask=mask, position_ids=pos).logits[:, -1]
        for j, r in enumerate(chunk):
            codes = json.loads(r.codes)
            out[r.probe_id] = dist_from_logits(lg[j], codes, digit)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--probes", default="probes.csv")
    ap.add_argument("--logits", default=None)
    args = ap.parse_args()

    P = pd.read_csv(args.probes)
    P = P[P["codes"].notna()].reset_index(drop=True)
    stored = pd.read_csv(args.logits or f"logit_{args.tag}.csv").set_index("probe_id")
    sample = P[P["probe_id"].isin(stored.index)].sample(n=min(args.n, len(P)), random_state=args.seed)
    print(f"표본 {len(sample)}건 (수준별: {sample['level'].value_counts().to_dict()})", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, revision=args.revision,
                                                 torch_dtype=torch.float16, attn_implementation="eager").to("cuda").eval()
    digit = digit_ids(tok)

    t0 = time.time()
    a = run_single(model, tok, digit, sample)
    print(f"A(한 건씩 재실행) 완료 {time.time() - t0:.0f}초", flush=True)
    t0 = time.time()
    b = run_batched(model, tok, digit, sample, args.batch)
    print(f"B(배치 {args.batch}) 완료 {time.time() - t0:.0f}초", flush=True)

    recs = []
    for _, r in sample.iterrows():
        pid = r["probe_id"]
        s = np.array(json.loads(stored.at[pid, "model_dist"]))
        ai = np.array(a[pid])
        bi = np.array(b[pid])
        recs.append(dict(probe_id=pid, level=r["level"], n_codes=len(s),
                         diff_A_stored=float(np.abs(ai - s).max()),
                         diff_B_A=float(np.abs(bi - ai).max()),
                         argmax_flip_B=int(np.argmax(bi) != np.argmax(ai)),
                         argmax_flip_A=int(np.argmax(ai) != np.argmax(s))))
    rep = pd.DataFrame(recs)
    rep.to_csv(f"repro_{args.tag}.csv", index=False, encoding="utf-8-sig")

    gate_a = rep["diff_A_stored"].max() < 1e-4
    gate_b = (rep["diff_B_A"].max() < 1e-3) and (rep["argmax_flip_B"].sum() == 0)
    print("\n=== 요약 ===")
    print(f"A 저장값과 차이: 최대 {rep['diff_A_stored'].max():.2e}, 평균 {rep['diff_A_stored'].mean():.2e}, 최빈 뒤집힘 {rep['argmax_flip_A'].sum()}건")
    print(f"B 배치와 차이:   최대 {rep['diff_B_A'].max():.2e}, 평균 {rep['diff_B_A'].mean():.2e}, 최빈 뒤집힘 {rep['argmax_flip_B'].sum()}건")
    print(f"게이트 A(재현성) {'통과' if gate_a else '실패'} / 게이트 B(배치) {'통과' if gate_b else '실패'}")
    print(f"저장: repro_{args.tag}.csv")


if __name__ == "__main__":
    main()
