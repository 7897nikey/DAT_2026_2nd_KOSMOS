"""확장 라운드 로짓 수집 (학과 클러스터, 비배치).

모델별 리비전·dtype은 본실험 수집과 동일하게 맞춘다.
    exaone/kanana       fp32 (build_full_logits.py, Kaggle)
    exaone78b/kanana8b  fp16 (cluster_run_big_models.py)
본실험 때 소형은 배치 8로 돌렸지만 대형은 배치 게이트 FAIL이라 비배치였다. 여기서는 전부 비배치.

500건마다 체크포인트 저장, 중단 후 재실행하면 이어서 계산한다.
사용법: python cluster_run_expand.py --models exaone kanana
"""
import argparse
import json
import os
import platform
import time

os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), "workspace", "hf_cache"))

import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

SPECS = {
    "exaone":    ("LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct", "8e6fc27d1910b526b5d48a2aa129b08a0293df5e", torch.float32),
    "kanana":    ("kakaocorp/kanana-1.5-2.1b-instruct-2505", None, torch.float32),
    "exaone78b": ("LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct", "496aef060b296b34c6b0035149f5af9e2b8c168c", torch.float16),
    "kanana8b":  ("kakaocorp/kanana-1.5-8b-instruct-2505", None, torch.float16),
}
CKPT = 500


def wrap(tok, user):
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                           add_generation_prompt=True) + "정답: "
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", default="experiment_expand/experiment_probes.csv")
    ap.add_argument("--out-dir", default="experiment_expand")
    ap.add_argument("--models", nargs="+", default=list(SPECS), choices=list(SPECS))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.probes, encoding="utf-8-sig")
    print(f"{len(df)}건 로드 | torch {torch.__version__} | transformers {transformers.__version__} | "
          f"GPU {torch.cuda.get_device_name(0)}", flush=True)

    for tag in args.models:
        mid, rev, dtype = SPECS[tag]
        path = f"{args.out_dir}/logit_{tag}.csv"
        done = set(pd.read_csv(path, encoding="utf-8-sig").probe_id) if os.path.exists(path) else set()
        todo = df[~df.probe_id.isin(done)].reset_index(drop=True)
        print(f"\n=== {tag}: 완료 {len(done)}건, 남은 {len(todo)}건 ===", flush=True)
        if len(todo) == 0:
            continue

        tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, revision=rev)
        model = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True, revision=rev,
                                                     torch_dtype=dtype, attn_implementation="eager").to("cuda").eval()
        digit = digit_ids(tok)
        with open(f"{args.out_dir}/meta_{tag}.json", "w", encoding="utf-8") as f:
            json.dump(dict(model=mid, revision=rev, dtype=str(dtype), batch=1,
                           torch=torch.__version__, transformers=transformers.__version__,
                           gpu=torch.cuda.get_device_name(0), host=platform.node()), f, ensure_ascii=False)

        rows, t0, n = [], time.time(), 0
        for _, r in todo.iterrows():
            codes = json.loads(r["codes"])
            ids = tok(wrap(tok, r["prompt"]), return_tensors="pt").to("cuda")
            with torch.no_grad():
                lg = model(**ids).logits[0, -1]
            p = torch.softmax(lg.float(), dim=-1)
            raw = [p[digit[str(c)]].item() for c in codes]
            m = sum(raw)
            rows.append(dict(probe_id=r["probe_id"], model=mid, model_dist=json.dumps([x / m for x in raw]), mass=m))
            n += 1
            if len(rows) >= CKPT or n == len(todo):
                pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False, encoding="utf-8-sig")
                rows = []
                el = time.time() - t0
                print(f"{tag} {n}/{len(todo)} 경과 {el/60:.1f}분, 남은 예상 {el/n*(len(todo)-n)/60:.1f}분", flush=True)

        del model
        torch.cuda.empty_cache()

    print("\n완료", flush=True)


if __name__ == "__main__":
    main()
