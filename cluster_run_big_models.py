"""EXAONE 7.8B, Kanana 8B 전체 로짓 추출 (fp16, 한 건씩 — 배치 게이트 FAIL로 배치 미사용).

500건마다 체크포인트 저장, 중단 후 재실행하면 이어서 계산한다.
사용법: python cluster_run_big_models.py
"""
import json
import os
import time

# /tmp는 잡이 끝나면 사라지는 노드 전용 공간이라, 모델 캐시를 workspace(영구 보존)로 돌린다.
# 재실행할 때마다 15GB+ 모델을 다시 받지 않도록 import 전에 설정해야 한다.
os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), "workspace", "hf_cache"))

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    ("LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct", "496aef060b296b34c6b0035149f5af9e2b8c168c", "exaone78b"),
    ("kakaocorp/kanana-1.5-8b-instruct-2505", None, "kanana8b"),
]
CKPT = 500
OUT_DIR = "experiment"


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


os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(f"{OUT_DIR}/experiment_probes.csv", encoding="utf-8-sig")
print(f"{len(df)}건 로드")

for mid, rev, tag in MODELS:
    path = f"{OUT_DIR}/logit_{tag}_fp16.csv"
    done = set(pd.read_csv(path, encoding="utf-8-sig").probe_id) if os.path.exists(path) else set()
    todo = df[~df.probe_id.isin(done)].reset_index(drop=True)
    print(f"\n=== {tag}: 완료 {len(done)}건, 남은 {len(todo)}건 ===", flush=True)
    if len(todo) == 0:
        continue

    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True, revision=rev,
                                                 torch_dtype=torch.float16, attn_implementation="eager").to("cuda").eval()
    digit = digit_ids(tok)

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

for tag in ["exaone78b", "kanana8b"]:
    p = f"{OUT_DIR}/logit_{tag}_fp16.csv"
    if os.path.exists(p):
        print(tag, len(pd.read_csv(p, encoding="utf-8-sig")), "행")
print("완료")
