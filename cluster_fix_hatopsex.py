"""HATOPSEX 로짓 재계산 (설문원문 결측 버그 수정 반영).

experiment_probes.csv에서 HATOPSEX 행만 뽑아 EXAONE 2.4B, Kanana 2.1B로 다시 계산하고,
logit_exaone.csv / logit_kanana.csv에서 HATOPSEX 행만 교체한다. fp16, 배치 8, 마지막 위치만 계산.

사용법:
  python cluster_fix_hatopsex.py
"""
import json

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    ("LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct", "8e6fc27d1910b526b5d48a2aa129b08a0293df5e", "exaone"),
    ("kakaocorp/kanana-1.5-2.1b-instruct-2505", None, "kanana"),
]
BATCH = 8


def wrap(tok, user):
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                           add_generation_prompt=True) + "정답: "
        except Exception:
            pass
    return user + "\n\n정답: "


def last_logits(model, input_ids, mask, pos):
    base = getattr(model, "model", None) or getattr(model, "transformer")
    h = base(input_ids=input_ids, attention_mask=mask, position_ids=pos).last_hidden_state
    return model.lm_head(h[:, -1])


def digit_ids(tok):
    d = {}
    for k in "123456789":
        ids = tok.encode(k, add_special_tokens=False)
        if len(ids) == 1:
            d[k] = ids[0]
    return d


probes = pd.read_csv("experiment/experiment_probes.csv", encoding="utf-8-sig")
todo = probes[probes["var"] == "HATOPSEX"].reset_index(drop=True)
print(f"HATOPSEX 프로브 {len(todo)}건 재계산 예정")

for mid, rev, tag in MODELS:
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True, revision=rev,
                                                 torch_dtype=torch.float16, attn_implementation="eager").to("cuda").eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    digit = digit_ids(tok)
    rows = []
    for i in range(0, len(todo), BATCH):
        chunk = todo.iloc[i:i + BATCH]
        enc = tok([wrap(tok, p) for p in chunk.prompt], return_tensors="pt", padding=True).to("cuda")
        mask = enc["attention_mask"]
        pos = (mask.cumsum(dim=1) - 1).clamp(min=0)
        with torch.no_grad():
            lgs = last_logits(model, enc["input_ids"], mask, pos)
        for j, (pid, codes) in enumerate(zip(chunk.probe_id, chunk.codes)):
            p = torch.softmax(lgs[j].float(), dim=-1)
            raw = [p[digit[str(c)]].item() for c in json.loads(codes)]
            m = sum(raw)
            rows.append(dict(probe_id=pid, model=mid, model_dist=json.dumps([x / m for x in raw]), mass=m))
    new = pd.DataFrame(rows)

    path = f"experiment/logit_{tag}.csv"
    old = pd.read_csv(path, encoding="utf-8-sig")
    before = len(old)
    old = old[~old.probe_id.isin(new.probe_id)]
    merged = pd.concat([old, new], ignore_index=True)
    merged.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"{tag}: {before}행 -> {len(merged)}행 (HATOPSEX {len(new)}건 교체), {path} 저장")

    del model
    torch.cuda.empty_cache()

print("완료")
