"""Kaggle용 소형 모델 로짓 노트북 생성 (fp32, 배치 8, 마지막 위치만, EXAONE 2.4B + Kanana 2.1B).

게이트 통과 설정 그대로 쓴다 (kgss_batch_gate.ipynb: 배치 8 최대 차이 1e-5 수준, 뒤집힘 0).
- 입력: --probes CSV를 gzip+base64로 노트북에 내장 (probe_id, prompt, codes만)
- 출력: /kaggle/working/logit_<tag>_fp32.csv (probe_id, model, model_dist, mass)
- 중단 후 재실행하면 이미 쓴 probe_id는 건너뛴다 (재개 가능).

사용법:
    uv run build_full_logits.py                                    # 본실험 (기본값)
    uv run build_full_logits.py --probes experiment_expand/experiment_probes.csv --out kgss_expand_logits.ipynb
"""
import argparse
import base64
import gzip
import io
import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--probes", default="experiment/experiment_probes.csv")
ap.add_argument("--out", default="kgss_full_logits.ipynb")
args = ap.parse_args()
probes = pd.read_csv(BASE / args.probes, encoding="utf-8-sig")
slim = probes[["probe_id", "prompt", "codes"]]
buf = io.BytesIO()
with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
    gz.write(slim.to_csv(index=False).encode("utf-8"))
B64 = base64.b64encode(buf.getvalue()).decode("ascii")
assert "'''" not in B64

PIP = '!pip install -q --no-deps "huggingface-hub==0.36.2" "transformers==4.48.3" "tokenizers>=0.21,<0.22"'
DATA = "B64 = '" + B64 + "'\nprint(len(B64), 'base64 chars')"

RUN = r'''import base64, gzip, io, os, time, gc, json
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| transformers 4.48.3 | dtype fp32 | batch 8 (last position)')
df = pd.read_csv(io.BytesIO(gzip.decompress(base64.b64decode(B64))), encoding='utf-8')
print(len(df), 'probes')
OUT = '/kaggle/working'
MODELS = [
    ('LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct', '8e6fc27d1910b526b5d48a2aa129b08a0293df5e', 'exaone'),
    ('kakaocorp/kanana-1.5-2.1b-instruct-2505', None, 'kanana'),
]
BS = 8
CKPT = 500

def wrap(tok, user):
    if getattr(tok, 'chat_template', None):
        try:
            return tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True) + '정답: '
        except Exception:
            pass
    return user + '\n\n정답: '

def last_logits(model, input_ids, mask, pos):
    base = getattr(model, 'model', None) or getattr(model, 'transformer')
    h = base(input_ids=input_ids, attention_mask=mask, position_ids=pos).last_hidden_state
    return model.lm_head(h[:, -1])

def digit_ids(tok):
    d = {}
    for k in '123456789':
        ids = tok.encode(k, add_special_tokens=False)
        if len(ids) == 1:
            d[k] = ids[0]
    return d

for mid, rev, tag in MODELS:
    path = f'{OUT}/logit_{tag}_fp32.csv'
    done = set(pd.read_csv(path, encoding='utf-8-sig').probe_id) if os.path.exists(path) else set()
    todo = df[~df.probe_id.isin(done)].reset_index(drop=True)
    print(f'\n=== {tag}: 완료 {len(done)}건, 남은 {len(todo)}건 ===', flush=True)
    if len(todo) == 0:
        continue
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True, revision=rev,
                                                 torch_dtype=torch.float32, attn_implementation='eager').to('cuda').eval()
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    digit = digit_ids(tok)
    rows, t0, n = [], time.time(), 0
    for i in range(0, len(todo), BS):
        chunk = todo.iloc[i:i + BS]
        enc = tok([wrap(tok, p) for p in chunk.prompt], return_tensors='pt', padding=True).to('cuda')
        mask = enc['attention_mask']
        pos = (mask.cumsum(dim=1) - 1).clamp(min=0)
        with torch.no_grad():
            lgs = last_logits(model, enc['input_ids'], mask, pos)
        for j, (pid, codes) in enumerate(zip(chunk.probe_id, chunk.codes)):
            p = torch.softmax(lgs[j].float(), dim=-1)
            raw = [p[digit[str(c)]].item() for c in json.loads(codes)]
            m = sum(raw)
            rows.append(dict(probe_id=pid, model=mid, model_dist=json.dumps([x / m for x in raw]), mass=m))
        n += len(chunk)
        if len(rows) >= CKPT or i + BS >= len(todo):
            pd.DataFrame(rows).to_csv(path, mode='a', header=not os.path.exists(path), index=False, encoding='utf-8-sig')
            rows = []
            el = time.time() - t0
            print(f'{tag} {n}/{len(todo)} 경과 {el/60:.1f}분, 남은 예상 {el/n*(len(todo)-n)/60:.1f}분', flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()

for tag in ['exaone', 'kanana']:
    p = f'{OUT}/logit_{tag}_fp32.csv'
    if os.path.exists(p):
        print(tag, len(pd.read_csv(p, encoding='utf-8-sig')), '행')
'''

cells = [PIP, DATA, RUN]
nb = {
    "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
               "source": c.splitlines(keepends=True)} for c in cells],
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
out = BASE / args.out
out.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
print("저장:", out, out.stat().st_size // 1024, "KB")
