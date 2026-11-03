"""Kaggle용 배치 크기 시험 + 두 모델 게이트 노트북 생성 (fp32, 200건 subset 내장).

게이트: 배치 결과 vs 한 건씩 결과 (둘 다 fp32). 저장된 로짓은 쓰지 않는다.
  - 최대 절대 차이 < 1e-3 이고 최빈 뒤집힘 0건이면 PASS
  - 배치 크기 8, 16, 32를 각각 재서 시간을 비교한다.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
subset = (BASE / "experiment" / "repro_subset_kanana.json").read_text(encoding="utf-8")
# 프롬프트와 코드만 쓴다 (stored 값은 게이트에 쓰지 않음)
rows = json.loads(subset)
slim = json.dumps([{"probe_id": r["probe_id"], "prompt": r["prompt"], "codes": r["codes"]} for r in rows], ensure_ascii=False)
assert "'''" not in slim

PIP = '!pip install -q --no-deps "huggingface-hub==0.36.2" "transformers==4.48.3" "tokenizers>=0.21,<0.22"'
DATA = "import json\nROWS = json.loads(r'''" + slim + "''')\nprint(len(ROWS), 'probes')"

GATE = r'''import json, time, numpy as np, torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM
print('torch', torch.__version__, '| cuda', torch.version.cuda)
MODELS = [
    ('LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct', '8e6fc27d1910b526b5d48a2aa129b08a0293df5e', 'exaone'),
    ('kakaocorp/kanana-1.5-2.1b-instruct-2505', None, 'kanana'),
]
BATCHES = [8, 16, 32]

def wrap(tok, user):
    if getattr(tok, 'chat_template', None):
        try:
            return tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True) + '정답: '
        except Exception:
            pass
    return user + '\n\n정답: '

def last_logits(model, input_ids, mask, pos):
    """마지막 위치 선택지 점수만 계산한다.
    전체 위치의 어휘 점수(배치×토큰×어휘)를 만들지 않아서 메모리와 계산이 줄어든다.
    base 모델의 은닉 상태를 얻은 뒤 마지막 토큰에만 lm_head를 적용한다."""
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

def dist(lg, codes, digit):
    p = torch.softmax(lg.float(), dim=-1)
    raw = [p[digit[str(c)]].item() for c in codes]
    m = sum(raw)
    return np.array([x / m for x in raw])

summary = []
for mid, rev, tag in MODELS:
    print('\n===', tag, '===', flush=True)
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True, revision=rev,
                                                 torch_dtype=torch.float32, attn_implementation='eager').to('cuda').eval()
    digit = digit_ids(tok)
    # A: 한 건씩 (기준)
    t0 = time.time()
    A = []
    for r in ROWS:
        codes = json.loads(r['codes'])
        ids = tok(wrap(tok, r['prompt']), return_tensors='pt').to('cuda')
        with torch.no_grad():
            lg = model(**ids).logits[0, -1]
        A.append(dist(lg, codes, digit))
    tA = time.time() - t0
    print(f'{tag} A(한 건씩) {tA:.1f}초', flush=True)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for bs in BATCHES:
        t0 = time.time()
        B = []
        for i in range(0, len(ROWS), bs):
            chunk = ROWS[i:i + bs]
            enc = tok([wrap(tok, r['prompt']) for r in chunk], return_tensors='pt', padding=True).to('cuda')
            mask = enc['attention_mask']
            pos = (mask.cumsum(dim=1) - 1).clamp(min=0)
            with torch.no_grad():
                lgs = last_logits(model, enc['input_ids'], mask, pos)
            for j, r in enumerate(chunk):
                B.append(dist(lgs[j], json.loads(r['codes']), digit))
        tB = time.time() - t0
        dB = np.array([np.abs(b - a).max() for b, a in zip(B, A)])
        flips = sum(int(np.argmax(b) != np.argmax(a)) for b, a in zip(B, A))
        ok = (dB.max() < 1e-3) and flips == 0
        line = f'{tag} 배치 {bs}(마지막 위치만): {tB:.1f}초, 최대 차이 {dB.max():.2e}, 평균 {dB.mean():.2e}, 최빈 뒤집힘 {flips}, 게이트 {"PASS" if ok else "FAIL"}'
        print(line, flush=True)
        summary.append(line)
    del model
    gc.collect()
    torch.cuda.empty_cache()

print('\n=== 요약 ===')
for s in summary:
    print(s)
'''

cells = [PIP, DATA, GATE]
nb = {
    "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
               "source": c.splitlines(keepends=True)} for c in cells],
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
out = BASE / "kgss_batch_gate.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
print("저장:", out, out.stat().st_size // 1024, "KB")
