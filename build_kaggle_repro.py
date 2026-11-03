"""Kaggle용 재현성 검증 노트북 생성 (kanana, 200건 subset 내장)."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
subset = (BASE / "experiment" / "repro_subset_kanana.json").read_text(encoding="utf-8")
assert "'''" not in subset

PIP = '!pip install -q --no-deps "huggingface-hub==0.36.2" "transformers==4.48.3" "tokenizers>=0.21,<0.22"'

DATA = "import json\nROWS = json.loads(r'''" + subset + "''')\nprint(len(ROWS), 'probes')"

CHECK = r'''import json, numpy as np, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
MID = 'kakaocorp/kanana-1.5-2.1b-instruct-2505'
BATCH = 8
tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MID, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation='eager').to('cuda').eval()
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| dtype fp32')
digit = {}
for k in '123456789':
    ids = tok.encode(k, add_special_tokens=False)
    if len(ids) == 1:
        digit[k] = ids[0]
def wrap(user):
    if getattr(tok, 'chat_template', None):
        try:
            return tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True) + '정답: '
        except Exception:
            pass
    return user + '\n\n정답: '
def dist(lg, codes):
    p = torch.softmax(lg.float(), dim=-1)
    raw = [p[digit[str(c)]].item() for c in codes]
    m = sum(raw)
    return np.array([x / m for x in raw])
# A: 방법 1 - 한 건씩 (패딩 없음)
t0 = time.time()
A = []
for r in ROWS:
    codes = json.loads(r['codes'])
    ids = tok(wrap(r['prompt']), return_tensors='pt').to('cuda')
    with torch.no_grad():
        lg = model(**ids).logits[0, -1]
    A.append(dist(lg, codes))
print('A(한 건씩) 완료', round(time.time() - t0), '초')
# B: 방법 2 - 배치 (왼쪽 패딩 + 어텐션 마스크 + 위치 번호)
tok.padding_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
t0 = time.time()
B = []
for i in range(0, len(ROWS), BATCH):
    chunk = ROWS[i:i + BATCH]
    enc = tok([wrap(r['prompt']) for r in chunk], return_tensors='pt', padding=True).to('cuda')
    mask = enc['attention_mask']
    pos = (mask.cumsum(dim=1) - 1).clamp(min=0)
    with torch.no_grad():
        lgs = model(input_ids=enc['input_ids'], attention_mask=mask, position_ids=pos).logits[:, -1]
    for j, r in enumerate(chunk):
        B.append(dist(lgs[j], json.loads(r['codes'])))
print('B(배치 8) 완료', round(time.time() - t0), '초')
S = [np.array(json.loads(r['stored'])) for r in ROWS]
dA = np.array([np.abs(a - s).max() for a, s in zip(A, S)])
dB = np.array([np.abs(b - a).max() for b, a in zip(B, A)])
flipB = sum(int(np.argmax(b) != np.argmax(a)) for b, a in zip(B, A))
flipA = sum(int(np.argmax(a) != np.argmax(s)) for a, s in zip(A, S))
print(f'A(fp32) vs 저장값(fp16): 최대 {dA.max():.2e}, 평균 {dA.mean():.2e}, 최빈 뒤집힘 {flipA}')
print(f'B(배치) vs A(한 건씩), 둘 다 fp32: 최대 {dB.max():.2e}, 평균 {dB.mean():.2e}, 최빈 뒤집힘 {flipB}')
print('게이트 B', 'PASS' if (dB.max() < 1e-3 and flipB == 0) else 'FAIL')
'''

cells = [PIP, DATA, CHECK]
nb = {
    "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
               "source": c.splitlines(keepends=True)} for c in cells],
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
out = BASE / "kgss_repro_check_kanana_fp32.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
print("저장:", out, out.stat().st_size // 1024, "KB")
