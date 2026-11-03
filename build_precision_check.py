"""저장된 fp16 로짓과 fp32 한 건씩 계산값의 차이 점검 노트북 생성 (두 모델, 200건 표본)."""
import base64, gzip, io, json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
P = pd.read_csv(BASE / "experiment" / "experiment_probes.csv", encoding="utf-8-sig")
rows = []
for tag in ["exaone", "kanana"]:
    L = pd.read_csv(BASE / "experiment" / f"logit_{tag}.csv", encoding="utf-8-sig")[["probe_id", "model_dist"]]
    L = L.rename(columns={"model_dist": f"stored_{tag}"})
    P = P.merge(L, on="probe_id", how="inner") if tag == "exaone" else P.merge(L, on="probe_id", how="inner")
sample = P.sample(n=200, random_state=7).reset_index(drop=True)
buf = io.BytesIO()
with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
    gz.write(sample[["probe_id", "prompt", "codes", "stored_exaone", "stored_kanana"]].to_csv(index=False).encode("utf-8"))
B64 = base64.b64encode(buf.getvalue()).decode("ascii")

PIP = '!pip install -q --no-deps "huggingface-hub==0.36.2" "transformers==4.48.3" "tokenizers>=0.21,<0.22"'
DATA = "B64 = '" + B64 + "'"
RUN = r'''import base64, gzip, io, json, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
df = pd.read_csv(io.BytesIO(gzip.decompress(base64.b64decode(B64))), encoding='utf-8')
MODELS = [('LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct', '8e6fc27d1910b526b5d48a2aa129b08a0293df5e', 'exaone'),
          ('kakaocorp/kanana-1.5-2.1b-instruct-2505', None, 'kanana')]
def wrap(tok, user):
    if getattr(tok, 'chat_template', None):
        try:
            return tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True) + '정답: '
        except Exception:
            pass
    return user + '\n\n정답: '
print('torch', torch.__version__, '| cuda', torch.version.cuda)
for mid, rev, tag in MODELS:
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True, revision=rev,
                                                 torch_dtype=torch.float32, attn_implementation='eager').to('cuda').eval()
    digit = {k: tok.encode(k, add_special_tokens=False)[0] for k in '123456789' if len(tok.encode(k, add_special_tokens=False)) == 1}
    diffs, flips = [], 0
    for _, r in df.iterrows():
        codes = json.loads(r['codes'])
        ids = tok(wrap(tok, r['prompt']), return_tensors='pt').to('cuda')
        with torch.no_grad():
            lg = model(**ids).logits[0, -1]
        p = torch.softmax(lg.float(), dim=-1)
        raw = [p[digit[str(c)]].item() for c in codes]
        m = sum(raw)
        f32 = np.array([x / m for x in raw])
        s = np.array(json.loads(r['stored_' + tag]))
        d = np.abs(f32 - s)
        diffs.append(d.max())
        flips += int(np.argmax(f32) != np.argmax(s))
    diffs = np.array(diffs)
    print(f'{tag} fp32 한 건씩 vs 저장 fp16: 최대 {diffs.max():.2e}, 평균(행별 최대차이의 평균) {diffs.mean():.2e}, 중앙값 {np.median(diffs):.2e}, 최빈 뒤집힘 {flips}/200', flush=True)
    del model; torch.cuda.empty_cache()
'''
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                 "source": c.splitlines(keepends=True)} for c in [PIP, DATA, RUN]],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(BASE / "kgss_precision_check.ipynb").write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
print("저장", len(sample), "건")
