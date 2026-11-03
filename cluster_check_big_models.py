"""대형 모델(EXAONE 7.8B, Kanana 8B) fp16 작동 점검. 200건 표본으로 한 건씩 대 배치 8을 비교한다.

모델 이름이 맞는지, 이 서버 GPU에서 로드되는지, fp16 배치가 한 건씩 계산과 같은 결과를 내는지 확인한다.
EXAONE 7.8B의 원격 코드 리비전은 2.4B와 다를 수 있다 — 에러가 나면 리비전을 지정해야 한다(아래 주석 참고).

사용법:
  python cluster_check_big_models.py
"""
import json
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# EXAONE 7.8B도 2.4B와 같은 패턴으로 'RopeParameters' ImportError가 났다.
# v5 대응 커밋(553ea250..., 2026-02-06) 이전인 2024-12-09 최초 공개 커밋으로 고정.
# huggingface.co API로 확인: 이 커밋의 configuration_exaone.py는 RopeParameters를 참조하지 않음.
MODELS = [
    ("LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct", "496aef060b296b34c6b0035149f5af9e2b8c168c", "exaone78b"),
    ("kakaocorp/kanana-1.5-8b-instruct-2505", None, "kanana8b"),
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


def dist(lg, codes, digit):
    p = torch.softmax(lg.float(), dim=-1)
    raw = [p[digit[str(c)]].item() for c in codes]
    m = sum(raw)
    return np.array([x / m for x in raw])


ROWS = pd.read_csv("probes_sample200.csv", encoding="utf-8-sig").to_dict("records")
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | {len(ROWS)}건")

for mid, rev, tag in MODELS:
    print(f"\n=== {tag} ({mid}) ===", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True, revision=rev,
                                                 torch_dtype=torch.float16, attn_implementation="eager").to("cuda").eval()
    print(f"로드 완료 {time.time()-t0:.0f}초", flush=True)
    digit = digit_ids(tok)

    # A: 한 건씩 (기준)
    t0 = time.time()
    A = []
    for r in ROWS:
        codes = json.loads(r["codes"])
        ids = tok(wrap(tok, r["prompt"]), return_tensors="pt").to("cuda")
        with torch.no_grad():
            lg = model(**ids).logits[0, -1]
        A.append(dist(lg, codes, digit))
    tA = time.time() - t0
    print(f"{tag} A(한 건씩) {tA:.1f}초 ({tA/len(ROWS)*1000:.0f}ms/건)", flush=True)

    # B: 배치 8, fp16, 마지막 위치만
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    B = []
    for i in range(0, len(ROWS), BATCH):
        chunk = ROWS[i:i + BATCH]
        enc = tok([wrap(tok, r["prompt"]) for r in chunk], return_tensors="pt", padding=True).to("cuda")
        mask = enc["attention_mask"]
        pos = (mask.cumsum(dim=1) - 1).clamp(min=0)
        with torch.no_grad():
            lgs = last_logits(model, enc["input_ids"], mask, pos)
        for j, r in enumerate(chunk):
            B.append(dist(lgs[j], json.loads(r["codes"]), digit))
    tB = time.time() - t0
    dB = np.array([np.abs(b - a).max() for b, a in zip(B, A)])
    flips = sum(int(np.argmax(b) != np.argmax(a)) for b, a in zip(B, A))
    ok = (dB.max() < 1e-3) and flips == 0
    print(f"{tag} 배치 8 {tB:.1f}초 ({tB/len(ROWS)*1000:.0f}ms/건), 최대 차이 {dB.max():.2e}, "
          f"최빈 뒤집힘 {flips}, 게이트 {'PASS' if ok else 'FAIL'}", flush=True)

    full_run_min = tB / len(ROWS) * 46545 / 60
    print(f"전체 46,545건 예상 소요(배치 8 기준): {full_run_min:.0f}분", flush=True)

    del model
    torch.cuda.empty_cache()

print("\n완료")
