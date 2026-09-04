r"""
선택지 토큰화 및 로짓 추출 가능성 점검 (v5)

v3에서 드러난 문제
  EXAONE 3.5는 원격 코드(modeling_exaone.py)가 구 transformers API 기준이라
  최신 버전에서 AssertionError로 로드가 실패한다. 어텐션 구현 선택 지점에서
  터지는 것으로 보이므로 attn_implementation="eager"를 포함한 여러 설정을
  순서대로 시도하고, 모두 실패하면 전체 트레이스백을 남긴다.

  Kanana 1.5 2.1B는 config.json의 hidden_size(1792)가 head 수(24)로 나누어
  떨어지지 않아 아키텍처 검증에 걸린다. 토크나이저는 tokenizer.json 직접
  로드로 우회하나, 모델 가중치는 같은 검증에 다시 걸린다.

확인된 사실 (v1~v3)
  EXAONE 3.5 / Kanana 1.5 공통
    - 숫자 1~9가 모두 단일 토큰. 번호 응답 방식이 성립한다.
    - 앞 공백이 별도 토큰으로 분리된다(EXAONE 582, Kanana 220).
      따라서 프롬프트를 공백으로 끝내 미리 소비하는 prefill이 필요하다.
    - 한국어 척도 선택지는 20개 중 20개가 다중 토큰
      (EXAONE 평균 4.1, Kanana 평균 5.8). 텍스트 기반 판별은 불가능하다.
    - 원문자는 모델마다 처리가 달라 배제한다.

v5에서 추가
  --contam-probes로 kgss_contamination.py build가 만든 probes.csv를 넘기면,
  A(로짓분포) 행에 대해 여기서 이미 확보한 첫 토큰 로짓 추출 인프라를 그대로
  재사용해 모델의 암묵적 응답 분포를 뽑아 logit_responses.csv로 저장한다.
  기존 토큰화 진단(A/B/C)과 같은 모델 로드를 공유하므로 추가 로드 비용이 없다.

사용법
    로컬(토크나이저만):
        uv run kgss_token_check.py --tokenizer-only --worded .\selection\variables_worded.csv
    Colab/Kaggle(로짓까지):
        !pip install -q transformers accelerate pandas
        !python kgss_token_check.py --models LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct --device cuda
    오염 프로브 로짓 추출(Colab/Kaggle):
        !python kgss_token_check.py --contam-probes probe/probes.csv --out probe --device cuda
"""

import argparse
import io
import json
import re
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DEFAULT_MODELS = [
    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
    "kakaocorp/kanana-1.5-2.1b-instruct-2505",
]

NON_SCALE_PAT = re.compile(r"^\d+\s*(세|번째|개|명|년|월|일|시간|분|만원)|만원|^\d+$")

FALLBACK_OPTIONS = [
    "매우 자랑스럽다", "약간 자랑스럽다", "별로 자랑스럽지 않다", "전혀 자랑스럽지 않다",
    "매우 신뢰한다", "다소 신뢰한다", "거의 신뢰하지 않는다",
    "원자력 발전 확대", "원자력 발전 현상 유지", "원자력 발전 축소",
    "매우 그렇다", "약간 그렇다", "별로 그렇지 않다", "전혀 그렇지 않다",
    "찬성", "반대", "적절하다", "다소 관대하다", "매우 관대하다", "모르겠다",
]


# ---------------------------------------------------------------- 선택지
def load_options(worded: Path | None, n: int) -> tuple[list[str], str]:
    if not worded or not worded.exists():
        return FALLBACK_OPTIONS[:n], "기본 예시"
    df = pd.read_csv(worded)
    if "선택지" not in df.columns:
        return FALLBACK_OPTIONS[:n], "기본 예시"
    sub = df
    if "ordinal_guess" in df.columns:
        sub = df[df["ordinal_guess"].astype(str).str.lower().isin(["true", "1"])]
    seen, out = set(), []
    for s in sub["선택지"].dropna():
        parts = [p for p in str(s).split("|") if "=" in p]
        if not (2 <= len(parts) <= 7):
            continue
        for p in parts:
            lab = p.split("=", 1)[1].strip()
            if (lab and lab not in seen and len(lab) <= 20
                    and not NON_SCALE_PAT.search(lab)):
                seen.add(lab)
                out.append(lab)
            if len(out) >= n:
                return out, "실제 척도 선택지"
    if len(out) < 5:
        return FALLBACK_OPTIONS[:n], "기본 예시(추출 부족)"
    return out, "실제 척도 선택지"


# ---------------------------------------------------------------- 로드
def get_tokenizer(mid: str, revision: str | None):
    """3단계 폴백. config.json 검증에 걸리면 tokenizer.json을 직접 받는다."""
    from transformers import AutoTokenizer
    errs = []
    kw = {"revision": revision} if revision else {}
    for trc in (True, False):
        try:
            return AutoTokenizer.from_pretrained(mid, trust_remote_code=trc, **kw), "Auto", None
        except Exception as e:
            errs.append(f"Auto(trc={trc}): {type(e).__name__}")
    try:
        from huggingface_hub import hf_hub_download
        from transformers import PreTrainedTokenizerFast
        tj = hf_hub_download(mid, "tokenizer.json", **kw)
        tok = PreTrainedTokenizerFast(tokenizer_file=tj)
        try:
            cfg = json.loads(Path(hf_hub_download(mid, "tokenizer_config.json", **kw))
                             .read_text(encoding="utf-8"))
            if cfg.get("chat_template"):
                tok.chat_template = cfg["chat_template"]
            for k in ("bos_token", "eos_token", "pad_token", "unk_token"):
                v = cfg.get(k)
                if isinstance(v, dict):
                    v = v.get("content")
                if isinstance(v, str):
                    setattr(tok, k, v)
        except Exception:
            pass
        return tok, "tokenizer.json 직접", None
    except Exception as e:
        errs.append(f"tokenizer.json: {type(e).__name__}: {e}")
    return None, None, " / ".join(errs)


def load_model(mid: str, device: str, revision: str | None):
    """여러 설정을 순서대로 시도한다.

    EXAONE 등 원격 코드 모델은 최신 transformers에서 어텐션 구현 선택이
    실패할 수 있으므로 eager를 우선한다. dtype 인자명도 버전에 따라 다르다.
    """
    import torch
    from transformers import AutoModelForCausalLM

    dt = torch.float16 if device != "cpu" else torch.float32
    base = {"trust_remote_code": True}
    if revision:
        base["revision"] = revision

    trials = []
    for dt_key in ("dtype", "torch_dtype"):
        trials.append(("eager + device_map",
                       {**base, dt_key: dt, "attn_implementation": "eager",
                        "device_map": device if device != "cpu" else None}))
        trials.append(("eager, 수동 이동",
                       {**base, dt_key: dt, "attn_implementation": "eager"}))
        trials.append(("기본 설정",
                       {**base, dt_key: dt}))

    errs = []
    for name, kw in trials:
        try:
            model = AutoModelForCausalLM.from_pretrained(mid, **kw)
            if "device_map" not in kw and device != "cpu":
                model = model.to(device)
            model.eval()
            return model, name, None
        except Exception as e:
            errs.append(f"[{name}] {type(e).__name__}: {e}\n"
                        + traceback.format_exc(limit=6))
    return None, None, "\n\n".join(errs)


# ---------------------------------------------------------------- 점검
def part_a(tok, options, src, L):
    print("  [A] 척도 선택지 토큰화")
    L.append(f"### A. 선택지 텍스트 (출처: {src})\n")
    L.append("| 선택지 | 토큰 수 | 분해 |\n|---|---|---|")
    counts = []
    for opt in options:
        ids = tok.encode(opt, add_special_tokens=False)
        counts.append(len(ids))
        L.append(f"| {opt} | {len(ids)} | "
                 f"{' · '.join(repr(tok.decode([i])) for i in ids)} |")
    multi = sum(1 for c in counts if c > 1)
    avg = sum(counts) / len(counts) if counts else 0
    L.append(f"\n> {len(options)}개 중 **{multi}개가 2토큰 이상**, 평균 {avg:.1f}토큰.\n")
    print(f"      {multi}/{len(options)} 다중 토큰 (평균 {avg:.1f})")


def part_b(tok, L) -> dict:
    print("  [B] 응답 표기 토큰화")
    L.append("### B. 응답 표기\n#### 숫자\n")
    L.append("| 숫자 | 그대로 | 앞 공백 | 앞 개행 |\n|---|---|---|---|")
    digit_ids, all_single = {}, True
    for d in "123456789":
        row = [d]
        for pre in ("", " ", "\n"):
            ids = tok.encode(pre + d, add_special_tokens=False)
            ok = len(ids) == 1
            if pre == "":
                digit_ids[d] = ids[0] if ok else None
                all_single &= ok
            row.append(f"{ids} {'✓' if ok else ''}")
        L.append("| " + " | ".join(row) + " |")
    L.append("\n> 숫자 1~9 모두 단일 토큰. 번호 응답 방식 성립.\n" if all_single
             else "\n> 일부 숫자가 다중 토큰. 선택지 수 제한 필요.\n")
    print(f"      숫자 단일 토큰: {'전부' if all_single else '일부'}")

    L.append("#### 영문자\n| 문자 | 그대로 | 앞 공백 |\n|---|---|---|")
    for c in "ABCDEFG":
        r = [c]
        for p in ("", " "):
            ids = tok.encode(p + c, add_special_tokens=False)
            r.append(f"{ids} {'✓' if len(ids) == 1 else ''}")
        L.append("| " + " | ".join(r) + " |")
    L.append("")
    return {d: i for d, i in digit_ids.items() if i is not None}


def build_prompts(tok):
    system = ("당신은 설문 응답자입니다. 제시된 보기 중 하나를 골라 "
              "번호로만 답하십시오.")
    user = ("당신은 45세 남성입니다. 고등학교를 졸업했고 인천에 거주하며 기혼입니다.\n\n"
            "귀하는 우리나라 원자력발전 정책이 어떠한 방향으로 나아가야 한다고 "
            "생각하십니까?\n\n"
            "1. 원자력 발전 확대\n2. 원자력 발전 현상 유지\n3. 원자력 발전 축소")
    out = []
    if getattr(tok, "chat_template", None):
        try:
            base = tok.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True)
            out.append(("chat_template", base))
            out.append(("chat_template + prefill", base + "정답: "))
        except Exception:
            pass
    out.append(("plain + prefill", f"{system}\n\n{user}\n\n정답: "))
    return out


def part_c(model, tok, digit_ids, L, device, topk):
    import torch
    print("  [C] 첫 토큰 로짓")
    L.append("### C. 첫 토큰 로짓\n")
    targets = {d: i for d, i in digit_ids.items() if d in "123"}
    if not targets:
        L.append("숫자가 단일 토큰이 아니어서 건너뜁니다.\n")
        return
    best = None
    for name, text in build_prompts(tok):
        ids = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**ids).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        top = torch.topk(probs, topk)
        mass = sum(probs[i].item() for i in targets.values())
        if best is None or mass > best[1]:
            best = (name, mass)
        L.append(f"#### {name}\n| 순위 | 토큰 | 확률 |\n|---|---|---|")
        for r, (p, i) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
            L.append(f"| {r} | {tok.decode([i])!r}"
                     f"{' ←선택지' if i in targets.values() else ''} | {p:.4f} |")
        L.append("\n| 선택지 | ID | 확률 |\n|---|---|---|")
        for d, i in sorted(targets.items()):
            L.append(f"| {d} | {i} | {probs[i].item():.4f} |")
        L.append(f"\n- 선택지 확률 합 **{mass:.4f}** / 누출 **{1-mass:.4f}**\n")
        print(f"      [{name}] 질량 {mass:.4f}")
    if best:
        L.append(f"> 최적 방식: **{best[0]}** ({best[1]:.4f})\n")
        if best[1] < 0.5:
            L.append("> 누출이 절반을 넘습니다. few-shot 예시나 형식 지시 강화가 필요합니다.\n")


# ---------------------------------------------------------------- 오염 프로브 로짓
def contam_prompt_text(tok, user_text: str) -> str:
    """오염 프로브용 프롬프트. chat_template + prefill을 우선 시도한다.

    part_c의 build_prompts와 같은 원칙(공백 prefill로 앞공백 토큰을 미리
    소비)을 단일 사용자 메시지에 적용한 버전이다.
    """
    if getattr(tok, "chat_template", None):
        try:
            base = tok.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False, add_generation_prompt=True)
            return base + "정답: "
        except Exception:
            pass
    return f"{user_text}\n\n정답: "


def part_d_contam(model, tok, digit_ids, probes_path: Path, device: str, mid: str) -> list[dict]:
    """probes.csv의 A_로짓분포 행마다 선택지 숫자 토큰의 첫 토큰 확률을 뽑는다.

    반환값의 model_dist는 codes 순서에 맞춘, 선택지 토큰들로만 재정규화한
    분포다(mass = 정규화 전 확률질량 합 = 선택지 밖으로 샌 정도의 보수).
    """
    import torch
    print(f"  [D] 오염 프로브 로짓 ({probes_path.name})")
    probes = pd.read_csv(probes_path)
    a = probes[probes["type"] == "A_로짓분포"].copy()
    print(f"      대상 {len(a)}건")

    out = []
    n_skip = 0
    for _, r in a.iterrows():
        codes = json.loads(r["codes"])
        targets = [digit_ids.get(str(c)) for c in codes]
        if any(t is None for t in targets):
            n_skip += 1
            continue
        text = contam_prompt_text(tok, str(r["prompt"]))
        ids = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**ids).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        raw = [probs[t].item() for t in targets]
        mass = sum(raw)
        dist = [x / mass for x in raw] if mass > 0 else [1 / len(raw)] * len(raw)
        out.append({"probe_id": r["probe_id"], "model": mid,
                    "model_dist": json.dumps(dist), "mass": mass})
    if n_skip:
        print(f"      단일 토큰이 아닌 코드 포함 {n_skip}건 건너뜀")
    print(f"      완료 {len(out)}건")
    return out


# ---------------------------------------------------------------- 오염 프로브 생성(B/C)
def part_e_generate(model, tok, probes_path: Path, device: str, mid: str,
                    max_new_tokens: int) -> list[dict]:
    """probes.csv의 B(원문완성)/C(선택지순서) 행을 실제로 생성해 텍스트 응답을 얻는다.

    kgss_contamination.py의 '기존 호출 파이프라인'이 따로 없어서, 이미 로드된
    모델을 재사용해 여기서 직접 생성한다. 채점(문자열 유사도, 순서 일치)이
    재현 가능해야 하므로 그리디 디코딩(do_sample=False)을 쓴다 — 표준 guided
    prompting 설계(Golchin & Surdeanu 2023)도 결정적 생성을 전제한다.
    """
    import torch
    print(f"  [E] B/C 생성 (max_new_tokens={max_new_tokens})")
    probes = pd.read_csv(probes_path)
    bc = probes[probes["type"].isin(["B_원문완성", "C_선택지순서"])].copy()
    print(f"      대상 {len(bc)}건")

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = []
    for _, r in bc.iterrows():
        user_text = str(r["prompt"])
        text = user_text
        if getattr(tok, "chat_template", None):
            try:
                text = tok.apply_chat_template(
                    [{"role": "user", "content": user_text}],
                    tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        ids = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=pad_id)
        resp = tok.decode(gen_ids[0][ids["input_ids"].shape[1]:],
                          skip_special_tokens=True)
        out.append({"probe_id": r["probe_id"], "model": mid, "response": resp})
    print(f"      완료 {len(out)}건")
    return out


def merge_and_save(path: Path, new_rows: list[dict], models: list[str]) -> pd.DataFrame:
    """새 결과를 기존 CSV에 병합한다. 이번에 실행한 모델의 행만 교체하고
    나머지 모델의 기존 행은 보존한다 (모델별로 --models를 나눠 실행해도 안전).
    """
    new_df = pd.DataFrame(new_rows)
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["model"].isin(models)] if "model" in old.columns else old.iloc[0:0]
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--worded", default="./selection/variables_worded.csv")
    ap.add_argument("--out", default="./token_check")
    ap.add_argument("--n-options", type=int, default=20)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--tokenizer-only", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--revision", default=None, help="원격 코드 변경 방지용 커밋 해시")
    ap.add_argument("--contam-probes", default=None,
                    help="kgss_contamination.py build가 만든 probes.csv 경로. "
                         "지정하면 각 모델 로드 후 A(로짓분포) 프로브를 함께 채점해 "
                         "<out>/logit_responses.csv에 모은다.")
    ap.add_argument("--no-contam-generate", action="store_true",
                    help="--contam-probes와 같이 쓸 때, B(원문완성)/C(선택지순서) "
                         "생성을 건너뛰고 A(로짓분포)만 채점한다.")
    ap.add_argument("--max-new-tokens", type=int, default=200,
                    help="B/C 생성 시 최대 생성 토큰 수.")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    options, src = load_options(Path(args.worded), args.n_options)
    print(f"선택지 {len(options)}개 ({src})\n  예: {', '.join(options[:5])}\n")

    contam_probes_path = Path(args.contam_probes) if args.contam_probes else None
    if contam_probes_path and not contam_probes_path.exists():
        print(f"경고: --contam-probes 경로가 없습니다: {contam_probes_path}")
        contam_probes_path = None
    if contam_probes_path and args.tokenizer_only:
        print("경고: --tokenizer-only에서는 로짓을 뽑을 수 없어 --contam-probes를 무시합니다.")
        contam_probes_path = None
    contam_rows: list[dict] = []
    gen_rows: list[dict] = []

    for mid in args.models:
        print(f"=== {mid} ===")
        L = [f"# 토큰화 점검: `{mid}`\n"]
        tok, how, err = get_tokenizer(mid, args.revision)
        if tok is None:
            print(f"  토크나이저 실패: {err}\n")
            L.append(f"토크나이저 로드 실패\n\n```\n{err}\n```\n")
            (outdir / f"{mid.replace('/', '_')}.md").write_text("\n".join(L), encoding="utf-8")
            continue

        print(f"  토크나이저 로드 ({how})")
        L.append(f"- 로드 경로: {how}\n- 어휘 크기: {len(tok):,}")
        L.append(f"- chat_template: "
                 f"{'있음' if getattr(tok, 'chat_template', None) else '없음'}\n")
        part_a(tok, options, src, L)
        digit_ids = part_b(tok, L)

        if not args.tokenizer_only:
            model, how_m, merr = load_model(mid, args.device, args.revision)
            if model is None:
                print(f"  모델 로드 실패 (아래 로그 참조)")
                print(merr[:1500])
                L.append(f"\n### C. 첫 토큰 로짓\n\n모델 로드 실패\n\n```\n{merr}\n```\n")
            else:
                print(f"  모델 로드 ({how_m})")
                L.append(f"\n- 모델 로드 방식: {how_m}\n")
                device = str(next(model.parameters()).device)
                try:
                    part_c(model, tok, digit_ids, L, device, args.topk)
                except Exception as e:
                    L.append(f"\n추론 실패: {type(e).__name__}: {e}\n"
                             f"```\n{traceback.format_exc(limit=8)}\n```\n")
                    print(f"  추론 실패: {type(e).__name__}")
                if contam_probes_path:
                    try:
                        contam_rows.extend(part_d_contam(
                            model, tok, digit_ids, contam_probes_path, device, mid))
                    except Exception as e:
                        print(f"  오염 프로브 로짓 실패: {type(e).__name__}: {e}")
                        traceback.print_exc(limit=8)
                    if not args.no_contam_generate:
                        try:
                            gen_rows.extend(part_e_generate(
                                model, tok, contam_probes_path, device, mid,
                                args.max_new_tokens))
                        except Exception as e:
                            print(f"  B/C 생성 실패: {type(e).__name__}: {e}")
                            traceback.print_exc(limit=8)
                try:
                    import torch
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        (outdir / f"{mid.replace('/', '_')}.md").write_text("\n".join(L), encoding="utf-8")
        print()

    if contam_probes_path:
        combined = merge_and_save(outdir / "logit_responses.csv", contam_rows, args.models)
        print(f"오염 프로브 로짓 {len(contam_rows)}건(이번 실행) / 누적 {len(combined)}건 "
             f"-> {outdir / 'logit_responses.csv'}")
        if not args.no_contam_generate:
            combined_g = merge_and_save(outdir / "responses.csv", gen_rows, args.models)
            print(f"B/C 생성 응답 {len(gen_rows)}건(이번 실행) / 누적 {len(combined_g)}건 "
                 f"-> {outdir / 'responses.csv'}")

    print(f"완료 -> {outdir.resolve()}")


if __name__ == "__main__":
    main()
