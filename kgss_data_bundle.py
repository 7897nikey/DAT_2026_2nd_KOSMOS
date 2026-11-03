r"""
gitignore된 데이터 폴더를 묶어 공유용 zip과 매니페스트를 만든다.

매니페스트(DATA_MANIFEST.csv)는 git에 커밋한다. 깃헙만 보는 사람도 어떤 데이터 버전으로
작업했는지 알 수 있고, 받은 쪽은 --verify로 자기 파일이 같은 버전인지 확인한다.

사용법:
    uv run kgss_data_bundle.py pack              # data_bundle/KOSMOS_data_<날짜>.zip + DATA_MANIFEST.csv
    uv run kgss_data_bundle.py verify            # 로컬 파일을 매니페스트와 대조
"""

import argparse
import hashlib
import io
import sys
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent
FOLDERS = ["inventory", "selection", "persona_sample", "experiment", "experiment_expand"]
MANIFEST = BASE / "DATA_MANIFEST.csv"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files():
    for d in FOLDERS:
        for p in sorted((BASE / d).rglob("*")):
            if p.is_file() and not p.name.startswith("~$"):
                yield p


def pack():
    version = date.today().isoformat()
    rows = [dict(path=p.relative_to(BASE).as_posix(), bytes=p.stat().st_size, sha256=sha256(p)) for p in files()]
    out = BASE / "data_bundle" / f"KOSMOS_data_{version}.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for r in rows:
            z.write(BASE / r["path"], r["path"])
    m = pd.DataFrame(rows)
    m.insert(0, "version", version)
    m.to_csv(MANIFEST, index=False, encoding="utf-8-sig")
    print(f"{len(rows)}개 파일, 원본 {m.bytes.sum() / 1e6:.1f}MB -> {out} ({out.stat().st_size / 1e6:.1f}MB)")
    print(f"매니페스트 -> {MANIFEST}")


def verify():
    m = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    print(f"매니페스트 버전 {m.version.iloc[0]}, {len(m)}개 파일")
    missing, changed = [], []
    for r in m.itertuples():
        p = BASE / r.path
        if not p.exists():
            missing.append(r.path)
        elif sha256(p) != r.sha256:
            changed.append(r.path)
    extra = sorted(set(p.relative_to(BASE).as_posix() for p in files()) - set(m.path))
    for name, lst in [("없음", missing), ("내용 다름", changed), ("매니페스트에 없는 파일", extra)]:
        print(f"{name}: {len(lst)}개")
        for x in lst[:20]:
            print("   ", x)
    print("일치" if not (missing or changed) else "불일치 — 최신 zip을 다시 받으세요")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pack", "verify"])
    {"pack": pack, "verify": verify}[ap.parse_args().cmd]()
