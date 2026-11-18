"""Download a BEIR dataset into data/<name>/.

Usage:
    python data/download.py            # scifact (default)
    python data/download.py fiqa       # FiQA-2018: personal finance Q&A (~57k documents)
"""
import io
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

import certifi

BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DEST = Path(__file__).parent


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    target = DEST / name
    if target.exists():
        print(f"{target} already exists, skipping download.")
        return
    url = BASE.format(name=name)
    print(f"Downloading {url} ...")
    # certifi's CA bundle avoids CERTIFICATE_VERIFY_FAILED on some Windows Python installs
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ctx) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(DEST)
    print(f"Done: {target}")
    for f in sorted(target.rglob("*")):
        print("  ", f.relative_to(DEST))


if __name__ == "__main__":
    main()
