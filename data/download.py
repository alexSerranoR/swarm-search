"""Descarga un dataset de BEIR a data/<nombre>/. Código de fontanería: ya está hecho.

Uso:
    python data/download.py            # scifact (por defecto)
    python data/download.py fiqa       # FiQA-2018: preguntas de finanzas (~57k documentos)
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
        print(f"Ya existe {target}, no descargo nada.")
        return
    url = BASE.format(name=name)
    print(f"Descargando {url} ... (puede tardar un poco)")
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ctx) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(DEST)
    print(f"Listo: {target}")
    for f in sorted(target.rglob("*")):
        print("  ", f.relative_to(DEST))


if __name__ == "__main__":
    main()
