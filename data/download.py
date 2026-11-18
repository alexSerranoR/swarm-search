"""Descarga SciFact (BEIR) a data/scifact/. Código de fontanería: ya está hecho."""
import io
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DEST = Path(__file__).parent


def main() -> None:
    target = DEST / "scifact"
    if target.exists():
        print(f"Ya existe {target}, no descargo nada.")
        return
    print(f"Descargando {URL} ...")
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(URL, context=ctx) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(DEST)
    print(f"Listo: {target}")
    for f in sorted(target.rglob("*")):
        print("  ", f.relative_to(DEST))


if __name__ == "__main__":
    main()