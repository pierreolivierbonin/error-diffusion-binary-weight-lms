import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BASE = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1"
SHARDS = ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]
OUT = DATA / "wt103_gpt2.bin"


def main():
    DATA.mkdir(exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    parts = []
    for name in SHARDS:
        dest = DATA / name
        if not dest.exists():
            print(f"downloading {name}", flush=True)
            urllib.request.urlretrieve(f"{BASE}/{name}", dest)
        rows = pq.read_table(dest, columns=["text"]).column("text").to_pylist()
        print(f"{name}: {len(rows)} rows", flush=True)
        for i in range(0, len(rows), 50000):
            chunk = enc.encode_ordinary_batch(rows[i : i + 50000])
            parts.append(np.fromiter((t for row in chunk for t in row), dtype=np.uint16))
        print(f"{name}: tokenized", flush=True)
    all_toks = np.concatenate(parts)
    all_toks.tofile(OUT)
    print(f"{OUT}: {len(all_toks) / 1e6:.1f}M tokens", flush=True)


if __name__ == "__main__":
    main()
