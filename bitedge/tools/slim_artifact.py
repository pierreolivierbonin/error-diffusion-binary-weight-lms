import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bitedge.gpt2e import pack_bits


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    ck = torch.load(src, map_location="cpu")
    art = {"wpk": {}, "shape": {}, "scale": {}, "fp": ck["fp"]}
    for n, wb in ck["wb"].items():
        art["wpk"][n] = pack_bits(wb.to(torch.int8))
        art["shape"][n] = tuple(wb.shape)
        art["scale"][n] = ck["scale"][n]
    torch.save(art, dst)
    print(f"{dst}: {dst.stat().st_size / 1e6:.2f} MB (from {src.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
