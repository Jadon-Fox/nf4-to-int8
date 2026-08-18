#!/usr/bin/env python3
"""Convert packed NF4 (bnb / Unsloth) to symmetric INT8.

This is dequant-then-requant. Not a free Tensor-Core win. Not train_ok.

    python3 nf4_to_int8.py --demo
    python3 nf4_to_int8.py --qweight w.nf4.bin --absmax w.absmax.f32.bin \\
        --n-elem 4096 --blocksize 64 --out-prefix out/w
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nf4 import (
    NIBBLE_HI_THEN_LO,
    NIBBLE_LO_THEN_HI,
    dequant_int8,
    dequant_nf4,
    max_abs_err,
    n_absmax,
    pack_f32,
    quantize_int8_symmetric,
    quantize_nf4,
    qweight_nbytes,
    rmse,
    unpack_f32,
)


def write_outputs(prefix: Path, q: bytes, scales: list[float], meta: dict) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    (prefix.with_suffix(".i8.bin")).write_bytes(bytes(q))
    (prefix.with_suffix(".scale.f32.bin")).write_bytes(pack_f32(scales))
    (prefix.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2) + "\n")


def convert(
    qweight: bytes,
    absmax: list[float],
    n_elem: int,
    blocksize: int,
    nibble_order: int,
    int8_blocksize: int,
) -> dict:
    f32 = dequant_nf4(qweight, absmax, n_elem, blocksize, nibble_order)
    q8, scales = quantize_int8_symmetric(f32, blocksize=int8_blocksize)
    recon = dequant_int8(q8, scales, blocksize=int8_blocksize)
    meta = {
        "schema": "nf4_to_int8_v1",
        "n_elem": n_elem,
        "nf4_blocksize": blocksize,
        "int8_blocksize": int8_blocksize,
        "nibble_order": "lo_then_hi" if nibble_order == NIBBLE_LO_THEN_HI else "hi_then_lo",
        "int8_scheme": "symmetric_per_block_zp0",
        "qmin": -127,
        "qmax": 127,
        "n_int8_scales": len(scales),
        "rmse_vs_nf4_dequant": rmse(f32, recon),
        "max_abs_err_vs_nf4_dequant": max_abs_err(f32, recon),
        "note": "requant of NF4 dequant. not a train_ok claim. not ORCH_BASE_PACK.",
        "train_ok": False,
    }
    return {"q8": q8, "scales": scales, "f32": f32, "recon": recon, "meta": meta}


def demo_weights(n: int = 256) -> list[float]:
    # Deterministic mix so the demo is replayable.
    out = []
    for i in range(n):
        x = ((i * 37) % 200 - 100) / 80.0
        out.append(x * (0.4 + (i % 17) / 40.0))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="NF4 → INT8 requant (bnb codebook)")
    p.add_argument("--qweight", type=Path, help="packed NF4 bytes (2 nibbles / byte)")
    p.add_argument("--absmax", type=Path, help="f32le absmax, one per NF4 block")
    p.add_argument("--n-elem", type=int, help="logical element count")
    p.add_argument("--blocksize", type=int, default=64, help="NF4 block size (64 or 128)")
    p.add_argument("--int8-blocksize", type=int, default=0, help="INT8 block size (0 = same as NF4)")
    p.add_argument("--nibble-order", choices=("lo_then_hi", "hi_then_lo"), default="lo_then_hi")
    p.add_argument("--out-prefix", type=Path, default=Path("out/converted"))
    p.add_argument("--demo", action="store_true", help="synthesize NF4 then convert")
    p.add_argument("--json", action="store_true", help="print meta JSON to stdout")
    args = p.parse_args(argv)

    nibble = NIBBLE_LO_THEN_HI if args.nibble_order == "lo_then_hi" else NIBBLE_HI_THEN_LO
    i8_bs = args.int8_blocksize if args.int8_blocksize > 0 else args.blocksize

    if args.demo:
        w = demo_weights(256)
        qw, am = quantize_nf4(w, blocksize=args.blocksize, nibble_order=nibble)
        n_elem = len(w)
        qweight = bytes(qw)
        absmax = am
    else:
        if not args.qweight or not args.absmax or not args.n_elem:
            p.error("need --qweight --absmax --n-elem, or --demo")
        qweight = args.qweight.read_bytes()
        absmax = unpack_f32(args.absmax.read_bytes())
        n_elem = args.n_elem
        if len(qweight) < qweight_nbytes(n_elem):
            print("HARD_BLOCK: qweight shorter than n_elem/2", file=sys.stderr)
            return 2
        if len(absmax) < n_absmax(n_elem, args.blocksize):
            print("HARD_BLOCK: absmax shorter than n_blocks", file=sys.stderr)
            return 2

    got = convert(qweight, absmax, n_elem, args.blocksize, nibble, i8_bs)
    write_outputs(args.out_prefix, got["q8"], got["scales"], got["meta"])
    if args.json:
        print(json.dumps(got["meta"], indent=2))
    else:
        m = got["meta"]
        print(
            f"nf4→int8  n={m['n_elem']}  nf4_bs={m['nf4_blocksize']}  "
            f"i8_bs={m['int8_blocksize']}  rmse={m['rmse_vs_nf4_dequant']:.6g}  "
            f"max|e|={m['max_abs_err_vs_nf4_dequant']:.6g}"
        )
        print(f"wrote {args.out_prefix}.i8.bin  {args.out_prefix}.scale.f32.bin  {args.out_prefix}.meta.json")
        print("requant only. train_ok=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
