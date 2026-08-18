# nf4-to-int8

Turn packed **NF4** (bitsandbytes / Unsloth `bnb-4bit`) into **symmetric INT8**.

This is **dequant then requant**. It is not a dest-pack flip and not a train speedup.

```
qweight nibble  →  codebook[idx] * absmax[block]  →  f32
f32             →  clip(round(w / (amax/127)), -127, 127)  →  int8 + per-block scale
```

Codebook is the 16-level NormalFloat4 table used by bitsandbytes and Unsloth (same literals as orch `nf4_codebook.h`). Default nibble order is **lo then hi**.

## CLI

```bash
python3 nf4_to_int8.py --demo

python3 nf4_to_int8.py \
  --qweight layer.qweight.bin \
  --absmax  layer.absmax.f32.bin \
  --n-elem  9437184 \
  --blocksize 64 \
  --out-prefix out/layer
```

Writes:

| file | contents |
|------|----------|
| `*.i8.bin` | `n_elem` bytes, two's-complement int8, zp=0 |
| `*.scale.f32.bin` | little-endian f32, one scale per INT8 block (`amax/127`) |
| `*.meta.json` | dims + RMSE vs NF4 dequant. `train_ok: false` |

`--int8-blocksize` defaults to the NF4 block size (64). `--nibble-order hi_then_lo` if your packer is inverted.

```bash
python3 test_nf4_to_int8.py
```

## What this is not

- Not `ORCH_BASE_PACK`. That knob does not exist on the orchestrator.
- Not an INT8 Tensor-Core GEMM.
- Not permission to claim the product pin is INT8. The pin stays NF4-resident.

Use it when you actually need an INT8 tensor (export, MMA experiment, dump). Expect reconstruction error; the meta file reports RMSE against the NF4 dequant, not against some invented dense original.

## Layout

```
nf4.py            codebook + dequant + int8 quant
nf4_to_int8.py    CLI
test_nf4_to_int8.py
```
