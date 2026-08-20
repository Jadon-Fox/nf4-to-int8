# nf4-to-int8

Drop a model in. Write a static **INT8 pin**. Offline, once.

```
NF4 (bnb / Unsloth, ± double-quant)
FP4 with on-disk 16-level map
BF16 / F16 / F32 dense linears
        ↓
INT8 pin  (I8 + per-block scale, zp=0)
norms / biases copied
embeddings copied (optional --embed int8)
```

Not a dest-pack flip. Not a train speedup. Ampere still unfolds to f32/f16 for GEMM.

## Convert

```bash
# Unsloth / bitsandbytes 4-bit (Phi-4 Mini etc.)
python3 nf4_to_int8.py pin \
  --src /path/to/Phi-4-mini-instruct-unsloth-bnb-4bit \
  --out /path/to/int8-pin

# Full-prec BF16/FP16 checkpoint — linears → INT8, norms stay BF16
python3 nf4_to_int8.py pin --src /path/to/bf16-model --out /path/to/int8-pin --dense int8

# Hybrid Unsloth pin: NF4 linears + skipped BF16 MLPs both become INT8
python3 nf4_to_int8.py pin --src /path/to/bnb-4bit --out /path/to/int8-pin --dense int8

python3 nf4_to_int8.py pin --src model.safetensors --out /tmp/pin --dry-run
python3 nf4_to_int8.py fixture --out /tmp/tiny-pin
```

| flag | default | meaning |
|------|---------|---------|
| `--dense` | `int8` | leftover BF16/F16/F32 **linears** (16-bit ckpt, or Unsloth skipped MLP 1/3/30) |
| `--embed` | `copy` | token embeddings / lm_head stay high prec |
| `--linears` | `int8` | bnb NF4/FP4 linears |
| (norms) | always `copy` | RMSNorm / LayerNorm / bias / rotary never INT8 |

GPTQ, AWQ, GGUF, sharded `model-0000k-of-*`: **refuse** (not v1). Merge / convert those first.

Writes `model.safetensors` + `pin.json` (`schema: nf4_to_int8_pin_v1` — orch loader ABI unchanged) + `CONVERT_REPORT.json`.

## Footprint vs unfold (why INT8 is not always the move)

Unfold = dequant to f32/f16, then GEMM. Ampere (3060) **trains** in TF32/FP16/BF16, not INT8 MMA.

| storage | bits/weight | unfold accuracy | fits a bigger frozen base on 12 GB |
|--------|-------------|-----------------|--------------------------------------|
| **NF4** | ~4.1 | 16 codebook levels. QLoRA default. | **yes — smallest** |
| **INT8** (this pin) | ~8.1 | 255 uniform levels. Better unfold than NF4 if you started from BF16. | middle |
| **BF16/FP16** | 16 | native | no, for 3B+ + LoRA + KV |
| GPTQ/AWQ INT4 | ~4 | similar 4-bit, different error | yes — **not this converter** |

INT8 is the **quality/size middle** for unfold, not the smallest. If the goal is “larger model on a 3060,” keep **NF4** (or GPTQ) as storage. If the goal is “more accurate unfold than NF4, still tighter than BF16,” this pin.

`train_ok=false`.
