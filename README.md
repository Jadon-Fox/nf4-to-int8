# nf4-to-int8

One-time **NF4 → INT8 pin**. Not a dest-pack flip. Not a train speedup.

```
Unsloth / bitsandbytes NF4 safetensors
        ↓  dequant (16-level codebook, double-quant absmax if present)
     dense f32
        ↓  symmetric INT8 per block (amax/127, zp=0)
INT8 pin directory  (model.safetensors + pin.json)
```

Orch product path stays **NF4-resident** until you load this pin on purpose.

## Pin convert

```bash
python3 nf4_to_int8.py pin \
  --src /path/to/Phi-4-mini-instruct-unsloth-bnb-4bit \
  --out /path/to/Phi-4-mini-int8-pin

python3 nf4_to_int8.py pin --src model.safetensors --out /tmp/pin --dry-run
python3 nf4_to_int8.py fixture --out /tmp/tiny-pin   # 8×64 pin for orch loader CI

```

`--src` is a HuggingFace snapshot dir or a single `.safetensors`.

Writes:

| file | role |
|------|------|
| `model.safetensors` | `*.weight` **I8 [out,in]**, `*.weight.int8_scale` F32, `*.weight.int8_state` JSON; BF16 norms / skipped MLPs copied |
| `pin.json` | counts + mean RMSE vs NF4 dequant. `train_ok: false` |
| `CONVERT_REPORT.json` | per-module RMSE |
| `config.json` | `quantization_config.quant_method = nf4_to_int8_pin` |

Double-quant (Unsloth default): `absmax` is U8 into `nested_quant_map`, then `× nested_absmax[i//256] + nested_offset` (orch R09 D1). Single-quant F32 absmax also works.

Layers left BF16 on the NF4 pin (Phi-4: embed, norms, MLP 1/3/30) stay BF16.

Sharded `model-0000k-of-0000n` is **not** v1 — merge first.

## Single tensor

```bash
python3 nf4_to_int8.py --demo
python3 nf4_to_int8.py tensor --qweight w.nf4.bin --absmax w.absmax.f32.bin --n-elem 4096
python3 test_nf4_to_int8.py
```

## What this is not

- Not `ORCH_BASE_PACK`. That knob does not exist.
- Not an INT8 Tensor-Core GEMM. The pin is storage.
- Not permission to claim the product pin is INT8. Orch stays NF4-resident.
- Requant cannot restore bits NF4 already dropped.

`train_ok=false`.
