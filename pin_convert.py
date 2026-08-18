"""Offline NF4 (bnb / Unsloth) → INT8 pin.

One-time. Not a dest-pack flip. Not orch hot path. train_ok=false.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nf4 import (
    NIBBLE_LO_THEN_HI,
    decode_absmax,
    dequant_int8,
    dequant_nf4,
    max_abs_err,
    n_absmax,
    pack_f32,
    quantize_int8_symmetric,
    rmse,
    unpack_f32,
)
from safetensors_io import SafeTensorsFile, write_safetensors

COPY_SIDE_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "added_tokens.json",
    "chat_template.jinja",
    "preprocessor_config.json",
)

NF4_SUFFIXES = (
    ".weight.absmax",
    ".weight.nested_absmax",
    ".weight.quant_map",
    ".weight.nested_quant_map",
    ".weight.quant_state.bitsandbytes__nf4",
    ".weight.quant_state",
)


def find_weight_file(src: Path) -> Path:
    if src.is_file() and src.suffix == ".safetensors":
        return src
    if not src.is_dir():
        raise FileNotFoundError(f"src not found: {src}")
    for name in ("model.safetensors", "model-00001-of-00001.safetensors"):
        p = src / name
        if p.is_file():
            return p
    shards = sorted(src.glob("*.safetensors"))
    if len(shards) == 1:
        return shards[0]
    if len(shards) > 1:
        raise ValueError("sharded safetensors not supported in v1 — merge first")
    raise FileNotFoundError(f"no .safetensors in {src}")


def _stem_of_nf4_aux(name: str) -> Optional[str]:
    for suf in NF4_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return None


def group_modules(names: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Return (nf4_stems, passthrough_names, consumed_aux)."""
    aux = {}
    for n in names:
        stem = _stem_of_nf4_aux(n)
        if stem is not None:
            aux.setdefault(stem, []).append(n)
    nf4 = []
    consumed = []
    for stem, extras in aux.items():
        w = stem + ".weight"
        if w in names:
            nf4.append(stem)
            consumed.extend(extras)
            consumed.append(w)
    consumed_set = set(consumed)
    passthrough = [n for n in names if n not in consumed_set]
    return sorted(nf4), passthrough, consumed


def _parse_quant_state(raw: bytes) -> dict:
    text = raw.decode("utf-8", errors="strict").strip().rstrip("\x00")
    return json.loads(text)


def convert_nf4_module(
    st: SafeTensorsFile,
    stem: str,
    int8_blocksize: int,
    nibble_order: int = NIBBLE_LO_THEN_HI,
) -> Tuple[bytes, List[float], dict, List[float]]:
    w_name = stem + ".weight"
    info = st.tensors[w_name]
    qweight = st.read_bytes(w_name)

    state: dict = {}
    state_name = stem + ".weight.quant_state.bitsandbytes__nf4"
    if state_name in st.tensors:
        state = _parse_quant_state(st.read_bytes(state_name))

    shape = tuple(state.get("shape") or [])
    if len(shape) != 2:
        # packed storage is often [out*in/2, 1]
        packed = 1
        for d in info.shape:
            packed *= int(d)
        n_elem = packed * 2
        # last-resort: unknown 2d
        shape = (n_elem, 1)
    else:
        n_elem = int(shape[0]) * int(shape[1])

    blocksize = int(state.get("blocksize") or 64)
    absmax_name = stem + ".weight.absmax"
    if absmax_name not in st.tensors:
        raise ValueError(f"{stem}: missing .weight.absmax")
    abs_info = st.tensors[absmax_name]
    abs_raw = st.read_bytes(absmax_name)

    if abs_info.dtype in ("U8", "I8"):
        nq_name = stem + ".weight.nested_quant_map"
        na_name = stem + ".weight.nested_absmax"
        if nq_name not in st.tensors or na_name not in st.tensors:
            raise ValueError(f"{stem}: U8 absmax without nested maps")
        nested_map = unpack_f32(st.read_bytes(nq_name))
        nested_am = unpack_f32(st.read_bytes(na_name))
        nested_bs = int(state.get("nested_blocksize") or 256)
        nested_off = float(state.get("nested_offset") or 0.0)
        absmax = decode_absmax(
            abs_raw,
            absmax_is_u8=True,
            nested_quant_map=nested_map,
            nested_absmax=nested_am,
            nested_offset=nested_off,
            nested_blocksize=nested_bs,
        )
        dq = "double"
    elif abs_info.dtype == "F32":
        absmax = unpack_f32(abs_raw)
        dq = "single"
    else:
        raise ValueError(f"{stem}: absmax dtype {abs_info.dtype} not supported")

    qmap_name = stem + ".weight.quant_map"
    # codebook is implicit in dequant_nf4 (SSOT). quant_map on disk is a check.
    if qmap_name in st.tensors:
        qmap = unpack_f32(st.read_bytes(qmap_name))
        if len(qmap) != 16:
            raise ValueError(f"{stem}: quant_map len {len(qmap)}")

    f32 = dequant_nf4(qweight, absmax, n_elem, blocksize, nibble_order)
    i8_bs = int8_blocksize if int8_blocksize > 0 else blocksize
    q8, scales = quantize_int8_symmetric(f32, blocksize=i8_bs)
    recon = dequant_int8(q8, scales, blocksize=i8_bs)
    meta = {
        "src": stem,
        "src_quant": "nf4",
        "double_quant": dq,
        "shape": [int(shape[0]), int(shape[1])],
        "n_elem": n_elem,
        "nf4_blocksize": blocksize,
        "int8_blocksize": i8_bs,
        "int8_scheme": "symmetric_per_block_zp0",
        "n_scales": len(scales),
        "rmse_vs_nf4_dequant": rmse(f32, recon),
        "max_abs_err_vs_nf4_dequant": max_abs_err(f32, recon),
    }
    return bytes(q8), scales, meta, f32


def convert_pin(
    src: Path,
    out_dir: Path,
    *,
    int8_blocksize: int = 64,
    dry_run: bool = False,
) -> dict:
    src = src.expanduser().resolve()
    weight_path = find_weight_file(src)
    src_dir = src if src.is_dir() else src.parent
    out_dir = out_dir.expanduser().resolve()

    with SafeTensorsFile(str(weight_path)) as st:
        nf4_stems, passthrough, _consumed = group_modules(st.names())
        plan = {
            "schema": "nf4_to_int8_pin_v1",
            "src": str(weight_path),
            "n_tensors_in": len(st.tensors),
            "n_nf4_modules": len(nf4_stems),
            "n_passthrough": len(passthrough),
            "int8_blocksize": int8_blocksize,
            "int8_scheme": "symmetric_per_block_zp0",
            "train_ok": False,
            "measured_omega": False,
            "note": "offline pin convert. not dest-pack flip. not orch hot path.",
            "modules": [],
        }
        if dry_run:
            plan["nf4_modules"] = nf4_stems
            plan["passthrough"] = passthrough
            return plan

        out_dir.mkdir(parents=True, exist_ok=True)
        out_tensors: List[Tuple[str, str, Tuple[int, ...], bytes]] = []
        reports = []
        for stem in nf4_stems:
            q8, scales, meta, _f32 = convert_nf4_module(st, stem, int8_blocksize)
            shape = tuple(meta["shape"])
            out_tensors.append((stem + ".weight", "I8", shape, q8))
            out_tensors.append(
                (stem + ".weight.int8_scale", "F32", (len(scales),), pack_f32(scales))
            )
            state = {
                "quant_type": "int8_symmetric",
                "blocksize": meta["int8_blocksize"],
                "shape": list(shape),
                "src_quant": "nf4",
                "double_quant": meta["double_quant"],
                "train_ok": False,
            }
            raw_state = json.dumps(state, separators=(",", ":")).encode("utf-8")
            out_tensors.append(
                (stem + ".weight.int8_state", "U8", (len(raw_state),), raw_state)
            )
            reports.append(meta)

        for name in passthrough:
            info = st.tensors[name]
            out_tensors.append((name, info.dtype, info.shape, st.read_bytes(name)))

    write_safetensors(
        str(out_dir / "model.safetensors"),
        out_tensors,
        metadata={
            "format": "pt",
            "quantization": "int8_symmetric_per_block_zp0",
            "converted_from": "nf4_bnb",
            "train_ok": "false",
        },
    )

    pin = {
        "schema": "nf4_to_int8_pin_v1",
        "src": str(weight_path),
        "dst": str(out_dir / "model.safetensors"),
        "n_nf4_modules": len(reports),
        "n_passthrough": len(passthrough),
        "int8_blocksize": int8_blocksize,
        "int8_scheme": "symmetric_per_block_zp0",
        "rmse_mean": (sum(m["rmse_vs_nf4_dequant"] for m in reports) / len(reports)) if reports else 0.0,
        "max_abs_err_max": max((m["max_abs_err_vs_nf4_dequant"] for m in reports), default=0.0),
        "train_ok": False,
        "measured_omega": False,
        "note": "static INT8 pin. orch product path remains NF4-resident until this pin is loaded on purpose.",
    }
    (out_dir / "pin.json").write_text(json.dumps(pin, indent=2) + "\n")
    (out_dir / "CONVERT_REPORT.json").write_text(
        json.dumps({"pin": pin, "modules": reports}, indent=2) + "\n"
    )

    # rewrite config quantization surface if present
    cfg_path = src_dir / "config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            cfg = {}
        cfg["quantization_config"] = {
            "quant_method": "nf4_to_int8_pin",
            "load_in_8bit": True,
            "int8_scheme": "symmetric_per_block_zp0",
            "int8_blocksize": int8_blocksize,
            "converted_from": "bitsandbytes_nf4",
            "train_ok": False,
        }
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    for name in COPY_SIDE_FILES:
        if name == "config.json":
            continue
        src_f = src_dir / name
        if src_f.is_file():
            shutil.copy2(src_f, out_dir / name)

    return {"pin": pin, "modules": reports}


def cmd_pin(args: Any) -> int:
    src = Path(args.src)
    out = Path(args.out)
    got = convert_pin(
        src,
        out,
        int8_blocksize=args.int8_blocksize,
        dry_run=args.dry_run,
    )
    print(json.dumps(got if args.dry_run else got["pin"], indent=2))
    return 0
