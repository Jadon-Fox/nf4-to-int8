#!/usr/bin/env python3
"""Golden checks. Codebook + roundtrip. Not train_ok."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from nf4 import (
    NF4_CODEBOOK,
    NIBBLE_LO_THEN_HI,
    dequant_int8,
    dequant_nf4,
    max_abs_err,
    quantize_int8_symmetric,
    quantize_nf4,
    rmse,
)
from nf4_to_int8 import convert, demo_weights

ROOT = Path(__file__).resolve().parent


def test_codebook_ends():
    assert NF4_CODEBOOK[0] == -1.0
    assert NF4_CODEBOOK[7] == 0.0
    assert NF4_CODEBOOK[15] == 1.0
    assert len(NF4_CODEBOOK) == 16


def test_nf4_roundtrip_zero_and_scale():
    w = [0.0] * 64 + [0.5] * 64
    qw, am = quantize_nf4(w, blocksize=64)
    rec = dequant_nf4(qw, am, 128, 64, NIBBLE_LO_THEN_HI)
    assert am[0] == 0.0
    assert abs(am[1] - 0.5) < 1e-6
    assert max(abs(x) for x in rec[:64]) < 1e-12
    # 0.5 / 0.5 = 1.0 → code 15
    assert abs(rec[64] - 0.5) < 1e-6


def test_int8_zero_point_and_clip():
    w = [0.0, 1.0, -1.0, 2.0]
    q, s = quantize_int8_symmetric(w, blocksize=4)
    rec = dequant_int8(q, s, 4)
    assert s[0] == 2.0 / 127.0
    # signed view
    signed = [b if b < 128 else b - 256 for b in q]
    assert signed[0] == 0
    assert signed[3] == 127
    assert abs(rec[3] - 2.0) < 1e-6


def test_convert_rmse_small():
    w = demo_weights(256)
    qw, am = quantize_nf4(w, blocksize=64)
    got = convert(bytes(qw), am, 256, 64, NIBBLE_LO_THEN_HI, 64)
    assert got["meta"]["train_ok"] is False
    assert got["meta"]["rmse_vs_nf4_dequant"] < 0.02
    assert got["meta"]["max_abs_err_vs_nf4_dequant"] < 0.05
    # recon vs nf4 dequant, not vs original dense
    assert rmse(got["f32"], got["recon"]) == got["meta"]["rmse_vs_nf4_dequant"]
    assert max_abs_err(got["f32"], got["recon"]) == got["meta"]["max_abs_err_vs_nf4_dequant"]


def test_cli_demo():
    with tempfile.TemporaryDirectory() as td:
        prefix = Path(td) / "w"
        r = subprocess.run(
            [sys.executable, str(ROOT / "nf4_to_int8.py"), "--demo", "--out-prefix", str(prefix), "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
        meta = json.loads(r.stdout)
        assert meta["n_elem"] == 256
        assert (prefix.with_suffix(".i8.bin")).stat().st_size == 256
        assert (prefix.with_suffix(".scale.f32.bin")).stat().st_size == 4 * 4  # 256/64


if __name__ == "__main__":
    test_codebook_ends()
    test_nf4_roundtrip_zero_and_scale()
    test_int8_zero_point_and_clip()
    test_convert_rmse_small()
    test_cli_demo()
    print("TEST_NF4_TO_INT8_GREEN codebook dequant requant cli NOT_train_ok")
