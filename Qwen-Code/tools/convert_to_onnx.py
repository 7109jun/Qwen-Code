#!/usr/bin/env python3
"""Converts a Hugging Face causal language model to ONNX for Qwen Code.

This script is deliberately separate from the program: `qwen` only *loads* ONNX files.

    pip install "optimum[onnxruntime]" transformers torch
    python3 tools/convert_to_onnx.py Qwen/Qwen3-Coder-Next models/coder
    python3 tools/convert_to_onnx.py Qwen/Qwen3-235B-A22B-Thinking-2507 models/thinking --dtype fp16

The output directory contains model.onnx (+ external data files), tokenizer.json, config.json and
generation_config.json, which is the layout [models.*].path in qwen.toml expects. The export uses
the `text-generation-with-past` task, so the KV cache inputs/outputs (past_key_values.* / present.*)
are supported by the runtime.
"""
import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="Hugging Face model id or local directory")
    ap.add_argument("output", help="output directory (e.g. models/coder)")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--no-past", action="store_true", help="export without KV cache (slower generation)")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    try:
        from optimum.exporters.onnx import main_export
        from transformers import AutoTokenizer
    except ImportError:
        print('missing dependencies: pip install "optimum[onnxruntime]" transformers torch', file=sys.stderr)
        return 2

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    task = "text-generation" if args.no_past else "text-generation-with-past"
    main_export(
        args.model,
        output=out,
        task=task,
        opset=args.opset,
        dtype=None if args.dtype == "fp32" else args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    tok.save_pretrained(out)
    if not (out / "tokenizer.json").exists():
        print("warning: tokenizer.json was not produced; the runtime needs a fast tokenizer", file=sys.stderr)
    if (out / "model.onnx").exists():
        print(f"done: {out}/model.onnx")
    else:
        found = sorted(p.name for p in out.glob("*.onnx"))
        print(f"done; ONNX files: {found}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
