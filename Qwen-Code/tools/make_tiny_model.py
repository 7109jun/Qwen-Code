#!/usr/bin/env python3
"""Builds a tiny character-level ONNX "language model" used to test Qwen Code's real ONNX Runtime path.

The model has the inputs `input_ids` / `attention_mask` and the output `logits`. It always predicts the
next character of the cycle a -> b -> c -> a (any other token maps to "a"), so greedy generation
after any prompt yields "abcabc...". Usage: python3 tools/make_tiny_model.py <output-dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

out = Path(sys.argv[1] if len(sys.argv) > 1 else "tiny-model")
out.mkdir(parents=True, exist_ok=True)

chars = [chr(c) for c in range(32, 127)]
vocab = {"<unk>": 0}
for ch in chars:
    vocab[ch] = len(vocab)
V = len(vocab)

tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
tok.pre_tokenizer = pre_tokenizers.Split(Regex("."), "isolated")
tok.decoder = decoders.Fuse()
tok.save(str(out / "tokenizer.json"))

table = np.full((V, V), -10.0, dtype=np.float32)
nxt = {"a": "b", "b": "c", "c": "a"}
for t in range(V):
    table[t, vocab["a"]] = 10.0
for src, dst in nxt.items():
    table[vocab[src], :] = -10.0
    table[vocab[src], vocab[dst]] = 10.0

ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "seq"])
mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, "seq"])
logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "seq", V])
emb = helper.make_tensor("table", TensorProto.FLOAT, [V, V], table.flatten().tolist())
nodes = [
    helper.make_node("Gather", ["table", "input_ids"], ["gathered"], axis=0),
    # keep attention_mask connected so the runtime must feed it
    helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
    helper.make_node("ReduceSum", ["mask_f"], ["mask_sum"], keepdims=0),
    helper.make_node("Mul", ["mask_sum", "zero"], ["zeroed"]),
    helper.make_node("Add", ["gathered", "zeroed"], ["logits"]),
]
zero = helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])
graph = helper.make_graph(nodes, "tiny_lm", [ids, mask], [logits], initializer=[emb, zero])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, str(out / "model.onnx"))
(out / "generation_config.json").write_text(json.dumps({"eos_token_id": []}))
print(f"wrote {out}/model.onnx ({V} tokens)")
