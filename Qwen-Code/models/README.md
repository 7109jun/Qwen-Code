# Models

Put the ONNX files here (or point `[models.*].path` in `qwen.toml` anywhere else):

```
models/thinking/  model.onnx (+ *.onnx_data)  tokenizer.json  [config.json] [generation_config.json]
models/coder/     model.onnx (+ *.onnx_data)  tokenizer.json  [config.json] [generation_config.json]
```

Convert Hugging Face models with the separate script `tools/convert_to_onnx.py`.
ONNX Runtime itself is loaded dynamically: install it and set `[runtime.onnx].dylib_path` or the
`ORT_DYLIB_PATH` environment variable.

No model files? Use the Qwen API instead: set `runtime = "api"` for a model in `qwen.toml` and export
`DASHSCOPE_API_KEY`.
