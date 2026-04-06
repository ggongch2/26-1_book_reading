# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PaddleOCR is a comprehensive OCR and document parsing toolkit built on PaddlePaddle. It converts images/PDFs into structured data (Markdown/JSON) with support for 100+ languages.

## Common Commands

**Install:**
```bash
pip install paddleocr                   # Core
pip install 'paddleocr[doc-parser]'     # With document parsing
pip install 'paddleocr[all]'            # All features
```

**CLI usage:**
```bash
paddleocr ocr --input image.jpg
paddleocr doc_parser --input document.pdf
```

**Run tests:**
```bash
pytest tests/
pytest -m 'not resource_intensive'      # Skip heavy model-download tests (recommended for dev)
pytest tests/test_ocr_system.py         # Run a single test file
```

**Code style (pre-commit hooks: Black + Flake8):**
```bash
pre-commit run --all-files
```

**Training:**
```bash
python tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml
# Distributed (8 GPUs):
python -m paddle.distributed.launch --gpus '0,1,2,3,4,5,6,7' tools/train.py -c <config>
```

**Evaluation / export:**
```bash
python tools/eval.py -c <config>
python tools/export_model.py -c <config>
```

## Architecture

The codebase has two layers:

### `paddleocr/` — Public API (v3.x)
The user-facing package. Exposes high-level classes and the CLI.
- `_models/` — Individual model wrappers (e.g., `TextDetection`, `FormulaRecognition`)
- `_pipelines/` — Composite pipelines combining multiple models
- `__init__.py` — Exports: `PaddleOCR`, `PaddleOCRVL`, `PPStructureV3`, `PPChatOCRv4Doc`, etc.
- `_cli.py` — CLI entrypoint

### `ppocr/` — Core implementations (legacy backend)
Low-level training/inference code that `paddleocr/` delegates to:
- `modeling/` — Model architectures (backbones, necks, heads)
- `data/` — Data loaders and augmentation
- `losses/`, `metrics/`, `optimizer/`, `postprocess/` — Training components

### Other top-level directories
- `ppstructure/` — Document layout analysis, table/KIE/recovery pipelines (v2 API)
- `configs/` — YAML training configs organized by task: `det/`, `rec/`, `cls/`, `table/`, `kie/`, `e2e/`, `sr/`
- `tools/` — Training (`train.py`), inference (`infer_*.py`), evaluation (`eval.py`)
- `tests/` — pytest test suite; `tests/test_files/` contains sample inputs
- `deploy/` — Docker, Kubernetes, C++, ONNX, TensorRT, OpenVINO deployment configs
- `langchain-paddleocr/` — LangChain document loader integration
- `skills/` — Claude Code skill definitions for text recognition and doc parsing
- `mcp_server/` — MCP (Model Context Protocol) server integration
- `docs/` — Full documentation; `mkdocs.yml` controls the doc site

### Key pipelines and their classes

| Class | Description |
|-------|-------------|
| `PaddleOCR` | PP-OCRv5: scene text detection + recognition (100+ langs) |
| `PaddleOCRVL` | Vision-language model for document parsing (111 langs) |
| `PPStructureV3` | PDF/image → Markdown/JSON with layout, tables, formulas |
| `PPChatOCRv4Doc` | Chat-based document Q&A |
| `PPDocTranslation` | Document translation pipeline |

### Model/Pipeline pattern
All models and pipelines share a standard interface: instantiate → call with `input=`. Models live in `paddleocr/_models/`, pipelines in `paddleocr/_pipelines/`. Training is fully config-driven via YAML files in `configs/`.

## Key Configuration Details

- **Python:** 3.8–3.13
- **PaddlePaddle:** 3.2.0+ required; `paddlex >= 3.4.0` is a core dependency
- **Model cache:** Downloaded models are cached in `~/.paddleocr/`
- **Optional extras:** `doc-parser` pulls `paddlex[ocr,genai-client]`; `ie` for information extraction; `trans` for translation
- **Test markers:** Tests tagged `resource_intensive` download large models — skip with `-m 'not resource_intensive'`
