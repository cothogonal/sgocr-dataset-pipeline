# SGOCR

`sgocr/` is the isolated implementation root for the OCR Spatial QA dataset pipeline.

Rules:
- all pipeline code lives under `sgocr/src/`
- all pipeline tests live under `sgocr/test/`
- dataset artifacts do **not** live here
- dataset artifacts stay in the shared repo data root under `data/ocr_spatial_qa/`

Layout:

```text
sgocr/
  src/
    sgocr/
      __init__.py
      paths.py
      layout.py
      stages.py
  test/
    test_paths.py
    test_layout.py
    test_stages.py
```

Recommended test command:

```bash
PYTHONPATH=sgocr/src .venv_local/bin/python -m unittest discover -s sgocr/test -v
```

Review app:

```bash
cd sgocr
bun run review -- --experiment ablate_gemini_flash_natural2q_40_min4clean
```

Review app tests:

```bash
cd sgocr
bun test test/review_app
```

The Bun review app lives under `sgocr/src/review_app/` and writes autosaved audit files under `data/ocr_spatial_qa/review/<experiment>/`.

## Secrets Policy

API keys must never be pasted into chat, written into docs, committed config files, or logged from the pipeline.

Use environment variables only:
- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

Rules:
- read secrets only from `os.environ`
- never accept API keys as CLI flags
- never print key values
- never dump `os.environ`
- never write request headers containing auth tokens to logs
- keep error messages secret-free; mention only the variable name, never the value
