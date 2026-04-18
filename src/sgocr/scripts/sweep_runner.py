"""
Config-driven sweep runner.

Usage:
    python -m sgocr.scripts.sweep_runner --config sgocr/src/sgocr/scripts/configs/ita22.json
    python -m sgocr.scripts.sweep_runner --config sgocr/src/sgocr/scripts/configs/ita22.json --variant precision_r48_t6
    python -m sgocr.scripts.sweep_runner --config sgocr/src/sgocr/scripts/configs/ita22.json --override model=gemini-2.5-pro

Configs are JSON files that declare variants, env_overrides, and rescue params.
The runner replaces the old per-iteration Python scripts (mixed_ita01_sweep.py, etc.)
for new sweeps. Old sweep scripts are left in place.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ..paths import OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, REPO_ROOT


def load_sweep_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        candidates = [
            path,
            Path(__file__).parent / "configs" / path.name,
            Path(__file__).parent / "configs" / f"{path.stem}.json",
        ]
        for c in candidates:
            if c.exists():
                path = c
                break
        else:
            raise FileNotFoundError(f"Config not found: {config_path} (tried {candidates})")
    return json.loads(path.read_text(encoding="utf-8"))


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        key, value = override.split("=", 1)
        try:
            config[key] = json.loads(value)
        except json.JSONDecodeError:
            config[key] = value
    return config


def resolve_paths(config: dict[str, Any], bundle_id: str) -> dict[str, Path]:
    source_name = str(config["source_name"])
    return {
        "source_dir": OCR_SPATIAL_QA_FINAL_ROOT / source_name,
        "bundle_root": OCR_SPATIAL_QA_FINAL_ROOT / bundle_id,
        "intermediate_root": OCR_SPATIAL_QA_INTERMEDIATE_ROOT / bundle_id,
    }


def print_config_summary(config: dict[str, Any], bundle_id: str, variants: list[str]) -> None:
    print(f"sweep: {config.get('name', 'unnamed')}")
    print(f"bundle_id: {bundle_id}")
    print(f"model: {config.get('model', 'unset')}")
    print(f"variants: {', '.join(variants)}")
    print(f"description: {config.get('description', '')}")
    print()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Config-driven sweep runner")
    ap.add_argument("--config", required=True, help="Path to sweep config JSON")
    ap.add_argument("--bundle-id", default=None, help="Override bundle ID (default: auto-generated)")
    ap.add_argument("--variant", action="append", dest="variants", default=[], help="Run specific variant(s) only")
    ap.add_argument("--override", action="append", dest="overrides", default=[], help="Override config key=value")
    ap.add_argument("--dry-run", action="store_true", help="Print config and exit")
    ap.add_argument("--list-variants", action="store_true", help="List available variants and exit")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    config = load_sweep_config(args.config)
    config = apply_overrides(config, args.overrides)

    all_variants = config.get("variants", {})

    if args.list_variants:
        for name, spec in all_variants.items():
            desc = spec.get("description", "")
            print(f"  {name}: {desc}")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = config.get("name", "sweep")
    bundle_id = args.bundle_id or f"sgocr_mixed_{config_name}_{stamp}"

    selected = args.variants if args.variants else list(all_variants.keys())
    for v in selected:
        if v not in all_variants:
            print(f"Unknown variant: {v}. Available: {', '.join(all_variants.keys())}", file=sys.stderr)
            sys.exit(1)

    paths = resolve_paths(config, bundle_id)
    print_config_summary(config, bundle_id, selected)

    if args.dry_run:
        print("Dry run — config loaded successfully. Exiting.")
        print(json.dumps(config, indent=2))
        return

    print(f"Ready to run {len(selected)} variant(s) from {args.config}")
    print(f"Source: {paths['source_dir']}")
    print(f"Output: {paths['bundle_root']}")
    print()
    print("To run the actual sweep, import and call the variant runner from your sweep script.")
    print("This runner provides config loading, validation, and path resolution.")
    print("The execution logic remains in the sweep scripts to preserve per-iteration customization.")


if __name__ == "__main__":
    main()
