"""Synchronize AGMARKNET state/district/market/commodity/variety masters.

Default behavior is a safe dry run. Pass --write to generate the normalized,
provider-namespaced JSON catalogs. Pass --refresh to fetch fresh metadata from
AGMARKNET before transforming it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.agmarknet_master_catalog import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    build_agmarknet_master_snapshot,
    sha256_file,
    write_snapshot_files,
)


def load_metadata(path: Path, refresh: bool) -> dict:
    if refresh:
        from services.agmarknet_metadata import AgmarknetMetadataResolver

        resolver = AgmarknetMetadataResolver(cache_path=str(path))
        metadata = resolver.load(force_refresh=True)
    else:
        if not path.is_file():
            raise FileNotFoundError(f"Metadata cache not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("data", payload) if isinstance(payload, dict) else payload

    if not isinstance(metadata, dict):
        raise ValueError("AGMARKNET metadata cache must contain a JSON object.")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and synchronize AGMARKNET source master data."
    )
    parser.add_argument(
        "--metadata-cache",
        type=Path,
        default=PROJECT_ROOT / "data" / "agmarknet_metadata.json",
        help="Existing AGMARKNET metadata cache (default: data/agmarknet_metadata.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for normalized source masters.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch fresh AGMARKNET filter metadata before synchronization.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write normalized JSON files. Without this flag, run a dry-run only.",
    )
    args = parser.parse_args()

    metadata = load_metadata(args.metadata_cache, args.refresh)
    metadata_hash = sha256_file(args.metadata_cache)

    snapshot = build_agmarknet_master_snapshot(
        metadata,
        metadata_sha256=metadata_hash,
    )

    print("\nAGMARKNET MASTER SYNC VALIDATION")
    print("Metadata cache:", args.metadata_cache)
    print("Metadata SHA-256:", metadata_hash)
    print("Output directory:", args.output_dir)
    print("\nValidated record counts:")
    for dimension, count in snapshot["manifest"]["counts"].items():
        excluded = snapshot["manifest"]["excluded_selector_rows"].get(dimension, 0)
        print(f"  {dimension:12s} {count:>6,} (excluded selector rows: {excluded})")

    print("\nRelationship checks:")
    for name, count in snapshot["manifest"]["relationship_validation"].items():
        print(f"  {name}: {count:,}")

    print("\nNote:", snapshot["manifest"]["coverage_note"])

    if not args.write:
        print("\nDRY RUN ONLY — no normalized files were written.")
        print("Review the counts, then rerun with --write to save the snapshots.")
        return 0

    written = write_snapshot_files(snapshot, args.output_dir)
    print("\nSYNC COMPLETE")
    for path in written:
        print("  Wrote:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
