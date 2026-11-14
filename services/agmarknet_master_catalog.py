"""Versioned AGMARKNET source-master synchronization and read catalog.

This module preserves provider-specific master IDs separately from
CropBalance's canonical crop/geography registries. It never invents LGD codes.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "reference" / "source_masters"
    / "agmarknet" / "current"
)
SOURCE_NAME = "agmarknet"
SOURCE_ENDPOINT = "https://api.agmarknet.gov.in/v1/daily-price-arrival/filters"
SCHEMA_VERSION = 1
PAGE_LIMIT_DEFAULT = 100
PAGE_LIMIT_MAX = 500

DIMENSION_CONFIG = {
    "states": {
        "source_key": "state_data",
        "id_keys": ("state_id", "stateId"),
        "name_keys": ("state_name", "stateName"),
        "sentinel_ids": {"100000"},
    },
    "districts": {
        "source_key": "district_data",
        "id_keys": ("id", "district_id", "districtId"),
        "name_keys": ("district_name", "districtName"),
        "sentinel_ids": {"100001"},
    },
    "markets": {
        "source_key": "market_data",
        "id_keys": ("id", "market_id", "marketId"),
        "name_keys": ("mkt_name", "market_name", "marketName"),
        "sentinel_ids": {"100002"},
    },
    "commodities": {
        "source_key": "cmdt_data",
        "id_keys": ("cmdt_id", "commodity_id", "commodityId"),
        "name_keys": ("cmdt_name", "commodity_name", "commodityName"),
        "sentinel_ids": set(),
    },
    "varieties": {
        "source_key": "variety_data",
        "id_keys": ("id", "variety_id", "varietyId"),
        "name_keys": ("variety_name", "varietyName"),
        "sentinel_ids": {"100007"},
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_value(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _required_id(record: dict[str, Any], keys: Iterable[str], label: str) -> str:
    value = _first_value(record, keys)
    if value is None:
        raise ValueError(f"{label} record has no source ID.")
    return str(value).strip()


def _required_name(record: dict[str, Any], keys: Iterable[str], label: str) -> str:
    value = _first_value(record, keys)
    if value is None:
        raise ValueError(f"{label} record has no source name.")
    name = " ".join(str(value).split())
    if not name:
        raise ValueError(f"{label} record has an empty source name.")
    return name


def _id_list(value: Any) -> list[str]:
    """Accept AGMARKNET variety commodity links as scalar, list or CSV string."""
    if value is None or value == "":
        return []
    raw_values = value if isinstance(value, list) else str(value).split(",")
    result = []
    for item in raw_values:
        key = str(item).strip()
        if key and key not in result:
            result.append(key)
    return result


def _normalize_dimension(
    metadata: dict[str, Any],
    dimension: str,
) -> tuple[list[dict[str, Any]], int]:
    config = DIMENSION_CONFIG[dimension]
    source_key = config["source_key"]
    raw_records = metadata.get(source_key)
    if not isinstance(raw_records, list):
        raise ValueError(
            f"AGMARKNET metadata field {source_key!r} must be a list."
        )

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    excluded_count = 0

    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ValueError(
                f"AGMARKNET {dimension} records must be JSON objects."
            )
        source_id = _required_id(raw, config["id_keys"], dimension)
        if source_id in config["sentinel_ids"]:
            excluded_count += 1
            continue
        if source_id in seen_ids:
            raise ValueError(
                f"Duplicate AGMARKNET {dimension} source ID: {source_id}"
            )
        seen_ids.add(source_id)
        name = _required_name(raw, config["name_keys"], dimension)
        record: dict[str, Any] = {
            "source_id": source_id,
            "name": name,
        }

        if dimension in {"districts", "markets"}:
            state_id = _first_value(raw, ("state_id", "stateId"))
            if state_id is None:
                raise ValueError(
                    f"AGMARKNET {dimension} {source_id} has no state_id."
                )
            record["state_source_id"] = str(state_id).strip()

        if dimension == "markets":
            district_id = _first_value(raw, ("district_id", "districtId"))
            if district_id is None:
                raise ValueError(
                    f"AGMARKNET market {source_id} has no district_id."
                )
            record["district_source_id"] = str(district_id).strip()

        if dimension == "commodities":
            group_id = _first_value(raw, ("cmdt_group_id", "commodity_group_id"))
            record["group_source_id"] = (
                str(group_id).strip() if group_id is not None else None
            )

        if dimension == "varieties":
            commodity_ids = _id_list(
                _first_value(raw, ("cmdt_id", "commodity_id", "commodityId"))
            )
            record["commodity_source_ids"] = commodity_ids

        records.append(record)

    return records, excluded_count


def build_agmarknet_master_snapshot(
    metadata: dict[str, Any],
    *,
    metadata_sha256: str | None = None,
    synced_at_utc: str | None = None,
) -> dict[str, Any]:
    """Validate AGMARKNET masters and prepare source-namespaced JSON files."""
    if not isinstance(metadata, dict):
        raise ValueError("AGMARKNET metadata must be a JSON object.")

    groups_raw = metadata.get("cmdt_group_data", [])
    if not isinstance(groups_raw, list):
        raise ValueError("AGMARKNET cmdt_group_data must be a list.")

    groups: dict[str, str] = {}
    for item in groups_raw:
        if not isinstance(item, dict):
            raise ValueError("AGMARKNET commodity group rows must be objects.")
        group_id = _first_value(item, ("id", "cmdt_group_id"))
        group_name = _first_value(item, ("cmdt_grp_name", "group_name"))
        if group_id is not None and group_name is not None:
            groups[str(group_id).strip()] = " ".join(str(group_name).split())

    normalized: dict[str, list[dict[str, Any]]] = {}
    excluded: dict[str, int] = {}
    for dimension in DIMENSION_CONFIG:
        normalized[dimension], excluded[dimension] = _normalize_dimension(
            metadata, dimension
        )

    state_ids = {item["source_id"] for item in normalized["states"]}
    district_index = {
        item["source_id"]: item for item in normalized["districts"]
    }
    commodity_ids = {item["source_id"] for item in normalized["commodities"]}

    invalid_districts = [
        item for item in normalized["districts"]
        if item["state_source_id"] not in state_ids
    ]
    if invalid_districts:
        raise ValueError(
            f"{len(invalid_districts)} AGMARKNET districts reference "
            "unknown state source IDs."
        )

    invalid_markets = []
    for market in normalized["markets"]:
        district = district_index.get(market["district_source_id"])
        if (
            district is None
            or district["state_source_id"] != market["state_source_id"]
            or market["state_source_id"] not in state_ids
        ):
            invalid_markets.append(market)
    if invalid_markets:
        raise ValueError(
            f"{len(invalid_markets)} AGMARKNET markets have inconsistent "
            "state/district parent references."
        )

    invalid_varieties = []
    for variety in normalized["varieties"]:
        unknown = [
            commodity_id
            for commodity_id in variety["commodity_source_ids"]
            if commodity_id not in commodity_ids
        ]
        if unknown:
            invalid_varieties.append((variety["source_id"], unknown))
    if invalid_varieties:
        raise ValueError(
            f"{len(invalid_varieties)} AGMARKNET varieties reference "
            "unknown commodity IDs."
        )

    # Keep the source's commodity-group vocabulary; do not force it into
    # CropBalance crop categories (which are a separate curated taxonomy).
    for commodity in normalized["commodities"]:
        group_id = commodity["group_source_id"]
        commodity["group_name"] = groups.get(group_id) if group_id else None

    synced_at = synced_at_utc or utc_now_iso()
    files: dict[str, dict[str, Any]] = {}
    for dimension, records in normalized.items():
        files[f"{dimension}.json"] = {
            "schema_version": SCHEMA_VERSION,
            "source": SOURCE_NAME,
            "source_endpoint": SOURCE_ENDPOINT,
            "dimension": dimension,
            "synced_at_utc": synced_at,
            "source_metadata_sha256": metadata_sha256,
            "record_count": len(records),
            "excluded_selector_rows": excluded[dimension],
            "records": records,
        }

    counts = {
        dimension: len(records)
        for dimension, records in normalized.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE_NAME,
        "source_endpoint": SOURCE_ENDPOINT,
        "synced_at_utc": synced_at,
        "source_metadata_sha256": metadata_sha256,
        "counts": counts,
        "excluded_selector_rows": excluded,
        "relationship_validation": {
            "districts_with_known_state": len(normalized["districts"]),
            "markets_with_consistent_state_and_district": len(normalized["markets"]),
            "varieties_with_known_commodity_links": len(normalized["varieties"]),
        },
        "coverage_note": (
            "This is an AGMARKNET source-master snapshot. Its IDs are "
            "AGMARKNET-specific, not LGD codes. It is not a claim that each "
            "listed market currently reports prices for every commodity."
        ),
    }
    files["manifest.json"] = manifest

    return {
        "files": files,
        "manifest": manifest,
        "summary": {
            "counts": counts,
            "excluded_selector_rows": excluded,
            "source_metadata_sha256": metadata_sha256,
        },
    }


def write_snapshot_files(
    snapshot: dict[str, Any],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """Atomically replace each JSON snapshot file after validation succeeds."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for filename, payload in snapshot["files"].items():
        target = destination / filename
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{filename}.",
            suffix=".tmp",
            dir=destination,
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
            written.append(target)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    return written


class AgmarknetMasterCatalog:
    """Read-only paginated API over a previously synchronized master snapshot."""

    DIMENSIONS = ("states", "districts", "markets", "commodities", "varieties")

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory or DEFAULT_OUTPUT_DIR)
        self._data: dict[str, dict[str, Any]] = {}
        for dimension in self.DIMENSIONS:
            path = self.directory / f"{dimension}.json"
            if not path.is_file():
                raise FileNotFoundError(
                    f"AGMARKNET master snapshot missing: {path}. "
                    "Run scripts/sync_agmarknet_master.py --write."
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, dict)
                or payload.get("source") != SOURCE_NAME
                or payload.get("dimension") != dimension
                or not isinstance(payload.get("records"), list)
                or payload.get("record_count") != len(payload.get("records", []))
            ):
                raise ValueError(
                    f"Invalid AGMARKNET source master file: {path}"
                )
            source_ids = [
                str(item.get("source_id", ""))
                for item in payload["records"]
                if isinstance(item, dict)
            ]
            if len(source_ids) != len(payload["records"]) or len(set(source_ids)) != len(source_ids):
                raise ValueError(
                    f"Duplicate or missing source IDs in AGMARKNET master file: {path}"
                )
            self._data[dimension] = payload

    @staticmethod
    def _paginate(
        records: list[dict[str, Any]],
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        if offset < 0:
            raise ValueError("offset must be zero or greater.")
        if limit < 1 or limit > PAGE_LIMIT_MAX:
            raise ValueError(f"limit must be between 1 and {PAGE_LIMIT_MAX}.")
        return {
            "count": len(records),
            "offset": offset,
            "limit": limit,
            "items": records[offset:offset + limit],
        }

    def coverage(self) -> dict[str, Any]:
        manifest_path = self.directory / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
        return {
            "status": "ok",
            "source": SOURCE_NAME,
            "synced_at_utc": manifest.get("synced_at_utc"),
            "source_metadata_sha256": manifest.get("source_metadata_sha256"),
            "counts": {
                dimension: self._data[dimension]["record_count"]
                for dimension in self.DIMENSIONS
            },
            "excluded_selector_rows": manifest.get(
                "excluded_selector_rows", {}
            ),
            "coverage_note": manifest.get("coverage_note"),
        }

    def list_records(
        self,
        dimension: str,
        *,
        query: str | None = None,
        state_id: str | None = None,
        district_id: str | None = None,
        commodity_id: str | None = None,
        group_id: str | None = None,
        offset: int = 0,
        limit: int = PAGE_LIMIT_DEFAULT,
    ) -> dict[str, Any]:
        if dimension not in self.DIMENSIONS:
            raise ValueError(f"Unsupported AGMARKNET dimension: {dimension}")
        items = self._data[dimension]["records"]
        query_key = str(query).strip().casefold() if query else ""
        filtered = []
        for item in items:
            if query_key and query_key not in str(item.get("name", "")).casefold():
                continue
            if state_id is not None and item.get("state_source_id") != str(state_id):
                continue
            if district_id is not None and item.get("district_source_id") != str(district_id):
                continue
            if commodity_id is not None:
                linked_ids = item.get("commodity_source_ids", [])
                if dimension == "commodities":
                    if item.get("source_id") != str(commodity_id):
                        continue
                elif str(commodity_id) not in linked_ids:
                    continue
            if group_id is not None and item.get("group_source_id") != str(group_id):
                continue
            filtered.append(item)

        page = self._paginate(filtered, offset, limit)
        page.update({
            "status": "ok",
            "source": SOURCE_NAME,
            "dimension": dimension,
        })
        return page
