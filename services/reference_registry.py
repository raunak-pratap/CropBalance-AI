"""Validated, provider-neutral reference registries for CropBalance AI.

Internal entity codes are stable CropBalance keys. Provider identifiers (LGD,
AGMARKNET, etc.) are stored separately in source_mappings.json and must never
be mixed across namespaces.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_DIR = PROJECT_ROOT / "data" / "reference"
REGISTRY_FILES = {
    "crops": "crops.json",
    "states": "states.json",
    "districts": "districts.json",
    "markets": "markets.json",
    "source_mappings": "source_mappings.json",
}
SUPPORTED_LANGUAGES = {"en", "hi", "mr"}


def _normalise(value: Any) -> str:
    """Normalise human-entered names and aliases for lookup."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


class ReferenceRegistry:
    """Read-only registry facade for crops, Indian geography and mandi markets.

    The registry is deliberately a seed catalogue, not a claim that market
    data has been loaded for every state or crop in India.
    """

    def __init__(self, registry_dir: str | Path | None = None) -> None:
        self.registry_dir = Path(registry_dir or DEFAULT_REGISTRY_DIR)
        self._data = {
            name: self._read_registry(name, filename)
            for name, filename in REGISTRY_FILES.items()
        }
        self._items = {
            name: {item["code"]: item for item in payload["items"]}
            for name, payload in self._data.items()
            if name != "source_mappings"
        }
        self._mappings = self._build_mapping_index(
            self._data["source_mappings"]["items"]
        )
        self._validate()

    def _read_registry(self, name: str, filename: str) -> dict[str, Any]:
        path = self.registry_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Reference registry file not found: {path}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read {name} registry: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"{filename} must contain a JSON object.")
        if payload.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported schema_version in {filename}; expected 1."
            )
        if payload.get("registry") != name:
            raise ValueError(
                f"{filename} registry name must be '{name}'."
            )
        if not isinstance(payload.get("items"), list):
            raise ValueError(f"{filename} must contain an 'items' list.")
        return payload

    @staticmethod
    def _build_mapping_index(
        mappings: list[dict[str, Any]]
    ) -> dict[tuple[str, str], dict[str, str]]:
        index: dict[tuple[str, str], dict[str, str]] = {}
        for mapping in mappings:
            key = (
                str(mapping.get("entity_type", "")).strip(),
                str(mapping.get("entity_code", "")).strip(),
            )
            source = str(mapping.get("source", "")).strip().casefold()
            external_id = str(mapping.get("external_id", "")).strip()
            if not all(key) or not source or not external_id:
                raise ValueError(
                    "Each source mapping requires entity_type, entity_code, "
                    "source and external_id."
                )
            if source in index.setdefault(key, {}):
                raise ValueError(
                    f"Duplicate source mapping for {key[0]}:{key[1]} "
                    f"from {source}."
                )
            index[key][source] = external_id
        return index

    @staticmethod
    def _labels(item: dict[str, Any]) -> set[str]:
        labels: set[str] = set()
        if item.get("name"):
            labels.add(_normalise(item["name"]))
        for name in (item.get("names") or {}).values():
            if name:
                labels.add(_normalise(name))
        for alias in item.get("aliases") or []:
            if alias:
                labels.add(_normalise(alias))
        return {label for label in labels if label}

    def _validate(self) -> None:
        """Validate unique codes, hierarchy links and source ID namespaces."""
        for registry_name, items in self._items.items():
            seen: set[str] = set()
            for item in self._data[registry_name]["items"]:
                if not isinstance(item, dict):
                    raise ValueError(
                        f"{registry_name} registry items must be objects."
                    )
                code = str(item.get("code", "")).strip()
                if not code:
                    raise ValueError(
                        f"An item in {registry_name} has no internal code."
                    )
                if code in seen:
                    raise ValueError(
                        f"Duplicate {registry_name} code: {code}"
                    )
                seen.add(code)
                if not self._labels(item):
                    raise ValueError(
                        f"{registry_name}:{code} has no searchable name."
                    )

        states = self._items["states"]
        districts = self._items["districts"]
        markets = self._items["markets"]

        for district in districts.values():
            state_code = district.get("state_code")
            if state_code not in states:
                raise ValueError(
                    f"District {district['code']} references unknown state "
                    f"{state_code!r}."
                )

        for market in markets.values():
            state_code = market.get("state_code")
            district_code = market.get("district_code")
            if state_code not in states:
                raise ValueError(
                    f"Market {market['code']} references unknown state "
                    f"{state_code!r}."
                )
            if district_code not in districts:
                raise ValueError(
                    f"Market {market['code']} references unknown district "
                    f"{district_code!r}."
                )
            if districts[district_code].get("state_code") != state_code:
                raise ValueError(
                    f"Market {market['code']} state and district do not match."
                )

        valid_codes = {
            "crop": set(self._items["crops"]),
            "state": set(states),
            "district": set(districts),
            "market": set(markets),
        }
        seen_external: set[tuple[str, str, str]] = set()
        for mapping in self._data["source_mappings"]["items"]:
            entity_type = str(mapping.get("entity_type", "")).strip()
            entity_code = str(mapping.get("entity_code", "")).strip()
            source = str(mapping.get("source", "")).strip().casefold()
            external_id = str(mapping.get("external_id", "")).strip()
            if entity_type not in valid_codes:
                raise ValueError(
                    f"Unsupported source mapping entity_type: {entity_type}"
                )
            if entity_code not in valid_codes[entity_type]:
                raise ValueError(
                    f"Source mapping references unknown "
                    f"{entity_type}:{entity_code}."
                )
            external_key = (entity_type, source, external_id)
            if external_key in seen_external:
                raise ValueError(
                    f"Duplicate external ID {source}:{external_id} "
                    f"for {entity_type}."
                )
            seen_external.add(external_key)

    def resolve(self, registry: str, identifier: str) -> dict[str, Any]:
        """Resolve an internal code, display name or alias to one registry item."""
        if registry not in self._items:
            raise ValueError(f"Unknown reference registry: {registry}")
        query = _normalise(identifier)
        if not query:
            raise ValueError("A non-empty registry identifier is required.")

        direct = self._items[registry].get(str(identifier).strip())
        if direct:
            return deepcopy(direct)

        matches = [
            item
            for item in self._items[registry].values()
            if query in self._labels(item)
        ]
        if not matches:
            raise ValueError(
                f"{identifier!r} was not found in the {registry} registry."
            )
        if len(matches) > 1:
            codes = ", ".join(sorted(item["code"] for item in matches))
            raise ValueError(
                f"{identifier!r} is ambiguous in {registry}: {codes}."
            )
        return deepcopy(matches[0])

    def external_id_for(
        self,
        entity_type: str,
        entity_code: str,
        source: str,
    ) -> str | None:
        """Return an external ID only in its named provider namespace."""
        mapping = self._mappings.get(
            (str(entity_type).strip(), str(entity_code).strip()), {}
        )
        return mapping.get(str(source).strip().casefold())

    def source_mappings_for(
        self,
        entity_type: str,
        entity_code: str,
    ) -> dict[str, str]:
        return deepcopy(
            self._mappings.get(
                (str(entity_type).strip(), str(entity_code).strip()), {}
            )
        )

    def _with_source_ids(
        self, entity_type: str, item: dict[str, Any]
    ) -> dict[str, Any]:
        result = deepcopy(item)
        result["source_ids"] = self.source_mappings_for(
            entity_type, item["code"]
        )
        return result

    def list_crops(
        self,
        category: str | None = None,
        query: str | None = None,
        language: str = "en",
    ) -> list[dict[str, Any]]:
        language = language.casefold()
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language {language!r}; use en, hi or mr."
            )
        query_key = _normalise(query) if query else ""
        results = []
        for item in self._items["crops"].values():
            if category and item.get("category") != category:
                continue
            if query_key and query_key not in self._labels(item):
                if not any(
                    query_key in label for label in self._labels(item)
                ):
                    continue
            result = self._with_source_ids("crop", item)
            names = result.get("names", {})
            result["display_name"] = names.get(language) or names.get("en")
            result["market_data_configured"] = bool(result["source_ids"])
            results.append(result)
        return sorted(results, key=lambda item: item["display_name"].casefold())

    def list_states(
        self,
        query: str | None = None,
        language: str = "en",
    ) -> list[dict[str, Any]]:
        language = language.casefold()
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language {language!r}; use en, hi or mr."
            )
        query_key = _normalise(query) if query else ""
        results = []
        for item in self._items["states"].values():
            name = item["name"]
            if query_key and not any(
                query_key in label for label in self._labels(item)
            ):
                continue
            district_items = [
                d for d in self._items["districts"].values()
                if d["state_code"] == item["code"]
            ]
            market_count = sum(
                1 for market in self._items["markets"].values()
                if market["state_code"] == item["code"]
            )
            result = self._with_source_ids("state", item)
            result.update({
                "display_name": name,
                "districts_catalogued": len(district_items),
                "markets_catalogued": market_count,
                "market_data_configured": market_count > 0,
            })
            results.append(result)
        return sorted(results, key=lambda item: item["name"].casefold())

    def list_districts(
        self,
        state_code: str | None = None,
        query: str | None = None,
        language: str = "en",
    ) -> list[dict[str, Any]]:
        language = language.casefold()
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language {language!r}; use en, hi or mr."
            )
        resolved_state = (
            self.resolve("states", state_code) if state_code else None
        )
        query_key = _normalise(query) if query else ""
        results = []
        for item in self._items["districts"].values():
            if resolved_state and item["state_code"] != resolved_state["code"]:
                continue
            if query_key and not any(
                query_key in label for label in self._labels(item)
            ):
                continue
            result = self._with_source_ids("district", item)
            state = self._items["states"][item["state_code"]]
            market_count = sum(
                1 for market in self._items["markets"].values()
                if market["district_code"] == item["code"]
            )
            result.update({
                "state_name": state["name"],
                "markets_catalogued": market_count,
                "market_data_configured": market_count > 0,
            })
            results.append(result)
        return sorted(results, key=lambda item: item["name"].casefold())

    def list_markets(
        self,
        state_code: str | None = None,
        district_code: str | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        resolved_state = (
            self.resolve("states", state_code) if state_code else None
        )
        resolved_district = (
            self.resolve("districts", district_code)
            if district_code else None
        )
        if resolved_district and resolved_state:
            if resolved_district["state_code"] != resolved_state["code"]:
                raise ValueError(
                    f"District {resolved_district['name']} does not belong "
                    f"to state {resolved_state['name']}."
                )
        if resolved_district and not resolved_state:
            resolved_state = self._items["states"][
                resolved_district["state_code"]
            ]

        query_key = _normalise(query) if query else ""
        results = []
        for item in self._items["markets"].values():
            if resolved_state and item["state_code"] != resolved_state["code"]:
                continue
            if resolved_district and item["district_code"] != resolved_district["code"]:
                continue
            if query_key and not any(
                query_key in label for label in self._labels(item)
            ):
                continue
            result = self._with_source_ids("market", item)
            state = self._items["states"][item["state_code"]]
            district = self._items["districts"][item["district_code"]]
            result.update({
                "state_name": state["name"],
                "district_name": district["name"],
                "market_data_configured": bool(result["source_ids"]),
            })
            results.append(result)
        return sorted(results, key=lambda item: item["name"].casefold())

    def coverage_summary(self) -> dict[str, Any]:
        states = list(self._items["states"].values())
        entity_counts = {
            name: len(items) for name, items in self._items.items()
        }
        mappings = self._data["source_mappings"]["items"]
        mapping_counts = Counter(item["entity_type"] for item in mappings)
        state_type_counts = Counter(
            item["administrative_type"] for item in states
        )
        return {
            "status": "ok",
            "country": "India",
            "catalog_scope": (
                "National state/UT name seed and crop starter taxonomy; "
                "district and mandi records currently cover the Dharashiv pilot."
            ),
            "is_national_market_coverage_complete": False,
            "is_national_district_coverage_complete": False,
            "counts": {
                "states_and_union_territories": entity_counts["states"],
                "states": state_type_counts.get("state", 0),
                "union_territories": state_type_counts.get("union_territory", 0),
                "starter_crops": entity_counts["crops"],
                "districts_catalogued": entity_counts["districts"],
                "markets_catalogued": entity_counts["markets"],
                "source_mappings": len(mappings),
            },
            "entities_with_agmarknet_mapping": dict(mapping_counts),
            "coverage_notes": [
                "The crop catalogue is a starter taxonomy, not a complete list of every crop or variety grown in India.",
                "All 36 state/UT names are seeded; official LGD codes have not yet been synchronized.",
                "Only Dharashiv district and its eight pilot markets are currently in the district/market catalogues.",
                "Only existing project-verified AGMARKNET IDs are mapped. Unmapped IDs are intentionally blank, never guessed.",
                "Catalogue presence does not imply that price observations or validated forecasts are available."
            ],
        }
