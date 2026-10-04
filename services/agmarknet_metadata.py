"""
services/agmarknet_metadata.py

AGMARKNET metadata resolver.

Discovers state, commodity, district, market, and variety IDs
from AGMARKNET's filter endpoint instead of hard-coding IDs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import requests
from loguru import logger


class AgmarknetMetadataResolver:
    """Resolve AGMARKNET names to their internal IDs."""

    BASE_URL = "https://api.agmarknet.gov.in/v1"

    def __init__(
        self,
        cache_path: str = "data/agmarknet_metadata.json",
    ):
        self.cache_path = Path(cache_path)
        self.session = requests.Session()

        self._metadata: Optional[dict[str, Any]] = None

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def load(self, force_refresh: bool = False) -> dict[str, Any]:
        """Load metadata from local cache or AGMARKNET."""

        if self._metadata is not None and not force_refresh:
            return self._metadata

        if self.cache_path.exists() and not force_refresh:
            logger.info(
                f"Loading AGMARKNET metadata from {self.cache_path}"
            )

            with open(
                self.cache_path,
                "r",
                encoding="utf-8",
            ) as f:
                response = json.load(f)

                # AGMARKNET API wraps the actual metadata inside "data".
                self._metadata = response.get("data", response)

            return self._metadata

        logger.info("Fetching fresh AGMARKNET metadata")

        metadata = self._fetch_metadata()

        self._save(metadata)

        self._metadata = metadata

        return metadata

    def get_state_id(self, state: str) -> int:
        """Resolve state name to AGMARKNET state ID."""

        metadata = self.load()

        return self._resolve_id(
            metadata.get("state_data", []),
            state,
            name_keys=("state_name", "stateName"),
            id_keys=("state_id", "stateId"),
            entity="state",
        )

    def get_commodity_id(self, commodity: str) -> int:
        """Resolve commodity name to AGMARKNET commodity ID."""

        metadata = self.load()

        return self._resolve_id(
            metadata.get("cmdt_data", []),
            commodity,
            name_keys=("cmdt_name", "commodity_name", "commodityName"),
            id_keys=("cmdt_id", "commodity_id", "commodityId"),
            entity="commodity",
        )

    def get_district_id(
        self,
        district: str,
        state: Optional[str] = None,
    ) -> int:
        """Resolve district name to AGMARKNET district ID."""

        metadata = self.load()

        records = metadata.get("district_data", [])

        if state:
            state_id = self.get_state_id(state)

            records = [
                item
                for item in records
                if item.get("state_id") == state_id
                or item.get("stateId") == state_id
            ]

        return self._resolve_id(
            records,
            district,
            name_keys=("district_name", "districtName"),
            id_keys=("district_id", "districtId"),
            entity="district",
        )

    def get_market_id(
        self,
        market: str,
        state: Optional[str] = None,
    ) -> int:
        """Resolve market name to AGMARKNET market ID."""

        metadata = self.load()

        records = metadata.get("market_data", [])

        if state:
            state_id = self.get_state_id(state)

            records = [
                item
                for item in records
                if item.get("state_id") == state_id
                or item.get("stateId") == state_id
            ]

        return self._resolve_id(
            records,
            market,
            name_keys=("market_name", "marketName"),
            id_keys=("market_id", "marketId"),
            entity="market",
        )

    def get_variety_id(
        self,
        variety: str,
        commodity: Optional[str] = None,
    ) -> int:
        """Resolve variety name to AGMARKNET variety ID."""

        metadata = self.load()

        records = metadata.get("variety_data", [])

        if commodity:
            commodity_id = self.get_commodity_id(commodity)

            records = [
                item
                for item in records
                if (
                    item.get("cmdt_id") == commodity_id
                    or item.get("commodity_id") == commodity_id
                    or item.get("commodityId") == commodity_id
                )
            ]

        return self._resolve_id(
            records,
            variety,
            name_keys=("variety_name", "varietyName"),
            id_keys=("variety_id", "varietyId"),
            entity="variety",
        )

    # ---------------------------------------------------------
    # AGMARKNET request
    # ---------------------------------------------------------

    def _fetch_metadata(self) -> dict[str, Any]:
        """Fetch AGMARKNET filter metadata."""

        url = f"{self.BASE_URL}/daily-price-arrival/filters"

        headers = {
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://agmarknet.gov.in",
            "Referer": "https://agmarknet.gov.in/",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/154.0.0.0 Safari/537.36"
            ),
        }

        response = self.session.get(
            url,
            headers=headers,
            timeout=30,
        )

        logger.info(
            f"AGMARKNET metadata response: "
            f"{response.status_code}"
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, dict):
            raise ValueError(
                "Unexpected AGMARKNET metadata response."
            )

        return data

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _save(self, metadata: dict[str, Any]) -> None:
        """Save metadata locally."""

        self.cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            self.cache_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                metadata,
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(
            f"AGMARKNET metadata saved → {self.cache_path}"
        )

    @staticmethod
    def _normalize(value: str) -> str:
        """Normalize names for case-insensitive matching."""

        return " ".join(
            str(value)
            .strip()
            .lower()
            .split()
        )

    def _resolve_id(
        self,
        records: list[dict[str, Any]],
        requested_name: str,
        name_keys: tuple[str, ...],
        id_keys: tuple[str, ...],
        entity: str,
    ) -> int:
        """Resolve a human-readable name to an AGMARKNET ID."""

        target = self._normalize(requested_name)

        exact_matches = []

        for item in records:
            item_name = None

            for key in name_keys:
                if item.get(key) is not None:
                    item_name = item[key]
                    break

            if item_name is None:
                continue

            if self._normalize(item_name) == target:
                exact_matches.append(item)

        if not exact_matches:
            available = []

            for item in records:
                for key in name_keys:
                    if item.get(key) is not None:
                        available.append(str(item[key]))
                        break

            raise ValueError(
                f"AGMARKNET {entity!r} not found: "
                f"{requested_name!r}. "
                f"Available records: {len(available)}"
            )

        if len(exact_matches) > 1:
            logger.warning(
                f"Multiple AGMARKNET {entity} matches found "
                f"for {requested_name!r}; using first."
            )

        record = exact_matches[0]

        for key in id_keys:
            if record.get(key) is not None:
                return int(record[key])

        raise ValueError(
            f"AGMARKNET {entity} {requested_name!r} "
            f"was found but has no usable ID."
        )