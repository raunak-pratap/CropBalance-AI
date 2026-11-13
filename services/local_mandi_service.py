"""Read and query CropBalance AI's local Dharashiv mandi archive.

Expected location in the project:
    services/local_mandi_service.py

Default archive path (relative to the project root):
    data/raw/agmarknet_weekly/dharashiv_prices_v3.csv

Important data rules:
- This archive is a historical snapshot, not a live price feed.
- NR / blank / invalid prices remain missing; they are never changed to zero.
- A reported numeric value is not automatically a modal or sale price.
- Market comparisons use only matching date, unit, metric, and variety.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math
from datetime import date, datetime
from zoneinfo import ZoneInfo
import pandas as pd

from services.reference_registry import ReferenceRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "agmarknet_weekly"
    / "dharashiv_prices_v3.csv"
)

# Columns that the service needs to identify and read an observation.
REQUIRED_COLUMNS = {
    "observation_date",
    "market",
    "market_id",
    "crop",
    "reported_price",
}

# These columns are useful but may be absent in older archive versions.
OPTIONAL_COLUMNS = {
    "state",
    "state_id",
    "district",
    "district_id",
    "commodity_name",
    "commodity_id",
    "variety",
    "variety_id",
    "reported_price_raw",
    "price_metric",
    "price_semantics_verified",
    "price_unit",
    "arrivals_value",
    "arrivals_unit",
    "price_status",
    "data_source",
    "endpoint",
    "fetched_at_utc",
    "raw_response_file",
    "provenance_status",
    "conflict_flag",
}


def _clean_text(value: Any) -> str:
    """Convert a CSV cell to clean text without turning missing data into 'nan'."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _normalise_id(value: Any) -> str:
    """Normalise IDs so values such as 1599 and '1599.0' match."""
    value = _clean_text(value)
    if not value:
        return ""
    try:
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return value


def _to_positive_number(value: Any) -> float | None:
    """Parse a strictly positive number; NR, blanks, zero and negatives stay missing."""
    value = _clean_text(value)
    if not value:
        return None
    try:
        number = float(value.replace(",", ""))
    except ValueError:
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _to_optional_number(value: Any) -> float | None:
    """Parse a finite number, allowing zero (used for non-price fields)."""
    value = _clean_text(value)
    if not value:
        return None
    try:
        number = float(value.replace(",", ""))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _to_optional_bool(value: Any) -> bool | None:
    value = _clean_text(value).casefold()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    return None


def _format_number(value: float | None) -> int | float | None:
    """Return whole-number prices as int for cleaner JSON responses."""
    if value is None:
        return None
    if float(value).is_integer():
        return int(value)
    return float(value)


class LocalMandiService:
    """Query the local Dharashiv archive without treating it as live market data.

    The CSV is loaded for each method call. This keeps results fresh after the
    weekly collector updates the archive and avoids holding a stale DataFrame.
    """

    def __init__(
        self,
        archive_path: str | Path | None = None,
        reference_registry: ReferenceRegistry | None = None,
    ) -> None:
        self.archive_path = Path(archive_path) if archive_path else DEFAULT_ARCHIVE_PATH
        self.reference_registry = reference_registry or ReferenceRegistry()

    def _load_archive(self) -> pd.DataFrame:
        """Read the CSV and add normalised helper columns used by the methods."""
        if not self.archive_path.exists():
            raise FileNotFoundError(
                "Local mandi archive was not found at "
                f"'{self.archive_path}'. Check that "
                "data/raw/agmarknet_weekly/dharashiv_prices_v3.csv exists."
            )

        try:
            frame = pd.read_csv(
                self.archive_path,
                dtype=str,
                keep_default_na=False,
                encoding="utf-8-sig",
            )
        except (OSError, UnicodeError, pd.errors.ParserError) as exc:
            raise RuntimeError(
                f"Could not read local mandi archive '{self.archive_path}': {exc}"
            ) from exc

        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                "Local mandi archive is missing required columns: "
                f"{missing_text}. Check the v3 CSV schema."
            )

        # Add optional columns as empty strings so older CSVs remain readable.
        for column in OPTIONAL_COLUMNS:
            if column not in frame.columns:
                frame[column] = ""

        # Keep original columns untouched. Underscore-prefixed columns are
        # temporary values used internally for filtering and comparisons.
        frame["_date"] = pd.to_datetime(
            frame["observation_date"].map(_clean_text),
            format="%Y-%m-%d",
            errors="coerce",
        )
        frame["_crop_key"] = frame.apply(
            lambda row: (
                _clean_text(row.get("crop"))
                or _clean_text(row.get("commodity_name"))
            ).casefold(),
            axis=1,
        )
        frame["_market_id_key"] = frame["market_id"].map(_normalise_id)
        frame["_state_key"] = frame["state"].map(
            lambda value: _clean_text(value).casefold()
        )
        frame["_district_key"] = frame["district"].map(
            lambda value: _clean_text(value).casefold()
        )
        frame["_state_id_key"] = frame["state_id"].map(_normalise_id)
        frame["_district_id_key"] = frame["district_id"].map(_normalise_id)
        frame["_source_key"] = frame["data_source"].map(
            lambda value: _clean_text(value).casefold()
        )
        frame["_market_key"] = frame.apply(
            lambda row: "|".join(
                part
                for part in (
                    _clean_text(row.get("data_source")).casefold(),
                    _normalise_id(row.get("state_id")) or _clean_text(row.get("state")).casefold(),
                    _normalise_id(row.get("district_id")) or _clean_text(row.get("district")).casefold(),
                    _normalise_id(row.get("market_id")) or _clean_text(row.get("market")).casefold(),
                )
                if part
            ),
            axis=1,
        )
        # Keep variety labels distinct. In particular, do not merge "Unknown"
        # with a named variety or silently replace either one.
        frame["_variety_key"] = frame["variety"].map(
            lambda value: _clean_text(value).casefold()
        )
        frame["_unit_key"] = frame["price_unit"].map(
            lambda value: _clean_text(value).casefold()
        )
        frame["_metric_key"] = frame["price_metric"].map(
            lambda value: _clean_text(value).casefold()
        )
        frame["_reported_price"] = frame["reported_price"].map(
            _to_positive_number
        )
        frame["_conflict"] = frame["conflict_flag"].map(
            lambda value: _to_optional_bool(value) is True
        )
        return frame

    def _matches_crop(self, frame: pd.DataFrame, crop: str) -> pd.DataFrame:
        crop_key = _clean_text(crop).casefold()
        if not crop_key:
            raise ValueError("crop must be a non-empty crop name, such as 'wheat'.")

        try:
            crop_item = self.reference_registry.resolve("crops", crop)
            crop_keys = {
                crop_item["code"].casefold(),
                *[
                    _clean_text(value).casefold()
                    for value in crop_item.get("names", {}).values()
                    if _clean_text(value)
                ],
                *[
                    _clean_text(value).casefold()
                    for value in crop_item.get("aliases", [])
                    if _clean_text(value)
                ],
            }
        except ValueError:
            crop_keys = {crop_key}

        return frame.loc[frame["_crop_key"].isin(crop_keys)].copy()

    def _filter_geography(
        self,
        frame: pd.DataFrame,
        state_code: str | None = None,
        district_code: str | None = None,
    ) -> pd.DataFrame:
        """Filter rows using internal registry codes, names or provider IDs."""
        state_item = (
            self.reference_registry.resolve("states", state_code)
            if state_code else None
        )
        district_item = (
            self.reference_registry.resolve("districts", district_code)
            if district_code else None
        )

        if district_item:
            parent_state = district_item["state_code"]
            if state_item and state_item["code"] != parent_state:
                raise ValueError(
                    f"District {district_item['name']} does not belong to "
                    f"state {state_item['name']}."
                )
            if not state_item:
                state_item = self.reference_registry.resolve("states", parent_state)

        result = frame
        if state_item:
            state_labels = {
                _clean_text(state_item["name"]).casefold(),
                *[
                    _clean_text(alias).casefold()
                    for alias in state_item.get("aliases", [])
                ],
            }
            state_mask = result["_state_key"].isin(state_labels)
            source_id = self.reference_registry.external_id_for(
                "state", state_item["code"], "agmarknet"
            )
            if source_id:
                state_mask = state_mask | (
                    result["_state_id_key"].eq(source_id)
                    & result["_source_key"].eq("agmarknet")
                )
            result = result.loc[state_mask].copy()

        if district_item:
            district_labels = {
                _clean_text(district_item["name"]).casefold(),
                *[
                    _clean_text(alias).casefold()
                    for alias in district_item.get("aliases", [])
                ],
            }
            district_mask = result["_district_key"].isin(district_labels)
            source_id = self.reference_registry.external_id_for(
                "district", district_item["code"], "agmarknet"
            )
            if source_id:
                district_mask = district_mask | (
                    result["_district_id_key"].eq(source_id)
                    & result["_source_key"].eq("agmarknet")
                )
            result = result.loc[district_mask].copy()

        return result

    @staticmethod
    def _location_metadata(frame: pd.DataFrame) -> dict[str, str | None]:
        """Return a location only if all rows refer to one state/district."""
        states = sorted({
            _clean_text(value)
            for value in frame.get("state", pd.Series(dtype=str)).tolist()
            if _clean_text(value)
        })
        districts = sorted({
            _clean_text(value)
            for value in frame.get("district", pd.Series(dtype=str)).tolist()
            if _clean_text(value)
        })
        return {
            "state": states[0] if len(states) == 1 else None,
            "district": districts[0] if len(districts) == 1 else None,
        }

    @staticmethod
    def _serialise_record(row: pd.Series) -> dict[str, Any]:
        """Convert one archive row into a JSON-friendly dictionary."""
        parsed_date = row.get("_date")
        observation_date = (
            parsed_date.strftime("%Y-%m-%d")
            if pd.notna(parsed_date)
            else _clean_text(row.get("observation_date")) or None
        )
        price = row.get("_reported_price")
        if pd.isna(price):
            price = None
        else:
            price = _format_number(float(price))

        return {
            "observation_date": observation_date,
            "state": _clean_text(row.get("state")) or None,
            "state_id": _normalise_id(row.get("state_id")) or None,
            "district": _clean_text(row.get("district")) or None,
            "district_id": _normalise_id(row.get("district_id")) or None,
            "market": _clean_text(row.get("market")) or None,
            "market_id": _normalise_id(row.get("market_id")) or None,
            "crop": _clean_text(row.get("crop")) or _clean_text(row.get("commodity_name")) or None,
            "commodity_name": _clean_text(row.get("commodity_name")) or None,
            "commodity_id": _normalise_id(row.get("commodity_id")) or None,
            "variety": _clean_text(row.get("variety")) or None,
            "variety_id": _normalise_id(row.get("variety_id")) or None,
            "reported_price_raw": _clean_text(row.get("reported_price_raw")) or None,
            "reported_price": price,
            "price_metric": _clean_text(row.get("price_metric")) or None,
            "price_semantics_verified": _to_optional_bool(
                row.get("price_semantics_verified")
            ),
            "price_unit": _clean_text(row.get("price_unit")) or None,
            "arrivals_value": _format_number(
                _to_optional_number(row.get("arrivals_value"))
            ),
            "arrivals_unit": _clean_text(row.get("arrivals_unit")) or None,
            "price_status": _clean_text(row.get("price_status")) or (
                "reported_unverified_metric" if price is not None else "not_reported_or_invalid"
            ),
            "data_source": _clean_text(row.get("data_source")) or None,
            "endpoint": _clean_text(row.get("endpoint")) or None,
            "fetched_at_utc": _clean_text(row.get("fetched_at_utc")) or None,
            "raw_response_file": _clean_text(row.get("raw_response_file")) or None,
            "provenance_status": _clean_text(row.get("provenance_status")) or None,
            "conflict_flag": bool(row.get("_conflict", False)),
            "is_live": False,
        }

    @staticmethod
    def _data_quality(frame: pd.DataFrame) -> dict[str, Any]:
        numeric_prices = int(frame["_reported_price"].notna().sum())
        conflict_count = int(frame["_conflict"].sum())
        unique_sources = sorted(
            {
                _clean_text(value)
                for value in frame["data_source"].tolist()
                if _clean_text(value)
            }
        )
        dated = frame["_date"].dropna()
        return {
            "observation_count": int(len(frame)),
            "numeric_price_count": numeric_prices,
            "missing_or_invalid_price_count": int(len(frame) - numeric_prices),
            "conflict_count": conflict_count,
            "invalid_date_count": int(frame["_date"].isna().sum()),
            "data_sources": unique_sources,
            "first_observation_date": (
                dated.min().strftime("%Y-%m-%d") if not dated.empty else None
            ),
            "last_observation_date": (
                dated.max().strftime("%Y-%m-%d") if not dated.empty else None
            ),
        }

    @staticmethod
    def _freshness_metadata(
        frame: pd.DataFrame,
        as_of: date | None = None,
    ) -> dict[str, Any]:
        """
        Calculate archive freshness using observation dates.

        Initial thresholds for this weekly data source:
        0–7 days: fresh
        8–14 days: aging
        More than 14 days: stale
        """

        dates = frame["_date"].dropna()

        if dates.empty:
            return {
                "last_observation_date": None,
                "data_age_days": None,
                "freshness_status": "unknown",
            }

        latest_date = pd.Timestamp(dates.max()).date()

        today = as_of or datetime.now(
            ZoneInfo("Asia/Kolkata")
        ).date()

        age_days = (today - latest_date).days

        if age_days < 0:
            status = "future_dated"
        elif age_days <= 7:
            status = "fresh"
        elif age_days <= 14:
            status = "aging"
        else:
            status = "stale"

        return {
            "last_observation_date": latest_date.isoformat(),
            "data_age_days": age_days,
            "freshness_status": status,
        }

    def get_markets(
        self,
        state_code: str | None = None,
        district_code: str | None = None,
    ) -> dict[str, Any]:
        """Return archive markets, optionally filtered by a registered geography."""
        frame = self._filter_geography(
            self._load_archive(), state_code, district_code
        )
        location = self._location_metadata(frame)
        if frame.empty:
            return {
                "status": "no_data",
                **location,
                "is_live": False,
                "markets": [],
                "message": "No market observations are available for the requested geography.",
            }

        markets: list[dict[str, Any]] = []
        # Group by market ID when available; otherwise use the market name.
        for market_key, group in frame.groupby("_market_key", dropna=False):
            dates = group["_date"].dropna()
            markets.append(
                {
                    "market_id": _normalise_id(group.iloc[0].get("market_id")) or None,
                    "market": _clean_text(group.iloc[0].get("market")) or "Unknown market",
                    "district": _clean_text(group.iloc[0].get("district")) or None,
                    "state": _clean_text(group.iloc[0].get("state")) or None,
                    "observation_count": int(len(group)),
                    "first_observation_date": (
                        dates.min().strftime("%Y-%m-%d") if not dates.empty else None
                    ),
                    "last_observation_date": (
                        dates.max().strftime("%Y-%m-%d") if not dates.empty else None
                    ),
                }
            )

        markets.sort(key=lambda item: (item["market"] or "").casefold())
        return {
            "status": "ok",
            **location,
            "is_live": False,
            **self._freshness_metadata(frame),
            "market_count": len(markets),
            "markets": markets,
        }

    def get_latest_prices(
        self,
        crop: str,
        state_code: str | None = None,
        district_code: str | None = None,
    ) -> dict[str, Any]:
        """Return each market's latest observation for a crop, including NR rows.

        The latest observation is selected before checking whether its price is
        numeric. Therefore, an old numeric price is not presented as current
        when a newer observation for that market says NR.
        """
        frame = self._matches_crop(self._load_archive(), crop)
        frame = self._filter_geography(frame, state_code, district_code)
        location = self._location_metadata(frame)
        if frame.empty:
            return {
                "status": "no_observations",
                "crop": _clean_text(crop),
                **location,
                "is_live": False,
                "records": [],
                "message": f"No archived observations were found for {crop} in the requested geography.",
            }

        dated = frame.loc[frame["_date"].notna()].copy()
        if dated.empty:
            return {
                "status": "no_valid_dates",
                "crop": _clean_text(crop),
                **location,
                "is_live": False,
                "records": [self._serialise_record(row) for _, row in frame.iterrows()],
                "message": "Rows exist, but none has a valid observation date.",
            }

        latest_dates = dated.groupby("_market_key")["_date"].transform("max")
        latest = dated.loc[dated["_date"] == latest_dates].copy()
        latest.sort_values(
            ["market", "variety", "observation_date"],
            kind="stable",
            inplace=True,
        )
        records = [self._serialise_record(row) for _, row in latest.iterrows()]
        numeric_count = sum(record["reported_price"] is not None for record in records)

        return {
            "status": "ok" if numeric_count else "no_numeric_prices",
            "crop": _clean_text(crop),
            **location,
            "is_live": False,
            **self._freshness_metadata(latest),
            "record_count": len(records),
            "numeric_price_count": numeric_count,
            "records": records,
            "data_quality": self._data_quality(latest),
            "message": (
                "Latest archived observations by market. NR and missing values are shown as missing, not zero."
            ),
            "caveat": (
                "These are archived reported values, not live quotes. The meaning of the reported-price "
                "field has not been independently verified; do not label it modal or guaranteed sale price."
            ),
        }

    def get_market_history(
        self,
        crop: str,
        market_id: str | int,
        state_code: str | None = None,
        district_code: str | None = None,
    ) -> dict[str, Any]:
        """Return all archived observations for one crop and one market ID."""
        requested_market_id = _normalise_id(market_id)
        if not requested_market_id:
            raise ValueError("market_id must be a non-empty market identifier.")

        frame = self._matches_crop(self._load_archive(), crop)
        frame = self._filter_geography(frame, state_code, district_code)
        history = frame.loc[frame["_market_id_key"] == requested_market_id].copy()
        location = self._location_metadata(history)
        history.sort_values(
            ["_date", "variety", "fetched_at_utc"],
            ascending=[True, True, True],
            kind="stable",
            na_position="last",
            inplace=True,
        )
        records = [self._serialise_record(row) for _, row in history.iterrows()]

        return {
            "status": "ok" if records else "no_observations",
            "crop": _clean_text(crop),
            **location,
            "market_id": requested_market_id,
            "market": (
                _clean_text(history.iloc[0].get("market")) if not history.empty else None
            ),
            "is_live": False,
            "record_count": len(records),
            **self._freshness_metadata(history),
            "records": records,
            "data_quality": self._data_quality(history),
            "message": (
                f"Found {len(records)} archived observations."
                if records
                else f"No observations found for {crop} at market ID {requested_market_id}."
            ),
        }

    def compare_markets(
        self,
        crop: str,
        state_code: str | None = None,
        district_code: str | None = None,
    ) -> dict[str, Any]:
        """Compare latest reported values only across genuinely comparable rows.

        Comparisons are grouped by observation date, price unit, price metric,
        and variety. The method uses each market's latest observation date for
        the crop, including NR rows, so it will not fall back to an older price
        when that market's newest observation is unreported.
        """
        frame = self._matches_crop(self._load_archive(), crop)
        frame = self._filter_geography(frame, state_code, district_code)
        location = self._location_metadata(frame)
        if frame.empty:
            return {
                "status": "no_observations",
                "crop": _clean_text(crop),
                **location,
                "is_live": False,
                "comparisons": [],
                "message": f"No archived observations were found for {crop} in the requested geography.",
            }

        dated = frame.loc[frame["_date"].notna()].copy()
        if dated.empty:
            return {
                "status": "no_valid_dates",
                "crop": _clean_text(crop),
                **location,
                "is_live": False,
                "comparisons": [],
                "message": "No valid observation dates are available for comparison.",
            }

        # Determine the newest observation for each market before filtering
        # numeric values. This prevents stale values from appearing as latest.
        latest_per_market = dated["_date"].groupby(dated["_market_key"]).transform("max")
        latest_rows = dated.loc[dated["_date"] == latest_per_market].copy()
        numeric = latest_rows.loc[latest_rows["_reported_price"].notna()].copy()

        # A missing unit makes even a same-day comparison unsafe. An empty
        # metric/variety is retained as its own exact category, never mapped to
        # a named metric/variety.
        comparable = numeric.loc[numeric["_unit_key"] != ""].copy()
        # Keep comparisons inside the same geography to avoid cross-region ranking.
        group_columns = [
            "_state_key", "_district_key", "_date",
            "_unit_key", "_metric_key", "_variety_key",
        ]

        comparisons: list[dict[str, Any]] = []
        for group_key, group in comparable.groupby(group_columns, dropna=False, sort=True):
            distinct_markets = group["_market_key"].nunique()
            if distinct_markets < 2:
                continue

            group = group.sort_values(
                "_reported_price", ascending=False, kind="stable"
            )
            records = [self._serialise_record(row) for _, row in group.iterrows()]
            highest_price = float(group["_reported_price"].max())
            highest_rows = group.loc[group["_reported_price"] == highest_price]
            highest_markets = sorted(
                {
                    _clean_text(value)
                    for value in highest_rows["market"].tolist()
                    if _clean_text(value)
                }
            )
            state_key, district_key, date_value, unit_key, metric_key, variety_key = group_key

            # Use original display text from the first row for unit/metric/variety.
            first = group.iloc[0]
            comparisons.append(
                {
                    "state": _clean_text(first.get("state")) or None,
                    "district": _clean_text(first.get("district")) or None,
                    "observation_date": date_value.strftime("%Y-%m-%d"),
                    "price_unit": _clean_text(first.get("price_unit")) or None,
                    "price_metric": _clean_text(first.get("price_metric")) or None,
                    "variety": _clean_text(first.get("variety")) or None,
                    "market_count": int(distinct_markets),
                    "highest_reported_value": _format_number(highest_price),
                    "highest_reported_markets": highest_markets,
                    "records_ranked_high_to_low": records,
                }
            )

        comparisons.sort(
            key=lambda item: (item["observation_date"], item["price_unit"] or ""),
            reverse=True,
        )
        latest_records = [
            self._serialise_record(row) for _, row in latest_rows.iterrows()
        ]
        status = "ok" if comparisons else "insufficient_comparable_markets"

        return {
            "status": status,
            "crop": _clean_text(crop),
            **location,
            "is_live": False,
            **self._freshness_metadata(latest_rows),
            "comparison_count": len(comparisons),
            "comparisons": comparisons,
            "latest_observations": latest_records,
            "data_quality": self._data_quality(latest_rows),
            "comparison_basis": [
                "same observation date",
                "same price unit",
                "same price metric label (including unspecified labels)",
                "same variety label",
                "latest observation date for each market; NR is not replaced by an older value",
            ],
            "message": (
                "Comparable market groups are ranked by reported numeric value."
                if comparisons
                else "There are not yet at least two markets with comparable numeric observations "
                "on the same date, unit, metric label, and variety."
            ),
            "caveat": (
                "The reported-price metric is not independently verified. A higher reported value "
                "does not prove that a mandi is the best place to sell; transport, quality, quantity, "
                "fees, and the actual transaction price also matter."
            ),
        }
