"""Record-level OGD/data.gov.in daily mandi-price connector.

Unlike fetcher.MandiPriceFetcher._fetch_data_gov_in, this connector does not
aggregate markets into daily state averages. It preserves each market,
commodity, variety and grade observation for storage and audit.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

import requests


RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
RESOURCE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
SOURCE_NAME = "data.gov.in"
DEFAULT_PAGE_SIZE = 1000
MAX_PAGE_SIZE = 1000
DEFAULT_MAX_PAGES = 1000

FIELD_ALIASES = {
    "observation_date": ("arrivaldate", "date", "observeddate"),
    "state": ("state", "statename"),
    "district": ("district", "districtname"),
    "market": ("market", "marketname", "apmc", "apmcname"),
    "commodity": ("commodity", "commodityname", "crop", "cropname"),
    "variety": ("variety", "varietyname"),
    "grade": ("grade", "gradename"),
    "min_price": ("minprice", "minimumprice"),
    "max_price": ("maxprice", "maximumprice"),
    "modal_price": ("modalprice", "modal"),
    "price_unit": ("priceunit", "unitofprice"),
    "arrival_quantity": (
        "arrivalquantity", "arrivalqty", "arrivalsquantity",
        "arrivals", "arrival", "arrivalstonnes",
    ),
    "arrival_unit": ("arrivalunit", "arrivalsunit", "unitofarrival"),
    "source_record_id": ("recordid", "id"),
    "source_state_id": ("stateid",),
    "source_district_id": ("districtid",),
    "source_market_id": ("marketid",),
    "source_commodity_id": ("commodityid", "cmdtid"),
    "source_variety_id": ("varietyid",),
}


@dataclass
class FetchResult:
    records: list[dict[str, Any]]
    pages_fetched: int
    reported_total: int | None
    pagination_complete: bool
    stop_reason: str


@dataclass
class NormalizationResult:
    records: list[dict[str, Any]]
    rejected_count: int = 0
    duplicate_count: int = 0
    out_of_range_count: int = 0
    issues: list[dict[str, Any]] = field(default_factory=list)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _field_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _index_record(record: dict[str, Any]) -> dict[str, Any]:
    return {_field_key(key): value for key, value in record.items()}


def _pick(indexed: dict[str, Any], field_name: str) -> Any:
    for alias in FIELD_ALIASES[field_name]:
        value = indexed.get(alias)
        if value is not None and str(value).strip() not in {"", "-", "--"}:
            return value
    return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    result = " ".join(str(value).split()).strip()
    if not result or result.casefold() in {"na", "n/a", "nr", "null", "none"}:
        return None
    return result


def _decimal(value: Any, field_name: str) -> Decimal | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw.casefold() in {"na", "n/a", "nr", "null", "none", "-", "--"}:
        return None
    cleaned = raw.replace(",", "").replace("₹", "").strip()
    cleaned = re.sub(r"^(rs\.?|inr)\s*", "", cleaned, flags=re.IGNORECASE)
    try:
        number = Decimal(cleaned)
    except InvalidOperation as exc:
        raise ValueError(f"{field_name} is not numeric: {raw!r}") from exc
    if not number.is_finite():
        raise ValueError(f"{field_name} must be finite.")
    return number


def _parse_date(value: Any) -> date:
    if value is None:
        raise ValueError("arrival_date is missing.")
    raw = str(value).strip()
    for pattern in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, pattern).date()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError as exc:
        raise ValueError(f"arrival_date has unsupported format: {raw!r}") from exc


def _identity_text(value: str | None) -> str:
    return " ".join((value or "").casefold().split())


def _hash_payload(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def normalize_market_price_record(
    raw: dict[str, Any],
    *,
    retrieved_at: datetime | None = None,
) -> dict[str, Any]:
    """Convert one API row to a validated, loss-aware market observation."""
    if not isinstance(raw, dict):
        raise ValueError("record must be a JSON object.")

    indexed = _index_record(raw)
    observation_date = _parse_date(_pick(indexed, "observation_date"))
    state = _text(_pick(indexed, "state"))
    district = _text(_pick(indexed, "district"))
    market = _text(_pick(indexed, "market"))
    commodity = _text(_pick(indexed, "commodity"))

    missing = [
        name for name, value in (
            ("state", state),
            ("district", district),
            ("market", market),
            ("commodity", commodity),
        ) if value is None
    ]
    if missing:
        raise ValueError("record is missing required dimensions: " + ", ".join(missing))

    variety = _text(_pick(indexed, "variety"))
    grade = _text(_pick(indexed, "grade"))

    min_price = _decimal(_pick(indexed, "min_price"), "min_price")
    max_price = _decimal(_pick(indexed, "max_price"), "max_price")
    modal_price = _decimal(_pick(indexed, "modal_price"), "modal_price")
    arrival_quantity = _decimal(
        _pick(indexed, "arrival_quantity"), "arrival_quantity"
    )

    for name, value in (
        ("min_price", min_price),
        ("max_price", max_price),
        ("modal_price", modal_price),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be greater than zero when reported.")
    if arrival_quantity is not None and arrival_quantity < 0:
        raise ValueError("arrival_quantity cannot be negative.")

    if min_price is not None and max_price is not None and min_price > max_price:
        raise ValueError("min_price cannot exceed max_price.")
    if min_price is not None and modal_price is not None and min_price > modal_price:
        raise ValueError("min_price cannot exceed modal_price.")
    if max_price is not None and modal_price is not None and modal_price > max_price:
        raise ValueError("modal_price cannot exceed max_price.")

    if all(value is None for value in (min_price, max_price, modal_price, arrival_quantity)):
        raise ValueError("record has no reported price or arrival quantity.")

    source_record_id = _text(_pick(indexed, "source_record_id"))
    source_ids = {
        key: _text(_pick(indexed, field_name))
        for key, field_name in (
            ("state", "source_state_id"),
            ("district", "source_district_id"),
            ("market", "source_market_id"),
            ("commodity", "source_commodity_id"),
            ("variety", "source_variety_id"),
        )
    }

    identity = {
        "source": SOURCE_NAME,
        "resource_id": RESOURCE_ID,
        "source_record_id": source_record_id,
        "observation_date": observation_date.isoformat(),
        "state_id": source_ids["state"],
        "district_id": source_ids["district"],
        "market_id": source_ids["market"],
        "commodity_id": source_ids["commodity"],
        "variety_id": source_ids["variety"],
        "state": source_ids["state"] or _identity_text(state),
        "district": source_ids["district"] or _identity_text(district),
        "market": source_ids["market"] or _identity_text(market),
        "commodity": source_ids["commodity"] or _identity_text(commodity),
        "variety": source_ids["variety"] or _identity_text(variety),
        "grade": _identity_text(grade),
    }
    # Use a true source row ID when provided; otherwise use the natural
    # market/date/commodity/variety/grade key. Prices are deliberately excluded
    # so corrected reports update the same observation.
    if source_record_id is None:
        identity.pop("source_record_id")
    record_key = _hash_payload(identity)
    retrieved = retrieved_at or utc_now()

    normalized = {
        "record_key": record_key,
        "source": SOURCE_NAME,
        "source_resource_id": RESOURCE_ID,
        "source_record_id": source_record_id,
        "observation_date": observation_date,
        "state": state,
        "district": district,
        "market": market,
        "commodity": commodity,
        "variety": variety,
        "grade": grade,
        "source_state_id": source_ids["state"],
        "source_district_id": source_ids["district"],
        "source_market_id": source_ids["market"],
        "source_commodity_id": source_ids["commodity"],
        "source_variety_id": source_ids["variety"],
        "min_price": min_price,
        "max_price": max_price,
        "modal_price": modal_price,
        "price_unit": _text(_pick(indexed, "price_unit")),
        "arrival_quantity": arrival_quantity,
        "arrival_unit": _text(_pick(indexed, "arrival_unit")),
        "arrivals_available": arrival_quantity is not None,
        "retrieved_at": retrieved,
        "source_updated_at": None,
        "raw_record": raw,
    }
    normalized["payload_hash"] = _hash_payload(raw)
    return normalized


def normalize_market_price_records(
    raw_records: Iterable[dict[str, Any]],
    *,
    retrieved_at: datetime | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> NormalizationResult:
    """Normalize records, filter an optional date range, and deduplicate by key."""
    unique: dict[str, dict[str, Any]] = {}
    rejected_count = 0
    duplicate_count = 0
    out_of_range_count = 0
    issues: list[dict[str, Any]] = []

    for index, raw in enumerate(raw_records):
        try:
            record = normalize_market_price_record(raw, retrieved_at=retrieved_at)
            if start_date and record["observation_date"] < start_date:
                out_of_range_count += 1
                continue
            if end_date and record["observation_date"] > end_date:
                out_of_range_count += 1
                continue
            if record["record_key"] in unique:
                duplicate_count += 1
            unique[record["record_key"]] = record
        except (TypeError, ValueError) as exc:
            rejected_count += 1
            if len(issues) < 20:
                issues.append({"index": index, "reason": str(exc)})
    return NormalizationResult(
        records=list(unique.values()),
        rejected_count=rejected_count,
        duplicate_count=duplicate_count,
        out_of_range_count=out_of_range_count,
        issues=issues,
    )


class DataGovInDailyPriceClient:
    """Paginated client for OGD resource 9ef84268-d588-465a-a308-a864a43d0070."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = RESOURCE_URL,
        session: requests.Session | None = None,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key or os.getenv("DATA_GOV_IN_API_KEY", "")
        self.base_url = base_url
        self.session = session or requests.Session()
        self.timeout = timeout

    def fetch(
        self,
        *,
        state: str | None = None,
        commodity: str | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_pages: int = DEFAULT_MAX_PAGES,
        max_records: int | None = None,
    ) -> FetchResult:
        if not self.api_key:
            raise RuntimeError("DATA_GOV_IN_API_KEY is not configured.")
        if page_size < 1 or page_size > MAX_PAGE_SIZE:
            raise ValueError(f"page_size must be between 1 and {MAX_PAGE_SIZE}.")
        if max_pages < 1:
            raise ValueError("max_pages must be at least 1.")
        if max_records is not None and max_records < 1:
            raise ValueError("max_records must be at least 1.")

        params: dict[str, Any] = {
            "api-key": self.api_key,
            "format": "json",
            "limit": page_size,
            "offset": 0,
        }
        if state:
            params["filters[State]"] = state
        if commodity:
            params["filters[Commodity]"] = commodity

        records: list[dict[str, Any]] = []
        reported_total: int | None = None
        pages = 0
        stop_reason = "unknown"

        for _ in range(max_pages):
            response = self.session.get(
                self.base_url, params=params.copy(), timeout=self.timeout
            )
            response.raise_for_status()
            try:
                payload = response.json()
            except (TypeError, ValueError) as exc:
                raise RuntimeError("data.gov.in returned invalid JSON.") from exc

            if not isinstance(payload, dict):
                raise RuntimeError("Invalid data.gov.in response: expected a JSON object.")
            if "records" not in payload:
                message = payload.get("message") or payload.get("error") or "records field is missing"
                raise RuntimeError(f"data.gov.in API error: {message}")

            page = payload["records"]
            if not isinstance(page, list):
                raise RuntimeError("Invalid data.gov.in response: 'records' must be a list.")
            if any(not isinstance(row, dict) for row in page):
                raise RuntimeError("Invalid data.gov.in response: every record must be a JSON object.")

            pages += 1
            total_value = payload.get("total")
            if total_value is not None:
                try:
                    reported_total = int(total_value)
                except (TypeError, ValueError):
                    raise RuntimeError(f"Invalid data.gov.in total value: {total_value!r}")

            remaining = None if max_records is None else max_records - len(records)
            if remaining is not None:
                page = page[:remaining]
            records.extend(page)

            if max_records is not None and len(records) >= max_records:
                stop_reason = "max_records_reached"
                return FetchResult(records, pages, reported_total, False, stop_reason)
            if not payload["records"]:
                stop_reason = "empty_page"
                return FetchResult(records, pages, reported_total, True, stop_reason)
            params["offset"] += len(payload["records"])

            if reported_total is not None and params["offset"] >= reported_total:
                stop_reason = "reported_total_reached"
                return FetchResult(records, pages, reported_total, True, stop_reason)
            if len(payload["records"]) < page_size:
                stop_reason = "short_page"
                return FetchResult(records, pages, reported_total, True, stop_reason)

        stop_reason = "max_pages_reached"
        return FetchResult(records, pages, reported_total, False, stop_reason)
