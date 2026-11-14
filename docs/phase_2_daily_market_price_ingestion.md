# Phase 2, Part 2 — Daily market-price ingestion

## Goal

Persist individual daily mandi-price observations from the OGD India resource:

- Resource: `9ef84268-d588-465a-a308-a864a43d0070`
- Dataset: Current Daily Price of Various Commodities from Various Markets (Mandi)
- Publisher: Directorate of Marketing and Inspection, Ministry of Agriculture and Farmers Welfare
- Resource URL: https://data.gov.in/resource/current-daily-price-various-commodities-various-markets-mandi
- API endpoint: `https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070`

The dataset is described by OGD India as daily wholesale minimum, maximum and modal market prices. The connector stores each market record separately instead of reusing the older forecasting fetcher's state/day aggregation.

## Components

- `services/data_gov_in_daily_prices.py`
  - OGD API client with offset pagination and a configurable maximum page count.
  - Field-name normalization for common API casing variants.
  - Record-level parsing for arrival date, state, district, market, commodity, variety and grade.
  - Independent minimum, maximum and modal price fields.
  - Optional arrivals and arrival unit when the source record actually supplies them.
  - Deterministic record keys and raw-payload hashes.
  - Rejects malformed rows and price-order violations; retains missing values as `NULL`.

- `database/market_price_store.py`
  - PostgreSQL table `market_price_observations`.
  - Unique deterministic `record_key`.
  - PostgreSQL `ON CONFLICT DO UPDATE` upserts so a corrected source report replaces the earlier version of the same observation.
  - Original `created_at` retained; `updated_at`, source retrieval time and payload hash refreshed.
  - Indexes for geography/commodity/date and market/date.
  - SQLite support for unit tests.

- `scripts/ingest_daily_mandi_prices.py`
  - Default is a dry run (fetches and validates, but does not write to PostgreSQL).
  - `--write` creates only the new table if necessary and upserts records.
  - Optional exact state/commodity filters.
  - Optional date range filter (applied after retrieval).
  - Partial/capped runs require an explicit `--allow-partial` acknowledgement.
  - Incomplete pagination without the explicit partial acknowledgement never writes.

- FastAPI:
  - `GET /market-prices/coverage`
  - `GET /market-prices/observations`
  - API reads do not call OGD India. They read only the PostgreSQL observation store.
  - Endpoints return HTTP 503 until the observation table exists.

## Setup

The API key must be stored in `.env` and must not be committed:

```dotenv
DATA_GOV_IN_API_KEY=your_data_gov_in_api_key
```

The project already has a `DATA_GOV_IN_API_KEY` setting and SQLAlchemy/PostgreSQL dependencies.

## Safe first run

From the repository root:

```bash
# 1. Run normalization and pagination as a dry run.
python scripts/ingest_daily_mandi_prices.py

# Optional small test fetch for inspecting a known commodity/state.
python scripts/ingest_daily_mandi_prices.py \
  --state Maharashtra \
  --commodity Soyabean \
  --max-records 100

# 2. Once the summary and rejected rows have been reviewed, run a complete write.
python scripts/ingest_daily_mandi_prices.py --write
```

For a full national run, do not pass `--state`, `--commodity` or `--max-records`. If the resource has more pages than the configured limit, increase `--max-pages`; the script refuses to present an incomplete paginated run as complete.

If the CLI reports rejected records, inspect the sample reasons before writing. The CLI prints at most 20 examples and never prints the API key.

## Database and API checks

After a successful `--write`, restart only the app container if it is running:

```bash
docker compose up -d --force-recreate app
curl -sS http://localhost:8000/market-prices/coverage | python -m json.tool
curl -sS "http://localhost:8000/market-prices/observations?state=Maharashtra&commodity=Soyabean&limit=20" | python -m json.tool
```

Query parameters:

- `state`, `district`, `market`, `commodity`: case-insensitive contains search.
- `start_date`, `end_date`: ISO `YYYY-MM-DD`.
- `offset`: zero-based paging offset.
- `limit`: 1–500.

## Data-quality and modeling rules

1. Do not aggregate across markets in the ingestion/store layer.
2. Keep variety and grade as identity dimensions.
3. Keep minimum, maximum and modal price as separate fields.
4. Do not infer modal price from other prices.
5. A non-positive reported price is rejected; a missing price remains NULL.
6. Validate `min_price <= modal_price <= max_price` where the relevant values are present.
7. Arrival quantity and unit are optional. The current OGD current-daily-price resource does not guarantee an arrivals field; the connector only stores arrival values if explicitly supplied by a source record. It does not convert unknown units to tonnes.
8. Preserve the raw API row, source resource ID, source IDs when present, retrieval timestamp and payload hash.
9. The record key excludes price values so corrected prices upsert the same observation.
10. A successful fetch does not prove national completeness. Coverage endpoints report only records stored by CropBalance.

## Known limitations

- OGD API key required.
- Pagination is bounded by `--max-pages`; a run that reaches the limit is incomplete and will not write by default.
- Optional source IDs are stored only if returned by the API. Name-to-ID crosswalks are not guessed.
- The API resource is described as a current daily dataset. Repeated daily snapshots capture the records the resource exposes at collection time; this is not automatically a full historic backfill.
- Arrivals may be absent from this price resource; a separately verified arrivals source/connector may be needed.
- The existing `fetcher.py` function still performs a separate state/day aggregation for its legacy LSTM workflow. This new store is the record-level source of truth for market intelligence and future training dataset work.

## Tests

```bash
python -m pytest -q tests/test_daily_market_price_ingestion.py
python -m pytest -q
```

Focused tests use mocked OGD API pages and in-memory SQLite. No real API key or external network call is needed.
