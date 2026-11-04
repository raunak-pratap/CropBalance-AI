# CropBalance AI — Local Mandi Data Layer v1

## Scope

This is a separate research/data-ingestion path for Dharashiv district. It does not modify `fetcher.py`, `PriceService`, the daily LSTM, or disease detection.

## Current collection source

The collector calls AGMARKNET's weekly commodity-price report:

`GET /prices-and-arrivals/commodity-price/lastweek`

with `marketId`, `stateId`, `commodityId`, and `includeExcel=false`.

The current collector output shape exposes date columns, a variety label and `unitOfPrice`. The source code does not establish that each numeric cell is specifically a modal, minimum or maximum price. Therefore this layer stores the numeric value as `reported_price`, sets `price_metric=unverified`, and marks `price_semantics_verified=false`. Do not use it as a modal-price target until an endpoint-specific field definition has been verified.

The weekly price endpoint used here does not populate arrivals. Arrivals remain null; they must not be converted to zero.

## Geography and initial coverage

- State: Maharashtra (ID 20)
- District: Dharashiv (ID 347)
- Markets: APMC Bhoom (1959), Dharashiv (1655), Kalamb (1599), Murum (1600), Paranda (1960), Tuljapur (1961), Umarga (1962), Washi (2053)
- Crops: Wheat (1), Onion (23), Soyabean (13), Tomato (65)

IDs are the values already used by the project's verified resolver/weekly collection work. Revalidate metadata if AGMARKNET changes its master data.

## Commands

```bash
python scripts/sync_dharashiv_mandi_archive.py
python scripts/audit_dharashiv_market_history.py
python -m unittest tests.test_agmarknet_local_archive -v
```

The collector writes only under `data/raw/agmarknet_weekly/`:

- `dharashiv_prices_v2.csv`: normalized observation archive, including not-reported statuses
- `collection_runs.csv`: one result per market/crop request, including empty and failed requests
- `responses/*.json`: raw JSON response snapshots

The audit writes under `data/processed/dharashiv_mandi_audit/` and does not mutate source observations.

## Data contract

Each observation preserves observation date, collection timestamp, source endpoint, state/district/market/commodity/variety identifiers and labels, raw reported value, parsed numeric value, original unit, quality status, price semantics verification flag, optional arrivals, and raw-response provenance.

`NR`, blank, nonnumeric and nonpositive cells are retained as status-bearing rows with no usable price target. The `reported_price` numeric column is never set to zero as a missing-value substitute.

Exact repeated records are collapsed on re-runs. Different source values for the same observation identity are retained and marked `conflict_flag=true`; the collector does not silently overwrite one with another.

## Audit interpretation

The report counts observations, unique dates, market/crop coverage, date gaps, price/unit/source consistency, arrival availability, duplicates/conflicts and IQR outlier flags. Outlier flags are review-only and are not removed automatically. Calendar gaps do not prove a mandi was expected to trade or report on every missing date.

## Forecasting readiness

The prior weekly sync reported 57 valid observations across four crops and eight mandis (as previously recorded in the project session). This is a small rolling-week archive, not enough to justify separate market-level time-series forecasting. Continue collecting and auditing actual observations first. Do not mix this with the statewide monthly weighted-average dataset or with sampled six-day market data as if they were daily observations.

## Known limitations

- The weekly endpoint's exact price field semantics remain unverified in this layer.
- This endpoint does not provide arrivals through the current parser.
- API failure and no-data responses are not the same as a market reporting zero price; request status is stored separately.
- The raw response files are local research evidence and are ignored by Git; never commit API secrets or large raw archives.
- CAPTCHA-protected endpoints are not bypassed.

## Archive v3 Migration — 4 November 2026

### Canonical archive

- Active archive: `data/raw/agmarknet_weekly/dharashiv_prices_v3.csv`
- Preserved v2 archive: `data/raw/agmarknet_weekly/dharashiv_prices_v2.csv`
- Pre-migration backup: `data/raw/agmarknet_weekly/backups/dharashiv_prices_v2_pre_v3.csv`

### Migration results

| Metric                               | Result |
| ------------------------------------ | -----: |
| Original v2 observations             |    450 |
| Consolidated v3 observations         |    226 |
| Raw responses archived               |    224 |
| Legacy records without raw responses |      2 |
| Missing archived raw files           |      0 |
| Duplicate observation identities     |      0 |

### Provenance tracking

The `provenance_status` column identifies:

- `raw_response_archived`: Raw API response path is recorded and the file was verified during migration.
- `legacy_import_raw_unavailable`: Legacy observation imported without its original raw response.

### Collector behavior

The weekly collector now writes to v3. When v3 exists, it loads v3 without re-importing the legacy CSV. Legacy import remains available when initializing an archive that does not yet exist.

### Validation

- Archive unit tests: 12 passed.
- Weekly collector tests: 2 passed.
- Semantic duplicate consolidation and idempotency verified.
- Original v2 archive and backup preserved.

### Data limitations

Price semantics remain unverified. `NR` remains missing rather than zero. Arrivals are not fabricated. The archive is a local research dataset, not yet a validated daily forecasting dataset.
