# Crop data readiness matrix

## Purpose

The readiness matrix makes four separate claims visible for each crop:

1. **Catalog presence** — the crop exists in CropBalance's starter taxonomy.
2. **Source mapping** — the crop has a verified external ID in a named provider namespace, such as AGMARKNET.
3. **Observed data** — records for the crop exist in the configured local mandi archive, with price semantics and quality counts shown separately.
4. **Model availability and validation** — checkpoint/scaler files are reported independently from forecast validation.

A crop's presence in the taxonomy does not imply source coverage, stored observations, training-ready prices or a validated forecast model.

## API

`GET /catalog/readiness` returns the readiness report for all 72 starter crops.

Optional crop filter:

```bash
curl "http://localhost:8000/catalog/readiness?crop=soyabean"
```

The filter accepts a canonical internal crop code, English display name or registered alias.

## Observation scope

The first version reads the path selected by `CROPBALANCE_MANDI_ARCHIVE`, or the default `data/raw/agmarknet_weekly/dharashiv_prices_v3.csv`.

This is the Dharashiv AGMARKNET weekly research archive, not the national `market_price_observations` PostgreSQL table. The matrix reports:
- observation rows and positive numeric price rows;
- rows whose `price_semantics_verified` field is explicitly true;
- rows whose price semantics remain unverified;
- market count, source, price units, date range and conflict count.

Missing, invalid, zero or negative prices do not count as numeric price observations. Only positive numeric prices explicitly marked as semantically verified count as verified price targets. The report does not promote the weekly archive's currently unverified metric to a modal price.

National database coverage remains available separately at `GET /market-prices/coverage`; that endpoint requires the national observation table to exist.

## Model interpretation

The service checks for:
- `models/saved/best_<crop_code>.pt`
- `models/scalers/scalers_<crop_code>.pkl`

It reports whether both artifacts exist. **It does not mark any model validated merely because these files exist.** A validated model requires reviewed, time-ordered holdout evaluation and comparison with a baseline. The current matrix therefore reports `validated_model.available=false` until that evidence is formally registered.

## Readiness stages

- `catalog_only`
- `source_mapped`
- `observations_present_semantics_unverified`
- `verified_price_observations_available`
- `model_artifacts_present_unvalidated`
- `forecast_validated` (reserved for explicit future validation evidence)

The detailed fields remain authoritative when more than one capability is present.

## Safety and limitations

- API reads are local and read-only; no OGD, AGMARKNET or database write is triggered.
- Missing local archive files are reported as unavailable instead of failing the full catalog report.
- Provider IDs stay in their source namespaces. AGMARKNET IDs are not treated as LGD codes.
- The matrix does not infer national coverage from the Dharashiv pilot archive.
- The matrix does not certify a model's accuracy or recommend using a forecast.

## Tests

```bash
python -m pytest -q tests/test_crop_readiness_service.py
python -m pytest -q
```
