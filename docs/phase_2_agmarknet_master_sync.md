# Phase 2, Part 1 — AGMARKNET national source-master synchronization

## Scope

This milestone synchronizes the source-specific AGMARKNET master dimensions already present in the project metadata cache:

- states
- districts
- markets
- commodities and their AGMARKNET commodity groups
- varieties and their linked commodity IDs

It does **not** overwrite CropBalance's canonical crop taxonomy or current state/district registries. AGMARKNET IDs are stored only as AGMARKNET source IDs. Official LGD codes must be synchronized separately from the Ministry of Panchayati Raj Local Government Directory.

## Existing cache inspection

The project cache at `data/agmarknet_metadata.json` contains these source records before the selector rows are removed:

| Dimension | Cached records | Excluded selector row |
|---|---:|---:|
| States | 37 | 1 ("All States/UTs") |
| Districts | 751 | 1 ("All Districts") |
| Markets | 4,176 | 1 ("All Markets") |
| Commodities | 609 | 0 |
| Varieties | 2,154 | 1 ("All Varieties") |

The corresponding normalized counts are 36 states/UTs, 750 districts, 4,175 markets, 609 commodities and 2,153 varieties. These counts describe the repository's cached snapshot; they are not a live coverage guarantee. The sync manifest records a local synchronization timestamp and the SHA-256 hash of the metadata cache.

## Implementation

- `services/agmarknet_master_catalog.py` validates metadata relationships, normalizes the five dimensions, writes versioned JSON files atomically and provides paginated read access.
- `scripts/sync_agmarknet_master.py` defaults to a dry run. It reads the cached metadata, validates IDs and parent relationships, and prints the counts. Use `--write` to write normalized data.
- `--refresh` asks the existing AGMARKNET metadata resolver to fetch the filter metadata endpoint before validation. Use `--refresh --write` to refresh the cache and write a new normalized snapshot.
- Generated files are written to `data/reference/source_masters/agmarknet/current/`, which is already inside the Docker read-only reference-data mount. The generated directory is ignored by Git because it is reproducible from the metadata cache.
- FastAPI source-catalog routes expose the synchronized source masters under `/source-catalog/agmarknet/*`. They return HTTP 503 until the snapshot has been written.

## Safe workflow

From the repository root on the host:

```bash
# 1. Validate the existing cached master; writes nothing.
python scripts/sync_agmarknet_master.py

# 2. Inspect the five counts and relationship checks.
# 3. Write the validated local source-master files.
python scripts/sync_agmarknet_master.py --write

# 4. Optional: refresh from the AGMARKNET metadata endpoint, then write.
python scripts/sync_agmarknet_master.py --refresh --write

# 5. Restart only the app container; no PyTorch image rebuild is required.
docker compose up -d --force-recreate app
```

The script's default is intentionally non-destructive. Do not use `--refresh` until you are ready to replace the local metadata cache with the endpoint's latest response.

## Source-master API

After the snapshot is written:

- `GET /source-catalog/agmarknet/coverage`
- `GET /source-catalog/agmarknet/states`
- `GET /source-catalog/agmarknet/districts?state_id=20`
- `GET /source-catalog/agmarknet/markets?state_id=20&district_id=347`
- `GET /source-catalog/agmarknet/commodities?group_id=1`
- `GET /source-catalog/agmarknet/varieties?commodity_id=13`

List endpoints support `q`, `offset` and `limit` (maximum 500). The `state_id`, `district_id`, `commodity_id` and `group_id` filters are AGMARKNET source IDs, not CropBalance internal codes or LGD codes.

## Validation

```bash
python -m pytest -q tests/test_agmarknet_master_catalog.py
python -m pytest -q
```

The test suite uses synthetic fixtures and does not make external requests. It checks reserved selector-row removal, unique source IDs, parent consistency, variety-to-commodity links, atomic snapshot writes and paginated filtering.

## Not yet included

This milestone does not yet:
- synchronize official LGD codes;
- ingest daily mandi price observations into a national fact store;
- assert that each market reports every commodity;
- map all 609 AGMARKNET commodities to CropBalance's curated crop taxonomy;
- declare any crop-region forecast model validated.

The official OGD daily mandi price resource describes wholesale minimum, maximum and modal prices at daily granularity. The daily data connector and durable observation store are the next implementation step, after the source masters are reviewed.

## Official references

- AGMARKNET: https://agmarknet.gov.in/
- OGD current daily mandi prices: https://data.gov.in/resource/current-daily-price-various-commodities-various-markets-mandi
- OGD variety-wise daily market prices: https://data.gov.in/resource/variety-wise-daily-market-prices-data-commodity
- LGD directory: https://lgdirectory.gov.in/
- OGD Local Government Directory catalog: https://data.gov.in/catalog/local-government-directory-lgd
- OGD LGD districts: https://data.gov.in/resource/local-government-directory-lgd-districts
