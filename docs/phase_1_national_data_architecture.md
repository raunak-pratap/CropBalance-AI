# Phase 1 — National reference data architecture

## Purpose

Phase 1 establishes provider-neutral reference registries so CropBalance AI can grow from the Dharashiv pilot to multiple Indian states, districts, markets and crops without embedding geographic IDs or crop IDs in application logic.

This phase is a **reference-data foundation**, not a claim that nationwide price observations are already ingested.

## What is included

| Registry | Current seed | Coverage status |
|---|---:|---|
| States and Union Territories | 36 (28 states + 8 UTs) | National name seed; official LGD codes are not yet synchronized |
| Crop taxonomy | Starter catalogue across cereals, pulses, oilseeds, fibre, sugar/starch, vegetables, fruits and plantation/spice crops | Not exhaustive; source commodity crosswalk is partial |
| Districts | Dharashiv | Pilot only |
| Markets | 8 Dharashiv APMCs | Pilot only |
| Source mappings | Existing project-verified AGMARKNET IDs | Only known IDs are populated |

The seeded crop taxonomy is designed for expansion. It does not represent every crop, horticulture commodity, local crop name, cultivar or market variety grown in India.

## Identifier strategy

Every entity has a stable internal CropBalance key, such as `soybean`, `maharashtra`, `dharashiv` or `apmc_kalamb`.

External identifiers are stored separately in `source_mappings.json` as:

- `entity_type`
- `entity_code`
- `source`
- `external_id`
- `verification_status`
- `evidence`

This prevents an AGMARKNET state ID from being mistaken for an LGD, Census or another provider's ID. Missing external IDs remain absent; they are never fabricated.

### Existing project-verified crosswalks

- AGMARKNET commodity IDs: wheat 1, soybean 13, onion 23, tomato 65.
- AGMARKNET state ID: Maharashtra 20.
- AGMARKNET district ID: Dharashiv 347.
- AGMARKNET market IDs: Bhoom 1959, Dharashiv 1655, Kalamb 1599, Murum 1600, Paranda 1960, Tuljapur 1961, Umarga 1962, Washi 2053.

The Dharashiv district ID above is the ID used by the existing AGMARKNET project metadata; it is **not asserted to be an LGD district code**.

## Code

- `services/reference_registry.py`: loads and validates reference data, resolves canonical codes and aliases, lists entities, exposes namespaced external IDs and reports coverage.
- `services/mandi_archive_service.py`: public generic archive-query import. `MandiArchiveService` supports optional `state_code` and `district_code` filters; `LocalMandiService` remains as a backward-compatible alias for the Dharashiv pilot.
- `CROPBALANCE_MANDI_ARCHIVE`: optional environment variable to point the generic service at another canonical observation CSV. If unset, it continues to use the Dharashiv pilot archive.
- `data/reference/crops.json`: crop taxonomy and English/Hindi/Marathi display names for the starter set.
- `data/reference/states.json`: Indian state and UT name seed.
- `data/reference/districts.json`: seeded district entries.
- `data/reference/markets.json`: seeded market entries.
- `data/reference/source_mappings.json`: provider-specific external-ID crosswalks.

The registry validates unique internal keys, district-to-state links, market-to-district links, and external-ID uniqueness within a source namespace.

## API

- `GET /catalog/coverage`
- `GET /catalog/crops?category=oilseeds&language=hi&q=soya`
- `GET /catalog/states?q=maha`
- `GET /catalog/districts?state_code=maharashtra`
- `GET /catalog/markets?state_code=maharashtra&district_code=dharashiv`

The `state_code` and `district_code` query parameters use CropBalance internal keys, not official government codes.

The existing `/markets/local` and `/mandi/prices/*` endpoints remain available for backward compatibility. They now accept optional `state_code` and `district_code` filters using CropBalance internal keys; they can resolve registered names and aliases too. The catalog endpoints are the first general discovery layer. National ingestion, LGD synchronization and a maintained all-India observation archive are subsequent data-engineering work.

## Data-source policy and next synchronization work

1. **Administrative hierarchy:** synchronize state and district codes from the Ministry of Panchayati Raj Local Government Directory (LGD). LGD is the authoritative maintained directory for administrative geography and assigns codes to entities. Do not substitute project or AGMARKNET IDs for LGD codes.
2. **Market/commodity master:** synchronize the AGMARKNET source's current commodity, variety, state, district and market master data, preserving source IDs and retrieval timestamps.
3. **Daily price facts:** prefer documented minimum, maximum and modal price fields from the daily AGMARKNET/OGD resource when available. Store metric, unit, variety, grade, arrival date, arrivals, source and ingestion time independently.
4. **Coverage states:** distinguish catalogue presence, source mapping, observed data availability and validated-model availability. A crop in the starter catalogue does not imply data or forecast coverage.
5. **Refresh and audit:** treat official master data as versioned inputs; keep each import's source URL, fetched timestamp, checksum and validation report.

Useful sources:

- Ministry of Panchayati Raj LGD: https://lgdirectory.gov.in/
- OGD LGD catalogue: https://data.gov.in/catalog/local-government-directory-lgd
- OGD LGD district data: https://data.gov.in/resource/local-government-directory-lgd-districts
- OGD daily mandi-price resource: https://data.gov.in/resource/current-daily-price-various-commodities-various-markets-mandi
- AGMARKNET: https://agmarknet.gov.in/

The OGD daily mandi-price catalogue describes daily wholesale minimum, maximum and modal prices. Actual endpoint fields and semantics must still be checked on each ingest connector before loading or training.

## Validation

Run:

```bash
python -m pytest -q tests/test_reference_registry.py
python -m pytest -q
```

A successful registry validation confirms schema and hierarchy consistency only; it does not certify national market-data completeness.
