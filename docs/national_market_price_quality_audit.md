# National market-price quality audit

## Purpose

The quality auditor inspects records already persisted in the
\`market_price_observations\` table. It is deliberately separate from the
Dharashiv AGMARKNET weekly research archive and from the ingestion pipeline.

The audit is read-only. It does not call data.gov.in or AGMARKNET, create or
alter tables, update observations, delete duplicates, or write to PostgreSQL.

## Run

Configure the existing \`DATABASE_URL\` for the intended database, then run:

\`\`\`bash
python scripts/audit_national_market_prices.py
\`\`\`

Optional filters are case-insensitive contains matches:

\`\`\`bash
python scripts/audit_national_market_prices.py \\
  --state Maharashtra \\
  --district Dharashiv \\
  --commodity Wheat \\
  --start-date 2026-10-01 \\
  --end-date 2026-11-15
\`\`\`

Limit or suppress calendar-gap examples:

\`\`\`bash
python scripts/audit_national_market_prices.py --max-gap-examples 5
python scripts/audit_national_market_prices.py --max-gap-examples 0
\`\`\`

The command prints a JSON report to standard output and returns a non-zero exit
code if the table is not initialized or the audit query fails. Use the
application's database URL intentionally: the report describes the database
that URL selects.

## Report sections

- \`coverage\`: persisted observation count, distinct states/districts/markets/
  commodities, first/last observation dates, latest retrieval time, and age at
  the time of the audit.
- \`quality\`: missing min/max/modal values, non-positive prices, price-order
  violations, missing price units, negative arrivals, and basic provenance
  gaps. Missing arrivals are counted but are not automatically a defect because
  the source may not provide them.
- \`distributions\`: stored price-unit and source counts. Different units must
  not be combined in analytics until their semantics are verified.
- \`duplicates\`: groups with the same source/resource/date/geography/commodity/
  variety/grade dimensions. These are review candidates, not confirmed errors:
  multiple source rows may require investigation. The auditor never chooses
  which record to keep.
- \`temporal_coverage\`: date series by source/resource/market/commodity/
  variety/grade, the number of calendar-day gaps between stored observations,
  and a few largest examples.

## Interpretation safeguards

1. Coverage counts describe only records stored by CropBalance. They do not
   establish the number of active mandis or prove complete national coverage.
2. A calendar gap is elapsed time between two stored observations. It does not
   prove a mandi was expected to report on every intervening date; market
   holidays, trading schedules, source publication behavior and collection
   cadence can all affect reporting.
3. Missing prices and invalid prices are different conditions. A NULL field is
   not imputed. A reported zero/negative price or invalid min/modal/max order
   is flagged for investigation.
4. Arrivals are optional in the current daily-price source. The audit counts
   missing values for visibility but does not classify them as errors.
5. Duplicate candidates are exposed without deduplication or mutation. Do not
   delete or overwrite source observations solely because they appear in a
   candidate group.
6. Data-quality results are diagnostic; they do not certify that prices are
   economically correct or comparable across commodities, grades or units.

## Tests

\`\`\`bash
python -m pytest -q tests/test_national_market_price_quality_audit.py
python -m pytest -q
\`\`\`

Tests use in-memory SQLite fixtures and synthetic rows. They do not require a
production database, external network access or API credentials.
