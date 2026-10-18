# Adding new data

The research data ends in epidemiological week 2019-52. New weeks enter through
`src/ingestion`, which every source shares.

## From a CSV (available now)

A spreadsheet export with one row per neighborhood and week:

| Column | Meaning |
|---|---|
| `neighborhood` | official municipal code (e.g. `007`) or name (accents and spelling variants are matched) |
| `epi_year`, `epi_week` | epidemiological year and week (Sunday–Saturday, as in SINAN) |
| `cases` | notified cases; blank if unknown |
| `disease` | optional, default `dengue` |

```bash
python -m src.ingestion cases weekly.csv --source "SMS Natal weekly report" --license "..." --citation "..."
```

What happens:
- New rows are inserted, changed rows updated, identical rows skipped; a second run changes nothing.
- Unknown neighborhoods are skipped and listed.
- The source is recorded in `data_source` and linked from every row it wrote.
- Stored forecasts made as of the earliest changed week or later are deleted (they were
  computed without the new data). Replays of earlier weeks stay valid.

A week only becomes the default forecast week once it is **complete**: all neighborhoods
reported cases and the model's input exists (ovitrap EDI for EDI models). Until then forecasts
use the latest complete week, and asking for the partial week returns `422` with what is missing.

## A live source

Implement `WeeklySource` (`src/ingestion/base.py`): a `name`, `file_or_url`, `license`,
`citation`, and `fetch()` returning `WeeklyCases` and/or `WeeklyOvitrap` records. Then:

```python
async with AsyncSessionLocal() as session, session.begin():
    report = await ingest(session, MySource())
```

Candidates to evaluate: weekly SINAN extracts from the municipal health secretariat, the
municipal ovitrap program, and InfoDengue (city-level series; check coverage for Natal and
whether it has neighborhood detail before relying on it).

The models were trained on 2016–2019. Forecasts for later weeks use the same models; after
several months of new data, retrain (`python -m src.ml.train`) and compare with the previous run
(`python -m src.ml.evaluate --run …`) before switching.
