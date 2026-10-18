"""
Adding new weekly data.

A source yields WeeklyCases / WeeklyOvitrap records; `ingest()` writes them
into the database (inserting or updating), records the source in
data_source, and clears stored forecasts that the new data makes stale.

Available sources:
    CsvCasesSource  — a CSV of weekly cases per neighborhood
                      (python -m src.ingestion cases FILE.csv --source "...")

A live feed (e.g. SINAN extracts, the municipal ovitrap program) is added
by implementing WeeklySource.fetch(); see docs/INGESTION.md.
"""
from src.ingestion.base import IngestReport, WeeklyCases, WeeklyOvitrap, WeeklySource, ingest
from src.ingestion.csv_source import CsvCasesSource

__all__ = ["CsvCasesSource", "IngestReport", "WeeklyCases", "WeeklyOvitrap", "WeeklySource", "ingest"]
