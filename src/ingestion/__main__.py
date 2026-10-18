"""
python -m src.ingestion cases FILE.csv --source "Name of the source" [--license ...] [--citation ...]

Loads weekly cases from a CSV (see src/ingestion/csv_source.py for the
columns), records the source, and clears stored forecasts the new data
makes stale.
"""
import argparse
import asyncio
import logging
from pathlib import Path

from src.core.database import AsyncSessionLocal, engine
from src.ingestion import CsvCasesSource, ingest


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m src.ingestion")
    sub = ap.add_subparsers(dest="kind", required=True)
    p = sub.add_parser("cases", help="weekly cases per neighborhood from a CSV")
    p.add_argument("file", type=Path)
    p.add_argument("--source", required=True, help="name stored in data_source, e.g. 'SMS Natal weekly report'")
    p.add_argument("--license")
    p.add_argument("--citation")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    source = CsvCasesSource(args.file, args.source, args.license, args.citation)

    async def run():
        try:
            async with AsyncSessionLocal() as session, session.begin():
                report = await ingest(session, source)
        finally:
            await engine.dispose()
        weeks = sorted(report.weeks)
        print(f"{report.inserted} inserted, {report.updated} updated, {report.unchanged} unchanged"
              + (f"; weeks {weeks[0][0]}-{weeks[0][1]:02d} to {weeks[-1][0]}-{weeks[-1][1]:02d}" if weeks else "")
              + f"; {report.forecasts_cleared} stored forecasts cleared")
        if report.unknown_neighborhoods:
            print(f"Unknown neighborhoods (skipped): {sorted(report.unknown_neighborhoods)}")
    asyncio.run(run())


if __name__ == "__main__":
    main()
