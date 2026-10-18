"""
core/epiweek.py
---------------
Brazilian epidemiological weeks (semanas epidemiológicas, SE), as used by
SINAN and the Ministry of Health.

Rules (same as the US MMWR week):
- A week runs Sunday to Saturday.
- Week 1 of a year is the first week that has at least 4 days in that year,
  i.e. the week containing the first Wednesday of January.
- A year therefore has 52 or 53 weeks, and early-January / late-December
  days can belong to the neighbouring epidemiological year.

The source spreadsheet labels its columns S201601 … S201952, which are
(epi_year, epi_week) pairs under these rules.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

_LABEL = re.compile(r"^S?(\d{4})(\d{2})$")


def _week1_start(epi_year: int) -> date:
    """Sunday on which week 1 of epi_year starts."""
    jan1 = date(epi_year, 1, 1)
    # date.weekday(): Monday=0 … Sunday=6  →  days since the previous Sunday
    days_since_sunday = (jan1.weekday() + 1) % 7
    sunday = jan1 - timedelta(days=days_since_sunday)
    # If Jan 1 falls Thu–Sat, that week has < 4 days in the new year.
    return sunday + timedelta(weeks=1) if days_since_sunday >= 4 else sunday


def weeks_in_year(epi_year: int) -> int:
    return (_week1_start(epi_year + 1) - _week1_start(epi_year)).days // 7


def week_start(epi_year: int, epi_week: int) -> date:
    """Sunday that starts the given epidemiological week."""
    if not 1 <= epi_week <= weeks_in_year(epi_year):
        raise ValueError(f"{epi_year} has no epidemiological week {epi_week}")
    return _week1_start(epi_year) + timedelta(weeks=epi_week - 1)


def from_date(d: date) -> tuple[int, int]:
    """(epi_year, epi_week) containing date d."""
    year = d.year + 1
    while _week1_start(year) > d:
        year -= 1
    return year, (d - _week1_start(year)).days // 7 + 1


def add_weeks(epi_year: int, epi_week: int, n: int) -> tuple[int, int]:
    """Move n weeks forward (or backward if n < 0)."""
    return from_date(week_start(epi_year, epi_week) + timedelta(weeks=n))


def parse_iso_label(label: str) -> tuple[int, int]:
    """'2019-30' → (2019, 30), the format used in API query parameters."""
    m = re.match(r"^(\d{4})-(\d{1,2})$", str(label).strip())
    if not m:
        raise ValueError(f"expected an epidemiological week as YYYY-WW, got {label!r}")
    year, week = int(m.group(1)), int(m.group(2))
    week_start(year, week)
    return year, week


def parse_label(label: str) -> tuple[int, int]:
    """'S201601' or '201601' → (2016, 1). Raises ValueError otherwise."""
    m = _LABEL.match(str(label).strip())
    if not m:
        raise ValueError(f"not an epidemiological week label: {label!r}")
    year, week = int(m.group(1)), int(m.group(2))
    week_start(year, week)  # validates the week number
    return year, week
