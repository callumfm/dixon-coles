"""Loader for the optional results data shipped alongside this repository.

The CSVs live in ``data/<competition>/`` at the repository root, not inside the
package, so they are available when you clone but are not installed as part of
the library. Point ``data_dir`` at any directory of ``<season>/results.csv``
files to use a different competition or your own data.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from dixon_coles._types import Results

DATE_FORMAT = "%Y-%m-%d"
REPO_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_COMPETITION = "premier-league"


def load_results(
    seasons: str | list[str] | None = None,
    data_dir: Path | str | None = None,
) -> Results:
    """Load results for the given seasons (default: all), sorted by date.

    ``seasons`` is one season (``"2425"``), a list of them, or None for all;
    returns numpy arrays keyed exactly as ``DixonColesModel.fit`` expects.
    """
    return _rows_to_results(_load_rows(seasons=seasons, data_dir=data_dir))


def available_seasons(data_dir: Path | str | None = None) -> list[str]:
    """List the seasons present in the data directory, earliest first.

    Ordering is lexicographic, so season directory names must sort
    chronologically — true for the bundled ``1617``-style names, but a
    scheme spanning a century boundary (``9900`` vs ``0001``) would not,
    and ``backtest``/``tune_hyperparams`` rely on this order for their
    train-on-the-past splits.
    """
    return _list_seasons(_resolve(data_dir))


def _load_rows(
    seasons: str | list[str] | None = None,
    data_dir: Path | str | None = None,
) -> list[dict[str, str]]:
    """Read raw result rows for the given seasons, sorted by date.

    Each row keeps every CSV column (including ``gameweek``) plus a
    ``season`` key naming the directory it came from.
    """
    root = _resolve(data_dir)
    found = _list_seasons(root)

    if seasons is None:
        selected = found
    elif isinstance(seasons, str):
        selected = [seasons]
    else:
        selected = list(seasons)

    unknown = [s for s in selected if s not in found]
    if unknown:
        raise ValueError(
            f"No results found for season(s): {', '.join(sorted(unknown))} "
            f"in {root}. Available: {', '.join(found)}"
        )

    rows: list[dict[str, str]] = []
    for season in selected:
        with (root / season / "results.csv").open(encoding="utf-8") as f:
            rows.extend({**r, "season": season} for r in csv.DictReader(f))

    rows.sort(key=lambda r: r["date"])
    return rows


def _rows_to_results(rows: list[dict[str, str]]) -> Results:
    """Convert raw result rows into the array form accepted by ``fit()``.

    ``home_xg``/``away_xg`` are NaN for rows without xG columns or values.
    """
    return {
        "home_teams": np.array([r["home_team"] for r in rows], dtype=np.str_),
        "away_teams": np.array([r["away_team"] for r in rows], dtype=np.str_),
        "home_goals": np.array([int(r["home_score"]) for r in rows], dtype=np.int64),
        "away_goals": np.array([int(r["away_score"]) for r in rows], dtype=np.int64),
        "dates": np.array(
            [datetime.strptime(r["date"], DATE_FORMAT) for r in rows],
            dtype="datetime64[D]",
        ),
        "home_xg": np.array(
            [float(r["home_xg"]) if r.get("home_xg") else np.nan for r in rows]
        ),
        "away_xg": np.array(
            [float(r["away_xg"]) if r.get("away_xg") else np.nan for r in rows]
        ),
    }


def _resolve(data_dir: Path | str | None) -> Path:
    """Resolve the data directory, failing with an actionable message."""
    if data_dir is None:
        path = REPO_DATA_DIR / DEFAULT_COMPETITION
    else:
        path = Path(data_dir)
    if not path.is_dir():
        raise FileNotFoundError(
            f"No data directory at {path}. The bundled results ship with the "
            "repository rather than the installed package — clone the repo, or "
            "pass data_dir pointing at your own results CSVs."
        )
    return path


def _list_seasons(root: Path) -> list[str]:
    """List the seasons in an already-resolved data directory, earliest first."""
    return sorted(p.name for p in root.iterdir() if (p / "results.csv").is_file())
