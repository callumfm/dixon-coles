"""Season-based evaluation and hyperparameter tuning over results data.

Both functions work from the bundled CSVs (or any ``data_dir`` in the same
layout) rather than requiring you to assemble train/eval splits by hand.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any

import numpy as np

from dixon_coles._types import EvalMetric, Results
from dixon_coles.datasets import (
    _load_rows,
    _rows_to_results,
    available_seasons,
    load_results,
)
from dixon_coles.model import DixonColesModel

logger = logging.getLogger(__name__)

LAMBDA0_CANDIDATES = (0.0, 0.1, 1.0, 3.0, 5.0, 10.0, 25.0)
XI_CANDIDATES = (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01)
XG_WEIGHT_CANDIDATES = (1.0, 0.75, 0.5, 0.25, 0.0)


def backtest(
    eval_season: str,
    metric: EvalMetric = "rps",
    data_dir: Path | str | None = None,
    **model_kwargs: Any,
) -> float:
    """Fit on every season before ``eval_season`` and score on that season.

    Extra keyword arguments go to ``DixonColesModel`` (e.g. ``lambda0``,
    ``seed``); returns the score, lower is better for both metrics.
    """
    seasons = available_seasons(data_dir)
    eval_set = load_results(eval_season, data_dir=data_dir)  # validates the season

    train_seasons = seasons[: seasons.index(eval_season)]
    if not train_seasons:
        raise ValueError(
            f"No seasons before {eval_season!r} to train on. "
            f"Available: {', '.join(seasons)}"
        )

    logger.info(f"Evaluating {eval_season} after training on {train_seasons}")

    model = DixonColesModel(**model_kwargs)
    model.fit(**load_results(train_seasons, data_dir=data_dir))
    return float(model.evaluate(**eval_set, metric=metric))


def tune_hyperparams(
    lambda0_candidates: tuple[float, ...] = LAMBDA0_CANDIDATES,
    xi_candidates: tuple[float, ...] = XI_CANDIDATES,
    xg_weight_candidates: tuple[float, ...] = XG_WEIGHT_CANDIDATES,
    train_end_gw: int = 3,
    eval_end_gw: int = 7,
    metric: EvalMetric = "rps",
    seed: int | None = None,
    data_dir: Path | str | None = None,
) -> tuple[float, float, float]:
    """Grid-search shrinkage, time decay and the goals/xG blend jointly.

    Scores each candidate on the early-season holdouts built by
    ``get_train_test_splits`` — the situation the shrinkage exists for. The
    axes are tuned jointly because they interact. Returns the
    ``(lambda0, xi, xg_weight)`` triple with the best mean score.
    """
    logger.info(f"Tuning hyperparams ({metric=}, {train_end_gw=}, {eval_end_gw=})")

    rows = _load_rows(data_dir=data_dir)
    splits = get_train_test_splits(rows, train_end_gw, eval_end_gw)

    best = (lambda0_candidates[0], xi_candidates[0], xg_weight_candidates[0])
    best_score = float("inf")

    for lambda0, xi, xg_weight in itertools.product(
        lambda0_candidates, xi_candidates, xg_weight_candidates
    ):
        mean_score = _mean_holdout_score(
            splits, metric, lambda0=lambda0, xi=xi, xg_weight=xg_weight, seed=seed
        )
        logger.info(
            f"lambda0={lambda0}, xi={xi}, xg_weight={xg_weight}, "
            f"mean score={mean_score:.4f}"
        )
        if mean_score < best_score:
            best_score = mean_score
            best = (lambda0, xi, xg_weight)

    logger.info(
        f"Best lambda0: {best[0]}, xi: {best[1]}, xg_weight: {best[2]} "
        f"(mean score {best_score:.4f})"
    )
    return best


def get_train_test_splits(
    rows: list[dict[str, str]],
    train_end_gw: int,
    eval_end_gw: int,
) -> list[tuple[str, Results, Results]]:
    """Build one (season, train, eval) split per season after the first.

    Training is every prior season plus the split season's gameweeks up to
    ``train_end_gw``; evaluation is its gameweeks ``(train_end_gw,
    eval_end_gw]``. ``rows`` come from ``_load_rows``; inputs are validated
    (two-season minimum, gameweek column, non-empty windows).
    """
    seasons = sorted({r["season"] for r in rows})
    if len(seasons) < 2:
        raise ValueError(f"Tuning needs at least two seasons; found {len(seasons)}.")

    missing_gw = sum(1 for r in rows if not r.get("gameweek"))
    if missing_gw:
        raise ValueError(
            "Tuning splits seasons by gameweek, so results.csv needs a "
            f"'gameweek' column; {missing_gw} of {len(rows)} rows lack one. "
            "backtest() and fit() have no such requirement."
        )

    splits = []
    for idx, season in enumerate(seasons[1:], start=1):
        prior = set(seasons[:idx])
        train_rows = [
            r
            for r in rows
            if r["season"] in prior
            or (r["season"] == season and int(r["gameweek"]) <= train_end_gw)
        ]
        eval_rows = [
            r
            for r in rows
            if r["season"] == season
            and train_end_gw < int(r["gameweek"]) <= eval_end_gw
        ]
        if not eval_rows:
            raise ValueError(
                f"Season {season} has no fixtures in gameweeks "
                f"({train_end_gw}, {eval_end_gw}] — nothing to score."
            )
        splits.append(
            (season, _rows_to_results(train_rows), _rows_to_results(eval_rows))
        )

    return splits


def _mean_holdout_score(
    splits: list[tuple[str, Results, Results]],
    metric: EvalMetric,
    **model_kwargs: Any,
) -> float:
    """Fit one model configuration on every split and average its scores."""
    scores = []
    for season, train, eval_set in splits:
        model = DixonColesModel(**model_kwargs)
        model.fit(**train)
        score = model.evaluate(**eval_set, metric=metric)
        scores.append(score)
        logger.debug(f"{model_kwargs}, season={season}, score={score:.4f}")
    return float(np.mean(scores))
