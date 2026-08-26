"""Evaluation metrics for scoring model predictions."""

import numpy as np

from dixon_coles._types import FloatArray, IntArray
from dixon_coles.kernels import outcome_probabilities


def mean_ranked_probability_score(
    home_goals: IntArray, away_goals: IntArray, prob_matrices: list[FloatArray]
) -> float:
    """Mean ranked probability score across a set of matches.

    Per match, the squared distance between the cumulative predicted and
    cumulative observed probabilities over the ordered outcomes (home win,
    draw, away win). Lower is better; 0 is a perfect forecast.
    """
    prob_home, prob_draw, _ = outcome_probabilities(np.stack(prob_matrices))
    outcome_home = (home_goals > away_goals).astype(np.float64)
    outcome_draw = (home_goals == away_goals).astype(np.float64)
    predicted_cum = np.stack([prob_home, prob_home + prob_draw], axis=-1)
    observed_cum = np.stack([outcome_home, outcome_home + outcome_draw], axis=-1)
    return float(np.mean(np.square(predicted_cum - observed_cum)))


def mean_absolute_error(
    home_goals: IntArray,
    away_goals: IntArray,
    home_preds: FloatArray,
    away_preds: FloatArray,
) -> float:
    """Mean absolute error of predicted goals, averaged over home and away."""
    home_errors = np.abs(home_goals - home_preds)
    away_errors = np.abs(away_goals - away_preds)
    mae = float(np.mean((home_errors + away_errors) / 2))
    return mae
