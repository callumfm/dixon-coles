import numpy as np
import pytest

from dixon_coles.metrics import (
    mean_absolute_error,
    mean_ranked_probability_score,
)


def rps(home_goals, away_goals, prob_matrix):
    """Ranked probability score for a single match."""
    return mean_ranked_probability_score(
        home_goals=np.array([home_goals]),
        away_goals=np.array([away_goals]),
        prob_matrices=[prob_matrix],
    )


class TestRankedProbabilityScore:
    def test_rps_is_zero_for_a_certain_correct_call(self):
        m = np.zeros((3, 3))
        m[2, 0] = 1.0  # home win with certainty
        assert rps(home_goals=2, away_goals=0, prob_matrix=m) == pytest.approx(0.0)

    def test_rps_is_worst_for_a_certain_wrong_call(self):
        m = np.zeros((3, 3))
        m[2, 0] = 1.0
        assert rps(home_goals=0, away_goals=2, prob_matrix=m) == pytest.approx(1.0)

    def test_rps_penalises_the_further_wrong_outcome_more(self):
        m = np.zeros((3, 3))
        m[2, 0] = 1.0
        draw = rps(home_goals=1, away_goals=1, prob_matrix=m)
        loss = rps(home_goals=0, away_goals=2, prob_matrix=m)
        assert loss > draw

    def test_mean_rps_averages(self):
        m = np.zeros((3, 3))
        m[2, 0] = 1.0
        score = mean_ranked_probability_score(
            home_goals=np.array([2, 0]),
            away_goals=np.array([0, 2]),
            prob_matrices=[m, m],
        )
        assert score == pytest.approx(0.5)


class TestMeanAbsoluteError:
    def test_mae_is_zero_for_exact_predictions(self):
        assert mean_absolute_error(
            home_goals=np.array([2, 1]),
            away_goals=np.array([0, 1]),
            home_preds=np.array([2.0, 1.0]),
            away_preds=np.array([0.0, 1.0]),
        ) == pytest.approx(0.0)

    def test_mae_averages_both_sides(self):
        assert mean_absolute_error(
            home_goals=np.array([2]),
            away_goals=np.array([0]),
            home_preds=np.array([3.0]),
            away_preds=np.array([1.0]),
        ) == pytest.approx(1.0)
