from datetime import date, timedelta

import numpy as np
import pytest
from conftest import all_pairs_fixtures, poisson_match_scores, write_season_csv

from dixon_coles import backtest, tune_hyperparams
from dixon_coles.backtesting import get_train_test_splits
from dixon_coles.datasets import _load_rows

SEASONS = ["2223", "2324", "2425"]
N_TEAMS = 6
N_GAMEWEEKS = 10
HOME_ADV = 0.3


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    """Three tiny synthetic seasons of a 6-team league with gameweeks."""
    rng = np.random.default_rng(7)
    attack = rng.normal(1.0, 0.3, N_TEAMS)
    defence = rng.normal(-1.0, 0.3, N_TEAMS)

    root = tmp_path_factory.mktemp("data")
    for season in SEASONS:
        fixtures = all_pairs_fixtures(N_TEAMS)
        rng.shuffle(fixtures)
        scores = poisson_match_scores(rng, attack, defence, HOME_ADV, fixtures)
        per_gw = len(fixtures) // N_GAMEWEEKS
        start = date(2000 + int(season[:2]), 8, 1)

        rows = [
            (
                start + timedelta(days=m),
                f"Team {i}",
                f"Team {j}",
                hg,
                ag,
                min(m // per_gw + 1, N_GAMEWEEKS),
            )
            for m, ((i, j), (hg, ag)) in enumerate(zip(fixtures, scores, strict=True))
        ]
        write_season_csv(root, season, rows)

    return root


class TestBacktest:
    def test_rps_score_is_in_range(self, data_dir):
        score = backtest("2425", data_dir=data_dir, seed=0)
        assert 0.0 <= score <= 1.0

    def test_mae_metric(self, data_dir):
        score = backtest("2425", metric="mae", data_dir=data_dir, seed=0)
        assert np.isfinite(score)
        assert score >= 0.0

    def test_reproducible_with_seed(self, data_dir):
        first = backtest("2425", data_dir=data_dir, seed=1)
        second = backtest("2425", data_dir=data_dir, seed=1)
        assert first == second

    def test_first_season_has_nothing_to_train_on(self, data_dir):
        with pytest.raises(ValueError, match="before"):
            backtest("2223", data_dir=data_dir)

    def test_unknown_season_lists_what_is_available(self, data_dir):
        with pytest.raises(ValueError, match="Available:"):
            backtest("9999", data_dir=data_dir)


class TestGetTrainTestSplits:
    def test_one_split_per_season_after_the_first(self, data_dir):
        rows = _load_rows(data_dir=data_dir)
        splits = get_train_test_splits(rows, train_end_gw=3, eval_end_gw=7)
        assert [s for s, _, _ in splits] == SEASONS[1:]

    def test_train_grows_and_eval_stays_in_window(self, data_dir):
        rows = _load_rows(data_dir=data_dir)
        splits = get_train_test_splits(rows, train_end_gw=3, eval_end_gw=7)
        per_gw = (N_TEAMS * (N_TEAMS - 1)) // N_GAMEWEEKS
        prev_train = 0
        for _, train, eval_set in splits:
            n_train = len(train["home_teams"])
            assert n_train > prev_train
            prev_train = n_train
            assert len(eval_set["home_teams"]) == 4 * per_gw  # gameweeks 4-7


class TestTuneHyperparams:
    def test_reproducible_with_seed(self, data_dir):
        kwargs = {
            "lambda0_candidates": (0.0, 5.0),
            "xi_candidates": (0.001,),
            "xg_weight_candidates": (1.0,),
            "data_dir": data_dir,
            "seed": 2,
        }
        assert tune_hyperparams(**kwargs) == tune_hyperparams(**kwargs)

    def test_joint_grid_returns_members_of_each_grid(self, data_dir):
        lambda0_candidates = (0.0, 5.0)
        xi_candidates = (0.0, 0.01)
        xg_weight_candidates = (1.0,)
        best_lambda0, best_xi, best_w = tune_hyperparams(
            lambda0_candidates=lambda0_candidates,
            xi_candidates=xi_candidates,
            xg_weight_candidates=xg_weight_candidates,
            data_dir=data_dir,
            seed=0,
        )
        assert best_lambda0 in lambda0_candidates
        assert best_xi in xi_candidates
        assert best_w in xg_weight_candidates

    def test_missing_gameweek_column_raises_upfront(self, tmp_path):
        for season in ("2324", "2425"):
            season_dir = tmp_path / season
            season_dir.mkdir()
            (season_dir / "results.csv").write_text(
                "date,home_team,away_team,home_score,away_score\n"
                "2024-08-01,Team 0,Team 1,1,0\n"
                "2024-08-02,Team 1,Team 0,2,2"
            )
        with pytest.raises(ValueError, match="gameweek"):
            tune_hyperparams(data_dir=tmp_path)

    def test_empty_eval_window_raises(self, data_dir):
        with pytest.raises(ValueError, match="no fixtures in gameweeks"):
            tune_hyperparams(data_dir=data_dir, train_end_gw=3, eval_end_gw=3)

    def test_needs_at_least_two_seasons(self, tmp_path):
        write_season_csv(
            tmp_path, "2425", [("2024-08-01", "Team 0", "Team 1", 1, 0, 1)]
        )
        with pytest.raises(ValueError, match="two seasons"):
            tune_hyperparams(data_dir=tmp_path)
