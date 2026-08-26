import numpy as np
import pytest
from conftest import write_season_csv

from dixon_coles import DixonColesModel, available_seasons, load_results

EXPECTED_SEASONS = 9
MATCHES_PER_SEASON = 380


class TestAvailableSeasons:
    def test_lists_bundled_seasons_sorted(self):
        seasons = available_seasons()
        assert len(seasons) == EXPECTED_SEASONS
        assert seasons == sorted(seasons)
        assert seasons[0] == "1617"

    def test_missing_directory_explains_where_data_lives(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="clone the repo"):
            available_seasons(tmp_path / "nope")


class TestLoadResults:
    def test_loads_every_season_by_default(self):
        data = load_results()
        assert len(data["home_teams"]) == EXPECTED_SEASONS * MATCHES_PER_SEASON

    def test_single_season_as_string(self):
        assert len(load_results("2425")["home_teams"]) == MATCHES_PER_SEASON

    def test_subset_as_list(self):
        data = load_results(["2324", "2425"])
        assert len(data["home_teams"]) == 2 * MATCHES_PER_SEASON

    def test_arrays_are_aligned_and_typed(self):
        data = load_results("2425")
        arrays = [
            data["home_teams"],
            data["away_teams"],
            data["home_goals"],
            data["away_goals"],
            data["dates"],
        ]
        assert len({len(a) for a in arrays}) == 1
        assert data["home_goals"].dtype == np.int64
        assert data["dates"].dtype == np.dtype("datetime64[D]")
        assert (data["home_goals"] >= 0).all()

    def test_rows_are_sorted_by_date(self):
        dates = load_results()["dates"]
        assert (np.diff(dates).astype(int) >= 0).all()

    def test_dates_are_uniformly_iso(self):
        """Upstream ships three different date encodings; ours are normalised."""
        dates = load_results()["dates"]
        assert dates.min() == np.datetime64("2016-08-13")
        assert dates.max() == np.datetime64("2025-05-25")

    def test_unknown_season_lists_what_is_available(self):
        with pytest.raises(ValueError, match="Available:"):
            load_results("9999")

    def test_custom_data_dir(self, tmp_path):
        src = load_results("2425")
        rows = [
            (f"2024-08-{(i % 28) + 1:02d}", h, a, hg, ag, 1)
            for i, (h, a, hg, ag) in enumerate(
                zip(
                    src["home_teams"],
                    src["away_teams"],
                    src["home_goals"],
                    src["away_goals"],
                    strict=True,
                )
            )
        ]
        write_season_csv(tmp_path, "2425", rows)
        assert (
            load_results(data_dir=tmp_path)["home_teams"].shape
            == src["home_teams"].shape
        )


class TestEndToEnd:
    def test_fits_the_full_bundled_dataset(self):
        """The default max_iter must handle every season at once."""
        model = DixonColesModel(seed=0)
        model.fit(**load_results())
        assert model.fitted
        assert model.n_teams > 30
        p = model.predict("Arsenal", "Liverpool")
        assert p["home_win"] + p["draw"] + p["away_win"] == pytest.approx(1.0, abs=1e-3)
