import numpy as np
import pytest
from scipy.special import gammaln
from scipy.stats import poisson

from dixon_coles.kernels import (
    expected_goals,
    goal_expectation,
    joint_probability_matrix,
    l2_regularisation,
    log_factorial,
    low_scoreline_correction,
    poisson_logpmf,
    rho_correction_with_grad,
    time_decay,
)


class TestPoissonKernels:
    @pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 10, 50, 170])
    def test_log_factorial_is_exact(self, n):
        assert log_factorial(n) == pytest.approx(gammaln(n + 1), rel=1e-12)

    @pytest.mark.parametrize("n", [171, 500, 5000])
    def test_log_factorial_stirling_fallback_stays_accurate(self, n):
        assert log_factorial(n) == pytest.approx(gammaln(n + 1), rel=1e-9)

    @pytest.mark.parametrize("k", [0, 1, 2, 5, 9])
    @pytest.mark.parametrize("mu", [0.3, 1.0, 2.5])
    def test_logpmf_matches_scipy(self, k, mu):
        """Regression: log_factorial was previously off by log(k+1) for k >= 2."""
        assert poisson_logpmf(k, mu) == pytest.approx(poisson.logpmf(k, mu), rel=1e-12)

    def test_logpmf_is_a_normalised_distribution(
        self,
    ):
        mu = 1.7
        total = sum(float(np.exp(poisson_logpmf(k, mu))) for k in range(60))
        assert total == pytest.approx(1.0, abs=1e-12)

    def test_expected_goals_applies_home_advantage(self):
        home, away = expected_goals(
            home_atk=1.2, away_atk=1.2, home_def=-1.0, home_adv=0.3, away_def=-1.0
        )
        assert home > away
        assert home == pytest.approx(np.exp(1.2 - 1.0 + 0.3))

    def test_expected_goals_is_the_one_sided_kernel_applied_twice(self):
        """Both sides must route through goal_expectation, so the expectation
        model cannot drift between a fixture and a versus-average lookup."""
        kw = {"home_atk": 1.2, "away_atk": 0.9, "home_def": -1.0, "away_def": -0.8}
        home, away = expected_goals(home_adv=0.3, **kw)
        assert home == goal_expectation(kw["home_atk"], kw["away_def"], 0.3)
        assert away == goal_expectation(kw["away_atk"], kw["home_def"], 0.0)

    def test_goal_expectation_splits_home_advantage(self):
        """Half the advantage to each side sits between the two venues."""
        atk, opp_def, home_adv = 1.2, -1.0, 0.3
        at_home = goal_expectation(atk, opp_def, home_adv)
        away = goal_expectation(atk, opp_def, 0.0)
        averaged = goal_expectation(atk, opp_def, home_adv / 2)
        assert away < averaged < at_home


class TestProbabilityMatrix:
    def test_is_a_valid_distribution(self):
        m = joint_probability_matrix(home_goal_exp=1.5, away_goal_exp=1.1)
        assert m.shape == (7, 7)
        assert (m >= 0).all()
        assert m.sum() == pytest.approx(1.0, abs=5e-3)

    def test_wide_grid_captures_essentially_all_mass(self):
        m = joint_probability_matrix(1.5, 1.2, max_goals=15)
        assert m.sum() == pytest.approx(1.0, abs=1e-9)

    def test_marginals_match_scipy_poisson(self):
        m = joint_probability_matrix(1.6, 1.1, max_goals=15)
        np.testing.assert_allclose(
            m.sum(axis=1), poisson.pmf(np.arange(15), 1.6), rtol=1e-9
        )

    def test_max_goals_controls_grid_size(self):
        assert joint_probability_matrix(1.5, 1.1, max_goals=12).shape == (12, 12)

    def test_wider_grid_captures_more_mass(self):
        narrow = joint_probability_matrix(2.5, 2.5, max_goals=5).sum()
        wide = joint_probability_matrix(2.5, 2.5, max_goals=15).sum()
        assert wide > narrow

    def test_correction_renormalises(self):
        m = joint_probability_matrix(1.4, 1.2)
        out = low_scoreline_correction(
            m=m, home_goal_exp=1.4, away_goal_exp=1.2, rho=-0.1
        )
        assert out.sum() == pytest.approx(1.0, abs=1e-9)

    def test_extreme_rho_never_goes_negative(self):
        """Regression: tau can exceed the valid region inside the optimiser
        bounds; corrected cells must be floored at zero, not negative."""
        m = joint_probability_matrix(2.5, 1.5)
        out = low_scoreline_correction(
            m=m, home_goal_exp=2.5, away_goal_exp=1.5, rho=0.3
        )
        assert (out >= 0).all()
        assert out.sum() == pytest.approx(1.0, abs=1e-9)

    def test_negative_rho_lifts_the_draw_cells(self):
        m = joint_probability_matrix(1.4, 1.2)
        out = low_scoreline_correction(
            m=m, home_goal_exp=1.4, away_goal_exp=1.2, rho=-0.1
        )
        assert out[0, 0] / out.sum() > m[0, 0] / m.sum()


class TestRegularisation:
    def test_penalty_and_gradient_are_zero_at_the_mean(self):
        """The priors are the parameter means, so uniform strengths cost nothing."""
        penalty, grad_atk, grad_def = l2_regularisation(
            attack=np.full(4, 1.5),
            defence=np.full(4, -1.0),
            counts=np.array([1, 2, 3, 4]),
            lambda0=3.0,
        )
        assert penalty == pytest.approx(0.0)
        np.testing.assert_allclose(grad_atk, 0.0, atol=1e-12)
        np.testing.assert_allclose(grad_def, 0.0, atol=1e-12)

    def test_teams_with_fewer_matches_are_penalised_harder(self):
        """This is the whole point: promoted sides get pulled to the mean."""
        args = {
            "attack": np.array([2.0, 0.0]),
            "defence": np.zeros(2),
            "lambda0": 3.0,
        }
        few = l2_regularisation(counts=np.array([2, 2]), **args)[0]
        many = l2_regularisation(counts=np.array([38, 38]), **args)[0]
        assert few > many

    def test_lambda0_scales_the_penalty(self):
        args = {
            "attack": np.array([2.0, 0.0]),
            "defence": np.zeros(2),
            "counts": np.array([5, 5]),
        }
        assert l2_regularisation(lambda0=6.0, **args)[0] == pytest.approx(
            2 * l2_regularisation(lambda0=3.0, **args)[0]
        )


class TestTimeDecay:
    def test_most_recent_match_has_full_weight(self):
        dates = np.array(["2024-01-01", "2024-06-01"], dtype="datetime64[D]")
        assert time_decay(dates, xi=0.001)[-1] == pytest.approx(1.0)

    def test_weights_decrease_with_age(self):
        dates = np.array(
            ["2020-01-01", "2022-01-01", "2024-01-01"], dtype="datetime64[D]"
        )
        w = time_decay(dates, xi=0.001)
        assert w[0] < w[1] < w[2]

    def test_unit_independent(self):
        """Regression: xi is per-day; nanosecond-resolution dates (pandas'
        default) must decay identically to day-resolution ones."""
        days = np.array(
            ["2020-01-01", "2022-01-01", "2024-01-01"], dtype="datetime64[D]"
        )
        np.testing.assert_allclose(
            time_decay(days.astype("datetime64[ns]"), xi=0.001),
            time_decay(days, xi=0.001),
        )

    def test_zero_xi_disables_decay(self):
        dates = np.array(["2020-01-01", "2024-01-01"], dtype="datetime64[D]")
        np.testing.assert_allclose(time_decay(dates, xi=0.0), [1.0, 1.0])


class TestRhoCorrectionClamp:
    def test_clamp_zeroes_the_gradient_terms(self):
        corr, d_home, d_away, d_rho = rho_correction_with_grad(
            np.array([0]), np.array([0]), np.array([2.0]), np.array([1.5]), 1.9
        )
        assert corr[0] == pytest.approx(1e-6)
        assert d_home[0] == d_away[0] == d_rho[0] == 0.0

    def test_unclamped_matches_tau(self):
        corr, d_home, _, d_rho = rho_correction_with_grad(
            np.array([0]), np.array([1]), np.array([1.4]), np.array([1.1]), -0.1
        )
        tau = 1 + 1.4 * -0.1
        assert corr[0] == pytest.approx(tau)
        assert d_home[0] == pytest.approx(-0.1 / tau)
        assert d_rho[0] == pytest.approx(1.4 / tau)
