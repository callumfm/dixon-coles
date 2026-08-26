import numpy as np
import pytest
from conftest import RNG_SEED
from scipy.optimize import approx_fprime, minimize

from dixon_coles import ConvergenceError, DixonColesModel, NotFittedError
from dixon_coles.kernels import time_decay


class TestFit:
    def test_recovers_known_strengths(self, fitted, synthetic):
        """Fitted attack/defence should track the generating parameters."""
        order = np.searchsorted(fitted.teams, synthetic["teams"])
        assert np.corrcoef(fitted.attacks[order], synthetic["attack"])[0, 1] > 0.9
        assert np.corrcoef(fitted.defences[order], synthetic["defence"])[0, 1] > 0.9

    def test_recovers_home_advantage(self, fitted, synthetic):
        assert fitted.home_adv == pytest.approx(synthetic["home_adv"], abs=0.1)

    def test_identifiability_constraint_holds(self, fitted):
        """Attack params must sum to the team count, else strengths float freely."""
        assert fitted.attacks.sum() == pytest.approx(fitted.n_teams, abs=1e-4)

    def test_sets_fit_diagnostics(self, fitted):
        assert fitted.fitted is True
        assert fitted.n_params == 2 * fitted.n_teams + 2
        assert fitted.aic == pytest.approx(
            -2 * fitted.loglikelihood + 2 * fitted.n_params
        )

    def test_all_team_strengths_covers_every_team(self, fitted):
        league = fitted.get_all_team_strengths()
        assert set(league.team_strengths) == set(fitted.teams)
        team = fitted.teams[0]
        assert league.team_strengths[team] == fitted.get_team_strength(team)

    def test_low_max_iter_raises_convergence_error(self, synthetic):
        model = DixonColesModel(seed=RNG_SEED, max_iter=2)
        with pytest.raises(ConvergenceError, match="max_iter"):
            model.fit(**synthetic["fit_kwargs"])

    def test_failed_refit_leaves_model_unfitted(self, synthetic):
        """Regression: a fit that raises must not leave the model claiming
        the previous fit's state while holding fresh random parameters."""
        kw = synthetic["fit_kwargs"]
        model = DixonColesModel(seed=RNG_SEED)
        model.fit(**{k: v[:200] for k, v in kw.items()})
        assert model.fitted

        model.max_iter = 2
        with pytest.raises(ConvergenceError):
            model.fit(**kw)
        assert model.fitted is False
        with pytest.raises(NotFittedError):
            _ = model.attacks


class TestReproducibility:
    def test_same_seed_gives_identical_params(self, synthetic):
        a = DixonColesModel(seed=RNG_SEED)
        b = DixonColesModel(seed=RNG_SEED)
        a.fit(**synthetic["fit_kwargs"])
        b.fit(**synthetic["fit_kwargs"])
        np.testing.assert_allclose(a.attacks, b.attacks)
        np.testing.assert_allclose(a.defences, b.defences)
        assert a.home_adv == b.home_adv
        assert a.rho == b.rho

    def test_different_seeds_start_from_different_params(self):
        a, b = DixonColesModel(seed=1), DixonColesModel(seed=2)
        a.teams = np.array(["x", "y"])
        b.teams = np.array(["x", "y"])
        assert not np.array_equal(a._init_params(), b._init_params())


class TestPredict:
    def test_outcome_probabilities_sum_to_one(self, fitted):
        for home, away in [("Team 00", "Team 01"), ("Team 05", "Team 12")]:
            p = fitted.predict(home, away)
            total = p["home_win"] + p["draw"] + p["away_win"]
            assert total == pytest.approx(1.0, abs=1e-3)

    def test_probability_matrix_is_normalised(self, fitted):
        m = fitted.predict_scoreline_matrix("Team 00", "Team 01")
        assert m.sum() == pytest.approx(1.0, abs=1e-9)
        assert (m >= 0).all()

    def test_stronger_team_at_home_is_favoured(self, fitted, synthetic):
        best = synthetic["teams"][np.argmax(synthetic["attack"])]
        worst = synthetic["teams"][np.argmin(synthetic["attack"])]
        p = fitted.predict(best, worst)
        assert p["home_win"] > p["away_win"]

    def test_max_goals_widens_the_grid(self, synthetic):
        model = DixonColesModel(seed=RNG_SEED, max_goals=12)
        model.fit(**synthetic["fit_kwargs"])
        assert model.predict_scoreline_matrix("Team 00", "Team 01").shape == (12, 12)

    def test_predict_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            DixonColesModel().predict("a", "b")

    def test_strength_properties_before_fit_raise(self):
        model = DixonColesModel()
        for prop in ("attacks", "defences", "home_adv", "rho"):
            with pytest.raises(NotFittedError):
                getattr(model, prop)


class TestUnseenTeams:
    def test_prior_is_weak_on_both_sides(self, fitted):
        """Regression: defences are negative with lower = stronger, so the
        defence prior must come from the high-percentile (weak) end — the
        old code handed unseen teams a near-elite defence."""
        atk, dfc = fitted.get_team_strength("Newly Promoted FC")
        assert atk < float(np.median(fitted.attacks))
        assert dfc > float(np.median(fitted.defences))  # concedes more than median

    def test_unknown_team_is_predictable(self, fitted):
        p = fitted.predict("Team 00", "Newly Promoted FC")
        assert p["home_win"] + p["draw"] + p["away_win"] == pytest.approx(1.0, abs=1e-3)


class TestDifficultyRating:
    def test_ratings_are_in_range(self, fitted):
        for team in fitted.teams:
            a, d = fitted.difficulty_rating(team)
            assert 1 <= a <= 5
            assert 1 <= d <= 5

    def test_stronger_attack_rates_higher(self, fitted):
        strongest = fitted.teams[np.argmax(fitted.attacks)]
        weakest = fitted.teams[np.argmin(fitted.attacks)]
        assert (
            fitted.difficulty_rating(strongest)[0]
            >= fitted.difficulty_rating(weakest)[0]
        )

    def test_unseen_team_gets_a_rating(self, fitted):
        a, d = fitted.difficulty_rating("Newly Promoted FC")
        assert 1 <= a <= 5
        assert 1 <= d <= 5

    def test_fdr_alias_matches(self, fitted):
        team = fitted.teams[0]
        assert fitted.fdr(team) == fitted.difficulty_rating(team)

    def test_team_summary_fields(self, fitted):
        summary = fitted.get_team_summary("Team 00")
        assert summary._fields == (
            "attack",
            "defence",
            "xg_vs_avg_opp",
            "xgc_vs_avg_opp",
            "attack_difficulty",
            "defence_difficulty",
        )
        assert summary.attack == pytest.approx(fitted.get_team_strength("Team 00")[0])

    def test_result_objects_round_every_float_to_float_precision(self, fitted):
        """Regression: strengths used to come back unrounded while the xG
        fields beside them were rounded, so one record printed ragged."""
        precision = fitted.float_precision
        team = fitted.teams[0]
        league = fitted.get_all_team_strengths()
        floats = [
            *fitted.get_team_summary(team)[:4],
            *fitted.get_team_strength(team),
            *fitted.xg_vs_average(team),
            league.avg_attack,
            league.avg_defence,
            *league.team_strengths[team],
            *fitted.predict(team, fitted.teams[1]).values(),
        ]
        for value in floats:
            assert value == round(value, precision)

    def test_rounding_does_not_leak_into_derived_quantities(self, fitted):
        """Rounded strengths must not be fed back into exp(), or xG shifts."""
        i = 0
        team = fitted.teams[i]
        raw_atk = float(fitted.attacks[i])
        raw_def = float(fitted.defences[i])
        expected_xg = np.exp(raw_atk + fitted.avg_defence + fitted.home_adv / 2)
        expected_xgc = np.exp(raw_def + fitted.avg_attack + fitted.home_adv / 2)

        xg, xgc = fitted.xg_vs_average(team)
        assert xg == round(float(expected_xg), fitted.float_precision)
        assert xgc == round(float(expected_xgc), fitted.float_precision)

    def test_predict_is_unchanged_by_display_rounding(self, fitted):
        """Regression: predict() read strengths through the rounding getter,
        so the 4dp display precision shifted the scoreline matrix itself."""
        coarse = DixonColesModel(seed=RNG_SEED)
        coarse.__dict__.update(fitted.__dict__)
        coarse.float_precision = 1

        baseline = fitted.predict_scoreline_matrix("Team 00", "Team 01")
        np.testing.assert_allclose(
            coarse.predict_scoreline_matrix("Team 00", "Team 01"), baseline
        )


class TestEvaluate:
    @pytest.mark.parametrize("metric", ["mae", "rps"])
    def test_metrics_are_finite_and_positive(self, fitted, synthetic, metric):
        kw = synthetic["fit_kwargs"]
        score = fitted.evaluate(
            home_teams=kw["home_teams"][:50],
            away_teams=kw["away_teams"][:50],
            home_goals=kw["home_goals"][:50],
            away_goals=kw["away_goals"][:50],
            metric=metric,
        )
        assert np.isfinite(score)
        assert score > 0

    def test_empty_evaluation_set_raises(self, fitted, synthetic):
        kw = synthetic["fit_kwargs"]
        with pytest.raises(ValueError, match="at least one fixture"):
            fitted.evaluate(
                home_teams=kw["home_teams"][:0],
                away_teams=kw["away_teams"][:0],
                home_goals=kw["home_goals"][:0],
                away_goals=kw["away_goals"][:0],
            )

    def test_unknown_metric_raises(self, fitted, synthetic):
        kw = synthetic["fit_kwargs"]
        with pytest.raises(ValueError, match="Unknown metric"):
            fitted.evaluate(
                home_teams=kw["home_teams"][:5],
                away_teams=kw["away_teams"][:5],
                home_goals=kw["home_goals"][:5],
                away_goals=kw["away_goals"][:5],
                metric="accuracy",  # type: ignore[arg-type]
            )


class TestUnseenTeamLookupRegression:
    """searchsorted returns an insertion point; unseen names that sort before or
    between known teams used to silently inherit a neighbour's strengths."""

    @pytest.mark.parametrize(
        "name", ["AAA United", "Team 05a", "Zzz Rovers", "Newly Promoted FC"]
    )
    def test_unseen_names_never_alias_a_real_team(self, fitted, name):
        assert name not in set(fitted.teams)
        precision = fitted.float_precision
        expected = (
            round(
                float(np.percentile(fitted.attacks, fitted.new_team_percentile)),
                precision,
            ),
            round(
                float(np.percentile(fitted.defences, 100 - fitted.new_team_percentile)),
                precision,
            ),
        )
        assert fitted.get_team_strength(name) == expected

    def test_known_teams_still_resolve_exactly(self, fitted):
        precision = fitted.float_precision
        for i, team in enumerate(fitted.teams):
            atk, dfc = fitted.get_team_strength(team)
            assert atk == round(float(fitted.attacks[i]), precision)
            assert dfc == round(float(fitted.defences[i]), precision)


class TestTunableParams:
    def test_xg_weight_changes_the_fit_when_xg_is_given(self, synthetic):
        kw = dict(synthetic["fit_kwargs"])
        rng = np.random.default_rng(3)
        kw["home_xg"] = kw["home_goals"] + rng.uniform(-0.5, 0.5, len(kw["home_goals"]))
        kw["away_xg"] = kw["away_goals"] + rng.uniform(-0.5, 0.5, len(kw["away_goals"]))
        goals_only = DixonColesModel(seed=0, xg_weight=1.0)
        blended = DixonColesModel(seed=0, xg_weight=0.5)
        goals_only.fit(**kw)
        blended.fit(**kw)
        assert not np.array_equal(goals_only.attacks, blended.attacks)

    def test_xg_weight_is_inert_without_xg(self, synthetic):
        goals_only = DixonColesModel(seed=0, xg_weight=1.0)
        blended = DixonColesModel(seed=0, xg_weight=0.5)
        goals_only.fit(**synthetic["fit_kwargs"])
        blended.fit(**synthetic["fit_kwargs"])
        np.testing.assert_allclose(goals_only.attacks, blended.attacks)

    def test_nan_xg_falls_back_to_goals(self, synthetic):
        kw = dict(synthetic["fit_kwargs"])
        kw["home_xg"] = np.full(len(kw["home_goals"]), np.nan)
        kw["away_xg"] = np.full(len(kw["away_goals"]), np.nan)
        with_nans = DixonColesModel(seed=0, xg_weight=0.5)
        without = DixonColesModel(seed=0)
        with_nans.fit(**kw)
        without.fit(**synthetic["fit_kwargs"])
        np.testing.assert_allclose(with_nans.attacks, without.attacks)

    def test_xi_changes_the_fit(self, synthetic):
        default = DixonColesModel(seed=0)
        decayed = DixonColesModel(seed=0, xi=0.05)
        default.fit(**synthetic["fit_kwargs"])
        decayed.fit(**synthetic["fit_kwargs"])
        assert not np.array_equal(default.attacks, decayed.attacks)

    def test_new_team_percentile_shifts_unseen_team_prior(self, synthetic):
        low = DixonColesModel(seed=0, new_team_percentile=10)
        median = DixonColesModel(seed=0, new_team_percentile=50)
        low.fit(**synthetic["fit_kwargs"])
        median.fit(**synthetic["fit_kwargs"])
        assert (
            median.get_team_strength("Unseen FC")[0]
            > low.get_team_strength("Unseen FC")[0]
        )

    def test_difficulty_quantiles_change_thresholds(self, synthetic):
        default = DixonColesModel(seed=0)
        coarse = DixonColesModel(seed=0, difficulty_quantiles=(0.2, 0.4, 0.6, 0.8))
        default.fit(**synthetic["fit_kwargs"])
        coarse.fit(**synthetic["fit_kwargs"])
        assert not np.array_equal(
            default.difficulty_thresholds.attack,
            coarse.difficulty_thresholds.attack,
        )


def fit_step_args(model, synthetic):
    """Precompute the fixed _fit_step args exactly as fit() does."""
    kw = synthetic["fit_kwargs"]
    teams = np.unique(np.concatenate((kw["home_teams"], kw["away_teams"])))
    model.teams = teams
    hidx = np.searchsorted(teams, kw["home_teams"])
    aidx = np.searchsorted(teams, kw["away_teams"])
    weights = time_decay(kw["dates"], xi=model.xi)
    counts = np.bincount(np.concatenate((hidx, aidx)), minlength=len(teams))
    return (
        kw["home_goals"].astype(np.float64),
        kw["away_goals"].astype(np.float64),
        kw["home_goals"],
        kw["away_goals"],
        hidx,
        aidx,
        weights,
        counts,
    )


class TestFitStepGradient:
    """The analytic gradient in _fit_step must track the objective exactly.

    A wrong gradient does not crash — it silently converges to a wrong
    optimum — so these tests pin it against scipy's finite differences and
    against a gradient-free fit of the same objective.
    """

    @pytest.mark.parametrize("trial", range(3))
    def test_matches_finite_differences(self, synthetic, trial):
        model = DixonColesModel()
        args = fit_step_args(model, synthetic)
        rng = np.random.default_rng(trial)
        x = np.concatenate(
            (
                rng.uniform(0.5, 1.5, model.n_teams),
                rng.uniform(-1.5, -0.5, model.n_teams),
                [rng.uniform(0.0, 0.5)],
                [rng.uniform(-0.3, 0.1)],
            )
        )

        numeric = np.asarray(
            approx_fprime(x, lambda p: model._fit_step(p, *args)[0], 1e-7),
            dtype=np.float64,
        )
        analytic = model._fit_step(x, *args)[1]
        np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-3)

    def test_matches_finite_differences_with_fractional_targets(self, synthetic):
        """Blended goals/xG targets are fractional; the deviance-form
        gradient must stay exact for them."""
        model = DixonColesModel()
        base = fit_step_args(model, synthetic)
        rng = np.random.default_rng(5)
        target_home = 0.5 * base[0] + 0.5 * rng.gamma(2.0, 0.7, base[0].shape)
        target_away = 0.5 * base[1] + 0.5 * rng.gamma(2.0, 0.6, base[1].shape)
        args = (target_home, target_away, *base[2:])
        x = np.concatenate(
            (
                rng.uniform(0.5, 1.5, model.n_teams),
                rng.uniform(-1.5, -0.5, model.n_teams),
                [0.25],
                [-0.1],
            )
        )

        numeric = np.asarray(
            approx_fprime(x, lambda p: model._fit_step(p, *args)[0], 1e-7),
            dtype=np.float64,
        )
        analytic = model._fit_step(x, *args)[1]
        np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-3)

    def test_matches_finite_differences_where_clamp_binds(self, synthetic):
        """A large positive rho drives tau below the 1e-6 floor on 0-0
        scorelines; objective and gradient must stay consistent there."""
        model = DixonColesModel()
        args = fit_step_args(model, synthetic)
        n = model.n_teams
        x = np.concatenate((np.full(n, 1.2), np.full(n, -0.8), [0.3], [1.9]))

        numeric = np.asarray(
            approx_fprime(x, lambda p: model._fit_step(p, *args)[0], 1e-7),
            dtype=np.float64,
        )
        analytic = model._fit_step(x, *args)[1]
        np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-3)

    def test_fit_matches_gradient_free_optimum(self, fitted, synthetic):
        """The jac fit must land on the same optimum as minimising the same
        objective with finite differences from the same starting point."""
        model = DixonColesModel(seed=RNG_SEED)  # same init as the fitted fixture
        args = fit_step_args(model, synthetic)
        res = minimize(
            fun=lambda p: model._fit_step(p, *args)[0],
            x0=model._init_params(),
            constraints=model._constraints,
            bounds=model._bounds,
            options={"maxiter": model.max_iter, "disp": False},
        )
        assert res.success
        np.testing.assert_allclose(fitted._params, res.x, atol=1e-3)
        assert -float(res.fun) == pytest.approx(fitted.loglikelihood, abs=1e-4)
