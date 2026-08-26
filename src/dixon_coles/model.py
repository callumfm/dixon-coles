"""Dixon-Coles team strength model."""

import logging
import warnings
from typing import Any

import numpy as np
from scipy.optimize import minimize

from dixon_coles._types import (
    DateArray,
    DifficultyThresholds,
    EvalMetric,
    FloatArray,
    IntArray,
    LeagueStrengths,
    MatchPrediction,
    StrArray,
    TeamStrength,
    TeamSummary,
)
from dixon_coles.kernels import (
    expected_goals,
    goal_expectation,
    joint_probability_matrix,
    l2_regularisation,
    low_scoreline_correction,
    outcome_probabilities,
    rho_correction_with_grad,
    time_decay,
)
from dixon_coles.metrics import (
    mean_absolute_error,
    mean_ranked_probability_score,
)

logger = logging.getLogger(__name__)

_NOT_FITTED_MSG = "Model not fitted, call fit() first"


class ConvergenceError(RuntimeError):
    """Raised when the optimiser fails to find a solution."""


class NotFittedError(RuntimeError):
    """Raised when a fitted quantity is requested before calling ``fit()``."""


class DixonColesModel:
    """Dixon Coles team strength model.

    Attributes:
        float_precision: Decimal places used to round the values in returned
            result objects. The raw fitted parameters (``attacks``,
            ``defences``, ``avg_attack``, ``avg_defence``) stay unrounded, and
            every internal calculation runs on those.

    """

    float_precision: int = 4

    def __init__(
        self,
        verbose: bool = False,
        lambda0: float = 3.0,
        seed: int | None = None,
        max_iter: int = 500,
        max_goals: int = 7,
        xi: float = 0.005,
        xg_weight: float = 0.5,
        new_team_percentile: int = 10,
        difficulty_quantiles: tuple[float, float, float, float] = (
            0.15,
            0.35,
            0.65,
            0.85,
        ),
    ):
        """Initialise the model.

        Args:
            verbose: Log progress while fitting.
            lambda0: Strength of the L2 pull toward the league mean. The penalty
                on each team is scaled by ``1 / (matches + 1)``, so teams with
                little history are shrunk hardest. The default is the
                best performer from ``tune_hyperparams`` on the bundled data.
            seed: Seed for the random parameter initialisation. Pass an integer
                for reproducible fits.
            max_iter: Maximum optimiser iterations. Larger fixture sets need
                more; a decade of one league converges in roughly 150.
            max_goals: Scoreline grid size for the probability matrix. Goals
                beyond this are truncated and the matrix renormalised.
            xi: Exponential time-decay rate on past fixtures; higher values
                discount old results faster. The default is the best
                performer from ``tune_hyperparams`` on the bundled data.
            xg_weight: Weight on actual goals versus xG in the fitting
                target (``w*goals + (1-w)*xg`` per match side). Matches
                without xG always use plain goals, so the setting is inert
                when no xG is passed to ``fit``. 1.0 fits on goals alone.
                The default is the best performer from ``tune_hyperparams``
                on the bundled data.
            new_team_percentile: How weak the prior for never-seen teams is,
                as a percentile of the fitted strengths: attacks take the
                p-th percentile and defences the (100 - p)-th, so both sides
                of the prior are weak.
            difficulty_quantiles: Quantile edges that bucket strengths into
                the 1-5 difficulty ratings.

        """
        self.verbose = verbose
        self.lambda0 = lambda0
        self.max_iter = max_iter
        self.max_goals = max_goals
        self.xi = xi
        self.xg_weight = xg_weight
        self.new_team_percentile = new_team_percentile
        self.difficulty_quantiles = difficulty_quantiles
        self._rng = np.random.default_rng(seed)
        self.teams: StrArray | None = None
        self._params: FloatArray | None = None
        self._loglikelihood: float | None = None
        self._difficulty_thresholds: DifficultyThresholds | None = None
        self._new_team_prior: tuple[float, float] | None = None
        self.fitted: bool = False

    def _init_params(self) -> FloatArray:
        """Initialise empty parameters."""
        team_atk = self._rng.uniform(0.5, 1.5, (self.n_teams))
        team_def = self._rng.uniform(-1.5, -0.5, (self.n_teams))
        home_adv = [0.25]
        rho = [-0.1]
        return np.concatenate((team_atk, team_def, home_adv, rho))

    def _require_fitted(self) -> tuple[StrArray, FloatArray]:
        """Get (teams, params), raising unless ``fit()`` has completed."""
        if not self.fitted or self.teams is None or self._params is None:
            raise NotFittedError(_NOT_FITTED_MSG)
        return self.teams, self._params

    @property
    def n_teams(self) -> int:
        """Get the number of teams."""
        return 0 if self.teams is None else len(self.teams)

    @property
    def attacks(self) -> FloatArray:
        """Get the attack parameters."""
        _, params = self._require_fitted()
        return params[: self.n_teams]

    @property
    def defences(self) -> FloatArray:
        """Get the defence parameters."""
        _, params = self._require_fitted()
        return params[self.n_teams : 2 * self.n_teams]

    @property
    def avg_attack(self) -> float:
        """Average attack strength across all teams."""
        return float(np.mean(self.attacks))

    @property
    def avg_defence(self) -> float:
        """Average defence strength across all teams."""
        return float(np.mean(self.defences))

    @property
    def difficulty_thresholds(self) -> DifficultyThresholds:
        """Get the quantile thresholds that bucket strengths into ratings."""
        if self._difficulty_thresholds is None:
            raise NotFittedError(_NOT_FITTED_MSG)
        return self._difficulty_thresholds

    @property
    def home_adv(self) -> float:
        """Get the home advantage parameter."""
        _, params = self._require_fitted()
        return float(params[-2])

    @property
    def rho(self) -> float:
        """Get the rho parameter for low-score correction."""
        _, params = self._require_fitted()
        return float(params[-1])

    @property
    def loglikelihood(self) -> float:
        """Get the maximised penalised log-likelihood of the fit.

        Reported up to a data-only constant (the Poisson factorial term),
        so it is comparable across parameter settings on the same data.
        """
        if self._loglikelihood is None:
            raise NotFittedError(_NOT_FITTED_MSG)
        return self._loglikelihood

    @property
    def n_params(self) -> int:
        """Get the number of fitted parameters."""
        _, params = self._require_fitted()
        return len(params)

    @property
    def aic(self) -> float:
        """Get the Akaike information criterion of the fit."""
        return -2 * self.loglikelihood + 2 * self.n_params

    @property
    def _constraints(self) -> dict[str, Any]:
        """Sum-to-n identifiability constraint on the attack parameters.

        The constraint is linear, so its jacobian is constant; supplying it
        spares the optimiser a finite-difference sweep per iteration.
        """
        jac = np.concatenate((np.ones(self.n_teams), np.zeros(self.n_teams + 2)))
        return {
            "type": "eq",
            "fun": lambda x: float(np.sum(x[: self.n_teams]) - self.n_teams),
            "jac": lambda x: jac,
        }

    @property
    def _bounds(self) -> list[tuple[int, int]]:
        """Define parameter bounds."""
        team_atk = [(0, 3)] * self.n_teams
        team_def = [(-3, 0)] * self.n_teams
        home_adv = [(0, 2)]
        rho = [(-2, 2)]
        return team_atk + team_def + home_adv + rho

    def fit(
        self,
        home_teams: StrArray,
        away_teams: StrArray,
        home_goals: IntArray,
        away_goals: IntArray,
        dates: DateArray,
        home_xg: FloatArray | None = None,
        away_xg: FloatArray | None = None,
    ) -> None:
        """Fits the model to the data, calculating the team strengths, home advantage and intercept.

        When xG arrays are given, strengths are fitted on the blended target
        ``xg_weight*goals + (1-xg_weight)*xg`` per side; NaN xG entries (and
        omitted arrays) fall back to plain goals for those matches.
        """
        if self.verbose:
            logger.info("Fitting model")

        # Invalidate before mutating state, so a fit that raises leaves the
        # model honestly unfitted instead of serving stale or random params.
        self.fitted = False
        self._loglikelihood = None
        self._difficulty_thresholds = None
        self._new_team_prior = None

        current_teams = np.unique(np.concatenate((home_teams, away_teams)))
        if self.teams is None or not np.array_equal(self.teams, current_teams):
            if self.verbose and self.teams is not None:
                logger.info("Teams have changed, re-initializing parameters.")
            self.teams = current_teams
            self._params = self._init_params()

        weights = time_decay(dates, xi=self.xi)
        home_team_indices = np.searchsorted(self.teams, home_teams)
        away_team_indices = np.searchsorted(self.teams, away_teams)
        counts = np.bincount(
            np.concatenate((home_team_indices, away_team_indices)),
            minlength=self.n_teams,
        )
        args = (
            self._blend_target(home_goals, home_xg),
            self._blend_target(away_goals, away_xg),
            home_goals,
            away_goals,
            home_team_indices,
            away_team_indices,
            weights,
            counts,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            res = minimize(
                fun=self._fit_step,
                x0=self._params,
                args=args,
                jac=True,
                constraints=self._constraints,
                bounds=self._bounds,
                options={"maxiter": self.max_iter, "disp": False},
            )

        if not res.success:
            raise ConvergenceError(
                f"Optimisation did not converge after {res.nit} iterations: "
                f"{res.message}. Try raising max_iter (currently {self.max_iter})."
            )

        self._params = np.asarray(res.x, dtype=np.float64)
        self._loglikelihood = -float(res.fun)
        self.fitted = True
        self._difficulty_thresholds = DifficultyThresholds(
            attack=np.quantile(
                self.attacks, self.difficulty_quantiles, method="linear"
            ),
            defence=np.quantile(
                -self.defences, self.difficulty_quantiles, method="linear"
            ),
        )
        # Defences are negative with lower = stronger, so the weak end of the
        # distribution is the (100 - p)th percentile, mirroring the attacks.
        self._new_team_prior = (
            float(np.percentile(self.attacks, self.new_team_percentile)),
            float(np.percentile(self.defences, 100 - self.new_team_percentile)),
        )

        if self.verbose:
            logger.info(f"Model successfully fitted (AIC: {self.aic:.2f})")

    def _blend_target(self, goals: IntArray, xg: FloatArray | None) -> FloatArray:
        """Get the per-match fitting target: goals blended with xG where present."""
        if xg is None:
            return goals.astype(np.float64)
        w = self.xg_weight
        return np.where(np.isnan(xg), goals, w * goals + (1.0 - w) * xg)

    def _fit_step(
        self,
        params: FloatArray,
        target_home: FloatArray,
        target_away: FloatArray,
        home_goals: IntArray,
        away_goals: IntArray,
        home_team_indices: IntArray,
        away_team_indices: IntArray,
        weights: FloatArray,
        counts: IntArray,
    ) -> tuple[float, FloatArray]:
        """Run one objective evaluation: penalised negative log-likelihood and its gradient.

        The Poisson term uses the deviance form ``k*log(mu) - mu`` so the
        (possibly fractional) blended targets are valid; the dropped
        factorial term is constant in the parameters, so the optimum is
        unchanged and ``loglikelihood`` is reported up to a data-only
        constant. The Dixon-Coles tau term always uses the actual scorelines.
        """
        n = self.n_teams
        attack = params[:n]
        defence = params[n : 2 * n]

        home_exp, away_exp = expected_goals(
            home_atk=attack[home_team_indices],
            away_atk=attack[away_team_indices],
            home_def=defence[home_team_indices],
            away_def=defence[away_team_indices],
            home_adv=params[-2],
        )

        home_llk = target_home * np.log(home_exp) - home_exp
        away_llk = target_away * np.log(away_exp) - away_exp

        dc_adj, d_home_exp, d_away_exp, d_rho = rho_correction_with_grad(
            home_goals=home_goals,
            away_goals=away_goals,
            home_exp=home_exp,
            away_exp=away_exp,
            rho=params[-1],
        )

        llk = (home_llk + away_llk + np.log(dc_adj)) * weights

        penalty, grad_attack_pen, grad_defence_pen = l2_regularisation(
            attack=attack,
            defence=defence,
            counts=counts,
            lambda0=self.lambda0,
        )
        objective = float(-np.sum(llk) + penalty)

        g_home = weights * ((target_home - home_exp) + home_exp * d_home_exp)
        g_away = weights * ((target_away - away_exp) + away_exp * d_away_exp)

        grad_attack = grad_attack_pen - (
            np.bincount(home_team_indices, weights=g_home, minlength=n)
            + np.bincount(away_team_indices, weights=g_away, minlength=n)
        )
        grad_defence = grad_defence_pen - (
            np.bincount(away_team_indices, weights=g_home, minlength=n)
            + np.bincount(home_team_indices, weights=g_away, minlength=n)
        )
        gradient = np.concatenate(
            (grad_attack, grad_defence, [-np.sum(g_home)], [-np.sum(weights * d_rho)])
        )

        return objective, gradient

    def predict(self, home_team: str, away_team: str) -> dict[str, float]:
        """Predicts the probabilities of the different possible match outcomes."""
        pred = self._predict_match(home_team, away_team)
        m = pred.prob_matrix
        home_win, draw, away_win = outcome_probabilities(m)

        results = {
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
            "home_clean_sheet": m[:, 0].sum(),
            "away_clean_sheet": m[0, :].sum(),
            "home_goals_for": pred.home_goals_for,
            "away_goals_for": pred.away_goals_for,
            "home_attack": pred.home_attack,
            "away_attack": pred.away_attack,
            "home_defence": pred.home_defence,
            "away_defence": pred.away_defence,
        }

        return {k: round(float(v), self.float_precision) for k, v in results.items()}

    def predict_scoreline_matrix(self, home_team: str, away_team: str) -> FloatArray:
        """Get the joint scoreline probability matrix; rows index home goals."""
        return self._predict_match(home_team, away_team).prob_matrix

    def _predict_match(self, home_team: str, away_team: str) -> MatchPrediction:
        """Predict one fixture: scoreline matrix, expected goals and strengths."""
        home_atk, home_def = self._team_strength(home_team)
        away_atk, away_def = self._team_strength(away_team)
        home_goal_exp, away_goal_exp = expected_goals(
            home_atk=home_atk,
            away_atk=away_atk,
            home_def=home_def,
            away_def=away_def,
            home_adv=self.home_adv,
        )
        return MatchPrediction(
            prob_matrix=self._scoreline_matrix(home_goal_exp, away_goal_exp),
            home_goals_for=float(home_goal_exp),
            away_goals_for=float(away_goal_exp),
            home_attack=home_atk,
            away_attack=away_atk,
            home_defence=home_def,
            away_defence=away_def,
        )

    def _scoreline_matrix(
        self, home_goal_exp: float, away_goal_exp: float
    ) -> FloatArray:
        """Build the corrected joint scoreline probability matrix."""
        m = joint_probability_matrix(
            home_goal_exp=home_goal_exp,
            away_goal_exp=away_goal_exp,
            max_goals=self.max_goals,
        )
        return low_scoreline_correction(
            m=m,
            home_goal_exp=home_goal_exp,
            away_goal_exp=away_goal_exp,
            rho=self.rho,
        )

    def evaluate(
        self,
        home_teams: StrArray,
        away_teams: StrArray,
        home_goals: IntArray,
        away_goals: IntArray,
        dates: DateArray | None = None,
        home_xg: FloatArray | None = None,
        away_xg: FloatArray | None = None,
        metric: EvalMetric = "rps",
    ) -> float:
        """Evaluate the fitted model against a test set with the given metric.

        ``dates``, ``home_xg`` and ``away_xg`` are unused but accepted so a
        ``Results`` mapping can be splatted straight in, mirroring
        ``fit(**results)``.
        """
        if len(home_teams) == 0:
            raise ValueError("evaluate() needs at least one fixture to score")

        home_atk, home_def = self._team_strengths(home_teams)
        away_atk, away_def = self._team_strengths(away_teams)
        home_exp, away_exp = expected_goals(
            home_atk=home_atk,
            away_atk=away_atk,
            home_def=home_def,
            away_def=away_def,
            home_adv=self.home_adv,
        )

        if metric == "mae":
            score = mean_absolute_error(
                home_goals=home_goals,
                away_goals=away_goals,
                home_preds=home_exp,
                away_preds=away_exp,
            )
        elif metric == "rps":
            score = mean_ranked_probability_score(
                home_goals=home_goals,
                away_goals=away_goals,
                prob_matrices=[
                    self._scoreline_matrix(h, a)
                    for h, a in zip(home_exp, away_exp, strict=True)
                ],
            )
        else:
            raise ValueError(f"Unknown metric: {metric!r}. Expected 'mae' or 'rps'.")

        if self.verbose:
            logger.info(f"Evaluation score ({metric}): {score:.3f}")

        return score

    def get_all_team_strengths(self) -> LeagueStrengths:
        """Get league-average strengths and every team's fitted pair."""
        teams, _ = self._require_fitted()

        team_strengths = {
            team: TeamStrength(
                round(float(self.attacks[i]), self.float_precision),
                round(float(self.defences[i]), self.float_precision),
            )
            for i, team in enumerate(teams)
        }

        return LeagueStrengths(
            avg_attack=round(self.avg_attack, self.float_precision),
            avg_defence=round(self.avg_defence, self.float_precision),
            team_strengths=team_strengths,
        )

    def get_team_strength(self, team: str) -> TeamStrength:
        """Get the attack and defence strength for a single team."""
        atk, dfc = self._team_strength(team)
        return TeamStrength(
            round(atk, self.float_precision), round(dfc, self.float_precision)
        )

    def _team_strength(self, team: str) -> TeamStrength:
        """Get one team's unrounded attack and defence strength.

        Derived quantities read strengths through here rather than through
        ``get_team_strength``, so display rounding never reaches the maths.
        """
        atk, dfc = self._team_strengths(np.array([team]))
        return TeamStrength(float(atk[0]), float(dfc[0]))

    def _team_strengths(self, teams: StrArray) -> tuple[FloatArray, FloatArray]:
        """Look up attack and defence strengths for many teams at once.

        Unseen teams get the ``new_team_percentile`` prior. searchsorted
        returns an insertion point, not a match, so an unseen team sorting
        before or between known ones would otherwise silently take a
        neighbour's strengths.
        """
        fitted_teams, _ = self._require_fitted()
        assert self._new_team_prior is not None  # set by fit() alongside fitted

        idx = np.minimum(np.searchsorted(fitted_teams, teams), len(fitted_teams) - 1)
        known = fitted_teams[idx] == teams
        new_atk, new_def = self._new_team_prior
        return (
            np.where(known, self.attacks[idx], new_atk),
            np.where(known, self.defences[idx], new_def),
        )

    def xg_vs_average(self, team: str) -> tuple[float, float]:
        """Get a team's expected goals (scored, conceded) against an average opponent."""
        return self._xg_vs_average(*self._team_strength(team))

    def _xg_vs_average(self, team_atk: float, team_def: float) -> tuple[float, float]:
        """Get expected goals for and against versus an average opponent.

        Half the home advantage goes to each side, averaging the team over
        home and away fixtures rather than assuming a venue.
        """
        half_home_adv = self.home_adv / 2
        xg = float(goal_expectation(team_atk, self.avg_defence, half_home_adv))
        xgc = float(goal_expectation(self.avg_attack, team_def, half_home_adv))
        return round(xg, self.float_precision), round(xgc, self.float_precision)

    def difficulty_rating(self, team: str) -> tuple[int, int]:
        """Bucket a team's strengths into 1-5 (attack, defence) difficulty ratings."""
        return self._difficulty_rating(*self._team_strength(team))

    def _difficulty_rating(self, team_atk: float, team_def: float) -> tuple[int, int]:
        """Bucket already-resolved strengths into 1-5 ratings.

        Defences are negated before bucketing because lower is stronger, so
        both sides run in the same direction as their thresholds.
        """
        thresholds = self.difficulty_thresholds
        attack_bucket = np.searchsorted(thresholds.attack, team_atk, side="left")
        defence_bucket = np.searchsorted(thresholds.defence, -team_def, side="left")
        attack = min(5, int(attack_bucket) + 1)
        defence = min(5, int(defence_bucket) + 1)
        return attack, defence

    #: Alias for :meth:`difficulty_rating`, using Fantasy Premier League's name
    #: for the same 1-5 scale.
    fdr = difficulty_rating

    def get_team_summary(self, team: str) -> TeamSummary:
        """Get all team metrics in a single call."""
        team_atk, team_def = self._team_strength(team)
        xg, xgc = self._xg_vs_average(team_atk, team_def)
        attack_difficulty, defence_difficulty = self._difficulty_rating(
            team_atk, team_def
        )

        return TeamSummary(
            attack=round(team_atk, self.float_precision),
            defence=round(team_def, self.float_precision),
            xg_vs_avg_opp=xg,
            xgc_vs_avg_opp=xgc,
            attack_difficulty=attack_difficulty,
            defence_difficulty=defence_difficulty,
        )
