"""Low-level mathematical primitives for Dixon-Coles model."""

import numpy as np
from numba import njit, vectorize

from dixon_coles._types import DateArray, FloatArray, IntArray, ScalarOrArray


@njit  # type: ignore[misc]
def goal_expectation(
    atk: ScalarOrArray, opp_def: ScalarOrArray, home_adv: float
) -> ScalarOrArray:
    """Get one side's goal expectation against a given opposing defence.

    ``home_adv`` is the share of the home advantage credited to this side:
    the full value for a home team, zero for an away team, and half each
    when averaging a team over both venues.
    """
    return np.exp(atk + opp_def + home_adv)  # type: ignore[return-value]


@njit  # type: ignore[misc]
def expected_goals(
    home_atk: ScalarOrArray,
    away_atk: ScalarOrArray,
    home_def: ScalarOrArray,
    away_def: ScalarOrArray,
    home_adv: float,
) -> tuple[ScalarOrArray, ScalarOrArray]:
    """Get the home and away goal expectation."""
    return (
        goal_expectation(home_atk, away_def, home_adv),
        goal_expectation(away_atk, home_def, 0.0),
    )


#: Exact log-factorials, covering every goal count that can occur in practice.
#: 170! is the largest representable in float64, so the Stirling fallback below
#: is unreachable for real fixtures and exists only for total safety.
#: The table is ~10x faster than math.lgamma inside the likelihood loop.
_MAX_TABULATED = 170
_LOG_FACTORIAL = np.concatenate(
    ([0.0], np.cumsum(np.log(np.arange(1, _MAX_TABULATED + 1, dtype=np.float64))))
)


@vectorize(["float64(int64)"])  # type: ignore[misc]
def log_factorial(n: int | IntArray) -> float | FloatArray:
    """Get log(n!), exactly for n <= 170 and by Stirling's series beyond.

    A scalar kernel compiled to a ufunc; broadcasts over integer arrays.
    """
    if n < 0:
        return np.nan
    if n <= _MAX_TABULATED:
        return _LOG_FACTORIAL[n]
    return n * np.log(n) - n + 0.5 * np.log(2 * np.pi * n) + 1.0 / (12.0 * n)


@vectorize(["float64(int64, float64)"])  # type: ignore[misc]
def poisson_logpmf(k: int | IntArray, mu: float | FloatArray) -> float | FloatArray:
    """Get the Poisson log-PMF without a scipy call in the hot loop.

    A scalar kernel compiled to a ufunc; broadcasts over arrays.
    """
    return k * np.log(mu) - log_factorial(k) - mu  # type: ignore[return-value]


@njit  # type: ignore[misc]
def joint_probability_matrix(
    home_goal_exp: float, away_goal_exp: float, max_goals: int = 7
) -> FloatArray:
    """Build the joint score probability matrix."""
    goals = np.arange(max_goals)
    home_probs = np.exp(poisson_logpmf(goals, home_goal_exp))
    away_probs = np.exp(poisson_logpmf(goals, away_goal_exp))
    return np.outer(home_probs, away_probs)


@njit  # type: ignore[misc]
def dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    home_exp: float,
    away_exp: float,
    rho: float,
) -> float:
    """Dixon-Coles adjustment factor for one scoreline (1.0 above 1-1)."""
    if home_goals == 0 and away_goals == 0:
        return 1 - home_exp * away_exp * rho
    if home_goals == 0 and away_goals == 1:
        return 1 + home_exp * rho
    if home_goals == 1 and away_goals == 0:
        return 1 + away_exp * rho
    if home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


@njit  # type: ignore[misc]
def low_scoreline_correction(
    m: FloatArray, home_goal_exp: float, away_goal_exp: float, rho: float
) -> FloatArray:
    """Dixon-Coles low score correction e.g. 0-0, 0-1, 1-0, 1-1.

    Returns a new, renormalised matrix; the input is left untouched. The
    corrected cells are floored at zero so an extreme rho cannot produce
    negative probabilities (mirroring the likelihood path's clamp).
    """
    m = m.copy()
    for hg in range(2):
        for ag in range(2):
            tau = dixon_coles_tau(hg, ag, home_goal_exp, away_goal_exp, rho)
            m[hg, ag] = max(m[hg, ag] * tau, 0.0)

    total = m.sum()
    if total > 0:
        m /= total

    return m


@njit  # type: ignore[misc]
def dixon_coles_tau_grad(
    home_goals: int,
    away_goals: int,
    home_exp: float,
    away_exp: float,
    rho: float,
) -> tuple[float, float, float]:
    """Get the partials of ``dixon_coles_tau`` w.r.t. (home_exp, away_exp, rho)."""
    if home_goals == 0 and away_goals == 0:
        return -away_exp * rho, -home_exp * rho, -home_exp * away_exp
    if home_goals == 0 and away_goals == 1:
        return rho, 0.0, home_exp
    if home_goals == 1 and away_goals == 0:
        return 0.0, rho, away_exp
    if home_goals == 1 and away_goals == 1:
        return 0.0, 0.0, -1.0
    return 0.0, 0.0, 0.0


@njit  # type: ignore[misc]
def rho_correction_with_grad(
    home_goals: IntArray,
    away_goals: IntArray,
    home_exp: FloatArray,
    away_exp: FloatArray,
    rho: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Vectorized rho correction (floored at 1e-6) and the partials of its log.

    Returns ``(corrections, d_home_exp, d_away_exp, d_rho)``, where the last
    three are per-match partial derivatives of ``log(correction)`` with
    respect to the goal expectations and rho, zeroed where the floor binds
    (the clamped objective is flat there).
    """
    n = home_goals.size
    corrections = np.empty(n, dtype=np.float64)
    d_home_exp = np.zeros(n, dtype=np.float64)
    d_away_exp = np.zeros(n, dtype=np.float64)
    d_rho = np.zeros(n, dtype=np.float64)

    for i in range(n):
        tau = dixon_coles_tau(
            home_goals[i], away_goals[i], home_exp[i], away_exp[i], rho
        )
        if tau > 1e-6:
            corrections[i] = tau
            dt_home, dt_away, dt_rho = dixon_coles_tau_grad(
                home_goals[i], away_goals[i], home_exp[i], away_exp[i], rho
            )
            d_home_exp[i] = dt_home / tau
            d_away_exp[i] = dt_away / tau
            d_rho[i] = dt_rho / tau
        else:
            corrections[i] = 1e-6

    return corrections, d_home_exp, d_away_exp, d_rho


@njit  # type: ignore[misc]
def l2_regularisation(
    attack: FloatArray,
    defence: FloatArray,
    counts: IntArray,
    lambda0: float,
) -> tuple[float, FloatArray, FloatArray]:
    """L2 penalty pulling teams toward the league mean, and its gradient.

    The penalty per team is scaled by ``1 / (matches + 1)``, so teams with
    little history are shrunk hardest. Returns
    ``(penalty, grad_attack, grad_defence)``. The priors are the means of the
    current parameters, so every gradient component carries a
    ``-(2/n) * sum(kappa * deviation)`` correction from the prior moving with
    the parameters.
    """
    kappa = lambda0 * (1.0 / (counts + 1.0))
    atk_dev = attack - np.mean(attack)
    def_dev = defence - np.mean(defence)
    penalty = float(np.sum(kappa * (atk_dev**2 + def_dev**2)))

    n = attack.size
    grad_attack = 2.0 * kappa * atk_dev - 2.0 * np.sum(kappa * atk_dev) / n
    grad_defence = 2.0 * kappa * def_dev - 2.0 * np.sum(kappa * def_dev) / n

    return penalty, grad_attack, grad_defence


def time_decay(dates: DateArray, xi: float) -> FloatArray:
    """Exponentially decay fixtures so that old ones influence the current strength less.

    ``xi`` is a per-day rate; ages are converted to days explicitly so any
    datetime64 unit (e.g. pandas' default nanoseconds) behaves identically.
    """
    age_days = (dates.max() - dates).astype("timedelta64[D]").astype(np.float64)
    return np.exp(-xi * age_days)


def outcome_probabilities(
    m: FloatArray,
) -> tuple[float | FloatArray, float | FloatArray, float | FloatArray]:
    """Get (home win, draw, away win) probabilities from scoreline matrices.

    Rows index home goals, so the lower triangle is the home win. Accepts a
    single ``(g, g)`` matrix or a stacked ``(n, g, g)`` array, returning
    scalars or length-``n`` arrays accordingly.
    """
    g = m.shape[-1]
    lower = np.tril(np.ones((g, g), dtype=bool), -1)
    home = m[..., lower].sum(axis=-1)
    draw = np.trace(m, axis1=-2, axis2=-1)
    away = m[..., lower.T].sum(axis=-1)
    return home, draw, away
