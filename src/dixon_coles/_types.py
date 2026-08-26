"""Shared type aliases and structured result types."""

from __future__ import annotations

from typing import Literal, NamedTuple, TypeAlias, TypedDict, TypeVar

import numpy as np
import numpy.typing as npt

DateArray: TypeAlias = npt.NDArray[np.datetime64]
StrArray: TypeAlias = npt.NDArray[np.str_]
IntArray: TypeAlias = npt.NDArray[np.int_]
FloatArray: TypeAlias = npt.NDArray[np.float64]

EvalMetric: TypeAlias = Literal["mae", "rps"]

#: Scalar-in gives scalar-out, array-in gives array-out — the numba kernels
#: broadcast over both.
ScalarOrArray = TypeVar("ScalarOrArray", float, FloatArray)


class TeamStrength(NamedTuple):
    """Attack and defence strength for one team."""

    attack: float
    defence: float


class TeamSummary(NamedTuple):
    """Headline metrics for one team."""

    attack: float
    defence: float
    xg_vs_avg_opp: float
    xgc_vs_avg_opp: float
    attack_difficulty: int
    defence_difficulty: int


class LeagueStrengths(NamedTuple):
    """League-average strengths plus every fitted team's pair."""

    avg_attack: float
    avg_defence: float
    team_strengths: dict[str, TeamStrength]


class Results(TypedDict):
    """Match results in the keyword form accepted by ``DixonColesModel.fit``."""

    home_teams: StrArray
    away_teams: StrArray
    home_goals: IntArray
    away_goals: IntArray
    dates: DateArray
    home_xg: FloatArray
    away_xg: FloatArray


class MatchPrediction(NamedTuple):
    """Scoreline matrix, expected goals and strengths for one fixture."""

    prob_matrix: FloatArray
    home_goals_for: float
    away_goals_for: float
    home_attack: float
    away_attack: float
    home_defence: float
    away_defence: float


class DifficultyThresholds(NamedTuple):
    """Quantile thresholds that bucket strengths into 1-5 ratings."""

    attack: FloatArray
    defence: FloatArray
