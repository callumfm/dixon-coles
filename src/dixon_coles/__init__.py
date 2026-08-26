"""Dixon-Coles team strength model."""

import logging

from dixon_coles._types import (
    LeagueStrengths,
    Results,
    TeamStrength,
    TeamSummary,
)
from dixon_coles.backtesting import backtest, tune_hyperparams
from dixon_coles.datasets import available_seasons, load_results
from dixon_coles.model import ConvergenceError, DixonColesModel, NotFittedError

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "ConvergenceError",
    "DixonColesModel",
    "LeagueStrengths",
    "NotFittedError",
    "Results",
    "TeamStrength",
    "TeamSummary",
    "available_seasons",
    "backtest",
    "load_results",
    "tune_hyperparams",
]
