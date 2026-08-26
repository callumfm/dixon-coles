# dixon-coles

[Dixon-Coles](https://www.jstor.org/stable/2986290) is a statistical model for predicting football match scores. It uses a refined Poisson model to rate every team's attack and defence from past results, then turns any two of those ratings into a distribution over scorelines i.e. the chance of 0-0, 1-0, and so on. The score probabilities can then be summed to find the probability of a home win, draw and away win.

This repo is a fast Python implementation of the Dixon-Coles model which could be used for things such as predicting clean sheet probabilities for Fantasy Premier League. This model has a couple of features which differ from the standard implementation:

- **Time decay** on past matches from raw dates, so recent form counts for more.
- **Shrinkage** toward the league mean, weighted by `1/(matches + 1)` — a team three games
  in is pulled hard toward average, a team at 38 games is barely touched.
- **Goals blended with xG** (50/50 by default) as the fitting target, since chances created
  are a less noisy signal than goals scored. Worth ~1-2% of ranked probability score.
- **Unseen teams** fall back to a low-percentile prior instead of a `KeyError`, so a newly
  promoted side is predictable from its first fixture.

## Install

```bash
pip install git+https://github.com/callumfm/dixon-coles
```

The bundled Premier League data is not part of the installed package — see [Data](#data)
to clone it or point at your own.

## Usage

Full API documentation can be found [here](https://callumfm.github.io/dixon-coles/).

### Forecast
```python
>>> import dixon_coles as dc

>>> model = dc.DixonColesModel(seed=42)
>>> model.fit(**dc.load_results(["2223", "2324", "2425"]))

>>> model.predict("Arsenal", "Liverpool")
{'home_win': 0.421, 'draw': 0.2496, 'away_win': 0.3294, ...}

>>> model.get_team_summary("Arsenal")
TeamSummary(attack=1.2433, defence=-1.2013, xg_vs_avg_opp=1.8172,
            xgc_vs_avg_opp=0.8634, attack_difficulty=5, defence_difficulty=5)
```

### Backtest
Fit on all prior seasons, score 2024/25. Lower is better for the ranked
probability score (default) and mean absolute goal error.

```python
>>> dc.backtest("2425", seed=42)
0.20516303202473937
>>> dc.backtest("2425", metric="mae", seed=42)
0.9475493966258298
```

### Tune
Grid-search lambda0, xi and xg_weight on early-season holdouts.
`tune_hyperparams` trains through gameweek 3 and scores gameweeks 4-7 of every season 
after the first, so it tunes for the case the shrinkage exists for rather than full-season 
hindsight.

```python
>>> dc.tune_hyperparams(seed=42)
(3.0, 0.005, 0.5)
```

## Data

9 seasons of Premier League results (2016/17–2024/25, 3,420 matches) ship with the repo but
not with the installed package — clone to get them:

```
# date, teams, scores, gameweek, home_xg, away_xg
data/premier-league/<season>/results.csv   
```

Point `data_dir` anywhere with the same shape to use your own:

```python
load_results(data_dir="~/my-data/la-liga")
```

Results are derived from [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
(MIT), which in turn derives from the FPL API and Understat. xG from 2022/23 GW16 onward is
Opta via the FPL player data; earlier fixtures are backfilled from [Understat](https://understat.com)
and rescaled onto the same scale by the sources' overlap ratio (0.93).
