import numpy as np
import pytest

RNG_SEED = 20260825
N_TEAMS = 20
HOME_ADV = 0.3

CSV_HEADER = "date,home_team,away_team,home_score,away_score,gameweek"


def all_pairs_fixtures(n_teams):
    """Every (home_idx, away_idx) pairing of n_teams, excluding self-play."""
    return [(i, j) for i in range(n_teams) for j in range(n_teams) if i != j]


def poisson_match_scores(rng, attack, defence, home_adv, fixtures):
    """Poisson (home, away) scores for each (home_idx, away_idx) fixture."""
    return [
        (
            rng.poisson(np.exp(attack[i] + defence[j] + home_adv)),
            rng.poisson(np.exp(attack[j] + defence[i])),
        )
        for i, j in fixtures
    ]


def write_season_csv(root, season, rows):
    """Write <root>/<season>/results.csv from (date, home, away, hg, ag, gw) rows."""
    season_dir = root / season
    season_dir.mkdir()
    lines = [CSV_HEADER] + [",".join(str(v) for v in row) for row in rows]
    (season_dir / "results.csv").write_text("\n".join(lines))


@pytest.fixture(scope="session")
def synthetic():
    """Poisson-generated fixtures from known team strengths.

    Every team plays every other home and away, so the recovered parameters
    should track the ones used to generate the scores.
    """
    rng = np.random.default_rng(RNG_SEED)
    teams = np.array([f"Team {i:02d}" for i in range(N_TEAMS)])

    # centred so the sum-to-n_teams constraint is satisfiable
    attack = rng.normal(0.0, 0.35, N_TEAMS) + 1.0
    attack = attack - attack.mean() + 1.0
    defence = rng.normal(0.0, 0.35, N_TEAMS) - 1.0

    fixtures = all_pairs_fixtures(N_TEAMS)
    scores = poisson_match_scores(rng, attack, defence, HOME_ADV, fixtures)

    n = len(fixtures)
    dates = np.array(
        [np.datetime64("2024-08-01") + np.timedelta64(d, "D") for d in range(n)]
    )

    return {
        "fit_kwargs": {
            "home_teams": np.array([teams[i] for i, _ in fixtures]),
            "away_teams": np.array([teams[j] for _, j in fixtures]),
            "home_goals": np.array([hg for hg, _ in scores], dtype=np.int64),
            "away_goals": np.array([ag for _, ag in scores], dtype=np.int64),
            "dates": dates,
        },
        "teams": teams,
        "attack": attack,
        "defence": defence,
        "home_adv": HOME_ADV,
    }


@pytest.fixture(scope="session")
def fitted(synthetic):
    from dixon_coles import DixonColesModel

    model = DixonColesModel(seed=RNG_SEED)
    model.fit(**synthetic["fit_kwargs"])
    return model
