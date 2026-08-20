import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import order matters on Windows: torch's OpenMP runtime has to be loaded
# before numpy's. See iagnn/_bootstrap.py for the full explanation.
import iagnn  # noqa: F401,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def feature_cols():
    return ["f1", "f2", "f3"]


@pytest.fixture
def synthetic_panel(feature_cols):
    """A small, fully synthetic panel with a known signal in `f1`.

    400 stocks across 8 industries, 12 dates. `next_return` is built as
    `0.5 * f1 + noise`, so any working model must find a positive IC on f1.
    """
    rng = np.random.default_rng(0)
    n_stocks, n_dates = 400, 12
    codes = [f"{i:06d}" for i in range(n_stocks)]
    industries = [f"IND{i % 8}" for i in range(n_stocks)]
    dates = pd.bdate_range("2020-01-01", periods=n_dates)

    rows = []
    for date in dates:
        features = rng.normal(size=(n_stocks, len(feature_cols)))
        rows.append(pd.DataFrame({
            "TradingDay": date,
            "SecuCode": codes,
            "industry": industries,
            "log_size": rng.normal(size=n_stocks),
            **{c: features[:, i] for i, c in enumerate(feature_cols)},
            "next_return": 0.5 * features[:, 0] + rng.normal(scale=0.5, size=n_stocks),
        }))
    return pd.concat(rows, ignore_index=True)
