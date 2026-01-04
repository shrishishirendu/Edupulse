"""Enrollment forecasting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

import pandas as pd


@dataclass
class EnrollmentForecaster:
    """Produce placeholder enrollment projections."""

    horizon_months: int = 6

    def forecast(self, historical: pd.Series) -> pd.DataFrame:
        """Return a simple linear projection as a placeholder."""
        start = datetime.utcnow()
        projections: List[int] = [int(historical.mean())] * self.horizon_months
        dates = pd.date_range(start=start, periods=self.horizon_months, freq="MS")
        return pd.DataFrame({"date": dates, "enrollment": projections})