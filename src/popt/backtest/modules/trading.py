import numpy as np
import pandas as pd
from typing import Literal

from popt.backtest.modules.backtestdata import DataBuilder

# NOTE
# - create an external trade policy (tp) object and give it to the strategy
#   togther with data builder (db). The strategy can then provide the tp
#   with the db to decide if there should be a trade on day t or not.
class TradingPolicy:
    def __init__(self, db: DataBuilder):
        self.db = db

    def trade_flag(self, t: int) -> bool:
        raise NotImplementedError


class FrequencyPolicy(TradingPolicy):
    def __init__(
            self,
            db: DataBuilder,
            freq: Literal["D", "W", "M", "Q", "Y", None] = "M",
        ):
        super().__init__(db)
        self.freq = freq

        timeline = db.timeline
        weekday, months, quarters, years = timeline.weekday, timeline.month, timeline.quarter, timeline.year
        match freq:
            case "D":  flag = np.ones(timeline.shape[0], dtype=bool)
            case "W":  flag = (weekday == 4)  # 4 extra rebal on red fridays over 20 years, negligible
            case "M":  flag = months != np.roll(months, shift=-1)
            case "Q":  flag = quarters != np.roll(quarters, shift=-1)
            case "Y":  flag = years != np.roll(years, shift=-1)
            case None: flag = np.zeros(timeline.shape[0], dtype=bool)
            case _: raise ValueError("Invalid rebal frequency")
        flag[1] = True  # first day is initial state, second day we trade according to strategy

        self.flag = flag

    def trade_flag(self, t):
        return self.flag[t]



class DrawdownPolicy(TradingPolicy):
    def __init__(self, db):
        super().__init__(db)

    def trade_flag(self, t):
        return super().trade_flag(t)


