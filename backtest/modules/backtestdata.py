import numpy as np
import pandas as pd
from typing import Literal

class DataBuilder:
    def __init__(
            self,
            first_date: str,
            final_date: str,
            alpha_d: pd.DataFrame, 
            return_d: pd.DataFrame,
            rf_d: pd.DataFrame,
            riskmodel: dict[str, np.ndarray],
            rebal_freq: Literal["D", "W", "M", "Q", "Y", None] = "M",
        ):
        dates_risk = riskmodel["dates"]
        sigma_sqrt = riskmodel["sigma_sqrt"]
        _, k, U = sigma_sqrt.shape
        assert dates_risk.shape[0] == sigma_sqrt.shape[0]
        assert alpha_d.shape[1] == U
        assert return_d.shape[1] == U

        self.d0 = first_date
        self.d1 = final_date
        timeline = self._dates_intersection([dates_risk, return_d.index, rf_d.index])
        timeline = self._dates_truncated(timeline)

        self.timeline   = timeline
        self.alpha      = alpha_d.loc[timeline].values
        self.ret        = return_d.loc[timeline].values + 1.0  # use simple returns
        self.rf         = rf_d.loc[timeline].values.ravel() + 1.0  # use simple returns
        self.sigma_sqrt = sigma_sqrt[np.isin(dates_risk, timeline)]
        self.asset_mask = ~np.isnan(self.sigma_sqrt).all(axis=1)
        self.trade_flag = self._trade_flag(timeline, rebal_freq)
        T, = self.trade_flag.shape

        self.ret = np.nan_to_num(self.ret, nan=0.0)
        self.sigma_sqrt = np.nan_to_num(self.sigma_sqrt, nan=0.0)

        assert np.any(np.isnan(self.timeline))   == False
        assert np.any(np.isnan(self.alpha))      == False
        assert np.any(np.isnan(self.ret))        == False
        assert np.any(np.isnan(self.rf))         == False
        assert np.any(np.isnan(self.sigma_sqrt)) == False
        assert np.any(np.isnan(self.asset_mask)) == False
        assert np.any(np.isnan(self.trade_flag)) == False

        assert self.timeline.shape   == (T, )
        assert self.alpha.shape      == (T, U)
        assert self.ret.shape        == (T, U)
        assert self.rf.shape         == (T,)
        assert self.sigma_sqrt.shape == (T, k, U)
        assert self.asset_mask.shape == (T, U)
        assert self.trade_flag.shape == (T,)

    def __repr__(self):
        return f"{self.d0} : {self.d1}\n" + \
               f" :a  - {self.alpha.shape}, {type(self.alpha)}\n" + \
               f" :r  - {self.ret.shape}, {type(self.ret)}\n" + \
               f" :rf - {self.rf.shape}, {type(self.rf)}\n" + \
               f" :Ss - {self.sigma_sqrt.shape}, {type(self.sigma_sqrt)}\n" + \
               f" :am - {self.asset_mask.shape}, {type(self.asset_mask)}\n" + \
               f" :tf - {self.trade_flag.shape}, {type(self.trade_flag)}\n"
        
    def _dates_intersection(self, indices: list) -> pd.DatetimeIndex:
        index_overlap = pd.DatetimeIndex(indices[0])
        for index in indices[1:]:
            index_overlap = index_overlap.intersection(pd.DatetimeIndex(index))
        return index_overlap
    
    def _dates_truncated(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        assert self.d0 in index  # only allow exact date selection, forces explicitness
        assert self.d1 in index
        index: pd.DatetimeIndex = index[(self.d0 <= index) & (index <= self.d1)]
        return index
    
    def _trade_flag(self, timeline: pd.DatetimeIndex, rebal_freq: str) -> np.ndarray:
        months, quarters, years = timeline.month, timeline.quarter, timeline.year
        match rebal_freq:
            case "D":  rebal_flag = np.ones(timeline.shape[0], dtype=bool)
            case "W":  rebal_flag = (timeline.weekday == 4)  # 4 extra rebal on red fridays over 20 years, negligible
            case "M":  rebal_flag = months != np.roll(months, shift=-1)
            case "Q":  rebal_flag = quarters != np.roll(quarters, shift=-1)
            case "Y":  rebal_flag = years != np.roll(years, shift=-1)
            case None: rebal_flag = np.zeros(timeline.shape[0], dtype=bool)
            case _: raise ValueError("Invalid rebal frequency")
        rebal_flag[1] = True  # first day is initial state, second day we trade according to strategy
        return rebal_flag


# Convention: Data at day t is t inclusive always.
class DataLoader:
    def __init__(
            self,
            tickers: list[str],
            universe: list[str],
            db: DataBuilder,
        ):
        T, _, U = db.sigma_sqrt.shape
        N = len(tickers)
        assert np.isin(tickers, universe).all()
        assert len(universe) == U, len(universe)

        t2i = {t: i for i, t in enumerate(universe)}
        i_N  = np.array([t2i[t] for t in tickers], dtype=int)
        self.T = T
        self.U = U
        self.N = N
        self.tickers  = tickers
        self.universe = universe
        self.timeline = db.timeline
        self._alpha      = db.alpha[:, i_N]
        self._ret        = db.ret[:, i_N]
        self._rf         = db.rf
        self._sigma_sqrt = db.sigma_sqrt[:, :, i_N]
        self._asset_mask = db.asset_mask[:, i_N]
        self._trade_flag = db.trade_flag
    
    def get_alpha(self, t:int) -> np.ndarray:
        return self._alpha[t]
    
    def get_return(self, t:int) -> np.ndarray:
        return self._ret[t]
    
    def get_rf(self, t:int) -> np.ndarray:
        return self._rf[t]
    
    def get_sigma_sqrt(self, t:int) -> np.ndarray:
        return self._sigma_sqrt[t]

    def get_asset_mask(self, t: int) -> np.ndarray:
        return self._asset_mask[t]
    
    def get_trade_flag(self, t: int) -> int:
        return self._trade_flag[t]
