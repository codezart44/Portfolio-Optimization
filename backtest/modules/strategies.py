import numpy as np
import cvxpy as cp
from .backtestdata import DataLoader
from abc import ABC, abstractmethod
from typing import Sequence

class BacktestStrategy(ABC):
    def __init__(self, dl: DataLoader):
        super().__init__()
        assert isinstance(dl, DataLoader), type(dl)
        self.dl = dl
        self.name = self.__class__.__name__.lower()

    @staticmethod
    def normalize_weights(w: np.ndarray, lev: float) -> np.ndarray:
        w = w.copy()
        w_sum   = w.sum()
        cap = 1.0 + lev
        if w_sum > cap + 1e-8:
            w *= cap / (w_sum + 1e-8)
        return w
    
    @abstractmethod
    def get_weights(self, t:int, w_prev:np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_trade_flag(self, t: int) -> bool:
        pass



class Markowitz(BacktestStrategy):
    def __init__(
            self,
            dl: DataLoader,
            lookahead: int,
            gamma: float,
            lev: float,
            w_max: float,
            vc_lim: float,
        ):
        super().__init__(dl)
        assert lookahead <= 0, lookahead
        assert lev >= 0.0, lev
        # assert w_max ...

        self.lookahead = lookahead
        self.gamma = gamma
        self.lev = lev
        self.w_max = w_max
        self.vc_lim = vc_lim / np.sqrt(252)

    def get_trade_flag(self, t: int) -> bool:
        return self.dl.get_trade_flag(t)

    def get_weights(self, t:int, w_prev:np.ndarray) -> np.ndarray:
        assert w_prev.shape == (self.dl.N,), w_prev.shape
        lookahead = self.lookahead
        w: np.ndarray = markowitz(
            a      = self.dl.get_alpha(t-1+lookahead),
            S_sqrt = self.dl.get_sigma_sqrt(t-1+lookahead), 
            w_prev = w_prev, 
            w_max  = self.w_max, 
            gamma  = self.gamma, 
            lev    = self.lev, 
            vc_lim = self.vc_lim
            )
        w = self.normalize_weights(w, self.lev)
        return w

def markowitz(
        a: np.ndarray, 
        S_sqrt: np.ndarray, 
        w_prev: np.ndarray, 
        w_max: float, 
        gamma: float, 
        lev: float, 
        vc_lim: float
    ) -> np.ndarray:
    w = cp.Variable(len(a))
    prob = cp.Problem(
        objective=cp.Maximize(a @ w - gamma * cp.norm(w - w_prev, p=1)),
        constraints=[
            cp.norm(S_sqrt @ w, p=2) <= vc_lim,  # volatility control
            0.0 <= w, w <= w_max,  # no shorting, capped positions
            cp.sum(w) <= 1 + lev,    # leverage
        ],
    )
    prob.solve(solver=cp.CLARABEL, verbose=False)
    if w.value is None:
        raise RuntimeError(f"Solver failed with status: {prob.status}")
    return w.value


class FixedWeights(BacktestStrategy):
    def __init__(
            self,
            dl: DataLoader,
            w_rebal: np.ndarray,
            lev: float,
            vc_lim: float,
        ):
        super().__init__(dl)
        assert w_rebal.sum().round(4) == 1.0, w_rebal.sum()
        assert (w_rebal >= 0).all(), w_rebal
        assert w_rebal.shape == (dl.N,), w_rebal.shape

        self.w_rebal = w_rebal
        self.lev = lev
        self.vc_lim = vc_lim / np.sqrt(252)

    def get_trade_flag(self, t: int) -> bool:
        return self.dl.get_trade_flag(t)

    def get_weights(self, t:int, w_prev:np.ndarray) -> np.ndarray:
        assert w_prev.shape       == (self.dl.N,), w_prev.shape
        assert self.w_rebal.shape == (self.dl.N,), self.w_rebal.shape
        asset_mask = self.dl.get_asset_mask(t-1)
        sigma_sqrt = self.dl.get_sigma_sqrt(t-1)
        w = self.w_rebal.copy()
        
        if not asset_mask.all():
            w = self.w_rebal.copy()
            w[~asset_mask] = 0.0
            w_sum = w.sum()
            assert w_sum > 0.0
            w = w / w_sum

        k = self.vc_lim / (np.linalg.norm(sigma_sqrt @ w) + 1e-8)
        k = np.minimum(k, 1.0 + self.lev)
        w = k * w  # scaling for vol control
        w = self.normalize_weights(w, self.lev)
        return w



class MinimumVol(BacktestStrategy):
    def __init__(
            self,
            dl: DataLoader,
        ):
        super().__init__(dl)

    def get_trade_flag(self, t: int) -> bool:
        return self.dl.get_trade_flag(t)
    
    def get_weights(self, dl: DataLoader, t:int, w_prev: np.ndarray) -> np.ndarray:
        ...
        return w_prev

