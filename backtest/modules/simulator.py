import numpy as np
import pandas as pd
from .backtestdata import DataLoader
from .strategies import BacktestStrategy


class BacktestSimulator:
    def __init__(
            self,
            spread: float = 5e-4,  # 5 basis points
        ):
        self.spread: float = spread
        self.pw: np.ndarray | None = None  # portfolio weights
        self.pv: np.ndarray | None = None  # portfolio value
        self.strategy: BacktestStrategy | None = None

    def run_backtest(self, strategy: BacktestStrategy) -> None:
        dl = strategy.dl
        T, N = dl.T, dl.N # T timesteps, N etfs
        
        portfolio_weights = np.empty((T, N+1), dtype=float)
        portfolio_value = np.empty((T, 1), dtype=float)
        v = 1.0 # running value
        w = np.ones(N) / N
        w_cash = 0.0

        portfolio_weights[0, :N] = w
        portfolio_weights[0,  N] = w_cash
        portfolio_value[0] = v

        for t in range(1, T):
            if strategy.get_trade_flag(t) == True:
                # pre market open - configuring the portfolio setup for today, t
                w_prev = w.copy()
                w = strategy.get_weights(t, w_prev)
                w_cash = 1.0 - w.sum()
                turnover = 0.5 * np.abs(w - w_prev).sum()  # pay half of full spread
                v -= self.spread * turnover * v

            # market opens - realizing returns, today t
            r, rf = dl.get_return(t), dl.get_rf(t)
            g = w @ r + w_cash * rf
            v *= g

            # weights have drifted
            w *= r / g
            w_cash *= rf / g

            portfolio_weights[t, :N] = w
            portfolio_weights[t,  N] = w_cash
            portfolio_value[t] = v
        
        self.pw = portfolio_weights
        self.pv = portfolio_value
        self.strategy = strategy

    @property
    def ann_sharpe(self) -> float:
        assert self.pv is not None
        assert self.strategy is not None
        pv = self.pv
        rp = pv[1:] / pv[:-1] - 1.0
        rf = self.strategy.dl._rf[1:].ravel() - 1.0
        return sharpe_ratio(rp, rf)

    @property
    def ann_vol(self) -> float:
        assert self.pv is not None
        pv = self.pv
        rp = pv[1:] / pv[:-1] - 1.0
        return np.std(rp) * np.sqrt(252)

    @property
    def ann_ret(self) -> float:
        assert self.pv is not None
        pv = self.pv
        last = pv[-1, 0]
        return last ** (252 / pv.shape[0]) - 1.0

    @property
    def timeline(self) -> pd.DatetimeIndex:
        assert self.strategy is not None
        return self.strategy.dl.timeline


def sharpe_ratio(rp: np.ndarray, rf: np.ndarray) -> float:
    return (rp - rf).mean() / rp.std(ddof=1) * np.sqrt(252)
