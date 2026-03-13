
# def backtest_data(
#     alphas: pd.DataFrame, 
#     rd: pd.DataFrame,
#     ffr: pd.DataFrame,
#     lookback: int,  # w.r.t. sigma
#     rebal_freq: Literal["D", "W", "M", "Q", "Y", None] = "M",
# ):
#     T, N = rd.shape
#     L = T-lookback

#     covs = np.empty((T-lookback, N, N))
#     for t in range(lookback, T):
#         cov = rd.iloc[t-lookback:t].cov().to_numpy() # excludes day t
#         d, Qm = np.linalg.eigh(cov)                # S = Q @ D @ Q.T
#         d = np.clip(d, 0, None)
#         covs[t-lookback] = np.diag(d**0.5) @ Qm.T  # S_sqrt = D_sqrt @ Q.T

#     rf = ffr[lookback:].to_numpy().ravel()      # includes day t
#     rets = rd[lookback:].to_numpy()             # includes day t
#     alphas = alphas[lookback:].to_numpy()       # includes day t, use t-1

#     timeline = pd.DatetimeIndex(rd.iloc[lookback:].index)
#     months, quarters, years = timeline.month, timeline.quarter, timeline.year
#     match rebal_freq:
#         case "D":  rebal_flag = np.ones(timeline.shape[0], dtype=bool)
#         case "W":  rebal_flag = (timeline.weekday == 4)  # NOTE 4 extra rebal on red friday over 20 years, negligible
#         case "M":  rebal_flag = months != np.roll(months, shift=-1)
#         case "Q":  rebal_flag = quarters != np.roll(quarters, shift=-1)
#         case "Y":  rebal_flag = years != np.roll(years, shift=-1)
#         case None: rebal_flag = np.zeros(timeline.shape[0], dtype=bool)
#         case _: raise ValueError("Invalid rebal frequency")
    
#     assert alphas.shape == (L, N)
#     assert covs.shape == (L, N, N)
#     assert rets.shape == (L, N)
#     assert rf.shape   == (L,)
#     assert rebal_flag.shape == (L,)
    
#     data_bt = {
#         "alpha": alphas,
#         "ret": rets,
#         "cov": covs,
#         "rf": rf,
#         "rebal": rebal_flag,
#         "dates": timeline,
#     }

#     return data_bt



# class OptimizerParams:
#     def __init__(self, 
#             w_prev: Sequence[float], 
#             w_max: float, 
#             gamma: float = 0.02,  # reg - 2 bsp
#             L: float = 0.3,       # lev30
#             vc_lim: float = 0.08, # vc8
#         ):
#         self.gamma = gamma
#         self.w_max = w_max
#         self.w_prev = w_prev
#         self.L = L
#         self.vc_lim = vc_lim

# def weights_fixed(S_sqrt: np.ndarray, op: OptimizerParams):
#     N = S_sqrt.shape[0]
#     w: np.ndarray = np.ones(N) / (N + 1e-8)
#     k = op.vc_lim / (np.linalg.norm(S_sqrt @ w) + 1e-8)
#     k = np.minimum(k, 1.0 + op.L)
#     w = k * w  # scaling for vol control
#     return w

# def weights_markowitz(a: np.ndarray, S_sqrt: np.ndarray, op: OptimizerParams, w_prev: np.ndarray):
#     w: np.ndarray = markowitz(
#         a=a, S_sqrt=S_sqrt, gamma=op.gamma, w_prev=w_prev, 
#         w_max=op.w_max, L=op.L, vc_lim=op.vc_lim
#     )
#     return w


#============
#  BACKTEST
#============

# def run_backtest(
#     tickers: list[str],
#     universe: list[str],
#     data_bt: dict[str, np.ndarray],
#     lookahead: int,   # NOTE cheating!
#     op: OptimizerParams,
#     fixed_weights: bool = False,
# ) -> tuple[np.ndarray, np.ndarray]:
#     name = "-".join(tickers)
#     # print(f"Running Backtest: {name}")
    
#     idx = [i for i, etf in enumerate(universe) if etf in tickers]
    
#     alphas = data_bt["alpha"][:, idx]
#     mu = alphas.mean(axis=1, keepdims=True)
#     s = alphas.std(axis=1, keepdims=True)
#     alphas = (alphas - mu)/s

#     rets = data_bt["ret"][:,idx]
#     covs = data_bt["cov"][:,idx,:][:,:,idx]
#     rf   = data_bt["rf"]
#     rebal_flag = data_bt["rebal"]

#     T, N = rets.shape # T timesteps, N etfs
#     L = T-lookahead-1  # remove -1 and just record drift last day ??

#     portfolio_holdings = np.empty((L, N+1), dtype=float)

#     h      = np.ones(N)  # holdings
#     h_cash = 1.0
#     h_tot  = h.sum() + h_cash
#     w = h / h_tot

#     for t in range(L):

#         # configuring the portfolio setup for today, t
#         if t == 0 or rebal_flag[t] == True:
#             w_prev = w
#             Sigma_half = covs[t]

#             if fixed_weights == True: 
#                 w = weights_fixed(Sigma_half, op) 
#             else:
#                 # alphas = np.ones(N) # hard to beat   # a = rets[t+lookahead] # FIXME
#                 a = alphas[t-1] if t > 0 else np.ones(N)  # pred for y[t-1]: t:t+H-1  <- prediction target
#                 w = weights_markowitz(a, Sigma_half, op, w_prev)
            
#             s, cap = w.sum(), 1.0 + op.L
#             if s > cap + 1e-8:
#                 w *= cap / (s + 1e-8)
            
#             turnover = 0.5 * np.abs(w - w_prev).sum()  # pay half of full spread
#             h_tot -= op.gamma * turnover * h_tot  # self finance transaction costs
#             h      = w * h_tot 
#             h_cash = h_tot - h.sum()

#         portfolio_holdings[t] = np.r_[h, h_cash]

#         # realizing the returns for today, t
#         # update holdings based on todays returns
#         h      *= 1+rets[t]
#         h_cash *= 1+rf[t] # + (funding_spread if h_cash < 0 else 0.0), e.g. funding_spread = 1.1**(1/252)-1 NOTE
#         h_tot = h.sum() + h_cash

#         # update
#         w = h/h_tot
    
#     return pd.DataFrame(
#         data=portfolio_holdings,
#         columns=tickers + ["Cash"],
#         index=data_bt["dates"][:-1], # remove -1 and just record drift last day ??
#     )


#============
#  LOGGING
#============

# save_backtest(metadata, performance, weights) # FIXME





# NOTE Only relevant for sing prediction 
# def _get_classification_statistics(y_true, y_pred):
#     tp = ((y_pred == 1) & (y_true == 1)).sum(axis=1)
#     tn = ((y_pred == -1) & (y_true == -1)).sum(axis=1)
#     fp = ((y_pred == 1) & (y_true == -1)).sum(axis=1)
#     fn = ((y_pred == -1) & (y_true == 1)).sum(axis=1)

#     accuracy = (tp + tn) / (tp + fp + fn + tn)

#     p_pred = (tp + fp).astype(float)
#     n_pred = (tn + fn).astype(float)
#     p_pred[p_pred == 0] = np.nan
#     n_pred[n_pred == 0] = np.nan

#     pre_p = tp / p_pred
#     pre_n = tn / n_pred

#     p_true = (tp + fn).astype(float)
#     n_true = (tn + fp).astype(float)
#     p_true[p_true == 0] = np.nan
#     n_true[n_true == 0] = np.nan

#     rec_p = tp / p_true
#     rec_n = tn / n_true

#     p_pre_rec = pre_p + rec_p
#     n_pre_rec = pre_n + rec_n
#     p_pre_rec[p_pre_rec == 0] = np.nan
#     n_pre_rec[n_pre_rec == 0] = np.nan

#     f1_p = 2 * (pre_p * rec_p) / p_pre_rec
#     f1_n = 2 * (pre_n * rec_n) / n_pre_rec

#     return accuracy, (pre_p, rec_p, f1_p), (pre_n, rec_n, f1_n)
