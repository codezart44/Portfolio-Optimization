# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D0 = "2005-01-03"
D1 = "2024-12-31"

factors = ["SPY", "AGG", "GLD"]
sectors = ["XLK", "XLV", "XLF", "XLY", "XLI", "XLP", "XLE", "XLU", "XLB"]
interna = ["EWJ", "EWG", "EWU", "EWA", "EWH", "EWS", "EWZ", "EWT", "EWY", "EWP", "EWW", "EWD", "EWL", "EWC"]

rd = pd.read_parquet("../data/return/return_d.parquet").loc[D0:D1]
rf = pd.read_parquet("../data/return/ffr_d.parquet").reindex(rd.index)
rx = rd - rf.values

# %%
# R = XB + E
# R in R^{T,N}, X in R^{T,K}, B in R^{K,N}, E in R^{T,N}
# X includes the intercept

def idiosyncratic_returns(
        r: pd.DataFrame, 
        factors: list[str], 
        assets: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
    T = r.shape[0]
    Rt = r[assets].values
    Xx = np.c_[np.ones(T), r[factors].values]
    Bb = np.linalg.solve(Xx.T @ Xx, Xx.T @ Rt)
    Ee = Rt - Xx @ Bb
    return Bb, Ee

Beta, Eps = idiosyncratic_returns(rx, factors, sectors)
df_eps = pd.DataFrame(Eps, columns=sectors, index=rx.index)
((1.0+df_eps).cumprod()-1.0).plot(legend=True)

# Two forms of regression happning
# 1. Factor removal from returns - residual returns
# 2. Fittign the OLS model - Alpha model

# NOTE start by precomputing the residual returns at each time step
# targets and features in blocks of data
# preprocess all data and save down to npz,
# then run model fit loop on top of that

def rolling_target() -> np.ndarray:
    ...

def rolling_features() -> np.ndarray:
    ...
    # add cyclif FE - seasons, day of year sin and cos

def fit_model() -> np.ndarray:
    ...





# %%
# NOTE Regress Sectors and International ETFs on market (potentially gold and bonds as well)
# Rank idiosyncratic returns to get assets that perform well orthogonally to the market
# Otherwise predictor may rank which ETF has biggest market component

# XLK - Technology
# XLV - Heathcare
# XLF - Financials
# XLY - Consumer Discretionary
# XLI - Industrials
# XLP - Consumer Staples
# XLE - Energy
# XLU - Utilities
# XLB - Materials

# IBB - Biotech
# IYR - Real Estate

# rd[[
#     "SPY", "AGG", "GLD",
#     "XLK", "XLV", "XLF", "XLY", "XLI", "XLP", "XLE", "XLU", "XLB",  # sectors
#     "IBB", "IYR",  # extra
#     # "EWJ", "EWG", "EWU", "EWA", "EWH", "EWS", "EWZ", "EWT", "EWY", "EWP", "EWW", "EWD", "EWL", "EWC",  # international
# ]].corr()


# r = B * rm + e
# e = r - B * rm
# rm = rx[["SPY"]].values
# r  = rx[sect].values
# Beta = r.T @ rm / (rm.T @ rm)
# eps = r - Beta.T * rm
# df_beta = pd.Series(Beta.ravel(), index=sect)
# df_eps = pd.DataFrame(eps, columns=sect)
# df_eps.corr()