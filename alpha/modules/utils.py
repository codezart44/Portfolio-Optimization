import numpy as np
import pandas as pd
from typing import Literal, Sequence
import os

def rmse(e: np.ndarray, axis=0) -> float:
    return np.sqrt(np.mean(e**2, axis=axis))

def mae(e: np.ndarray, axis=0) -> float:
    return np.mean(np.abs(e), axis=axis)

def nrmse_score(
        df_prd: pd.DataFrame,
        df_ref: pd.DataFrame,
        axis: int = 0
    ) -> pd.DataFrame:
    mu = df_ref.mean(axis=axis)
    nrmse_asset = (rmse(df_ref - df_prd) / rmse(df_ref - mu, axis=axis))
    return nrmse_asset

def r2_score(
        df_prd: pd.DataFrame, 
        df_ref: pd.DataFrame,
        axis: int = 0,
    ) -> pd.DataFrame:
    mu = df_ref.mean(axis=axis)
    r2_asset = 1 - ((df_ref - df_prd)**2).sum(axis=0) / ((df_ref - mu)**2).sum(axis=axis)
    return r2_asset

def ic_score(  # NOTE Preferable for cross-sectional asset prediction
        df_prd: pd.DataFrame, 
        df_ref: pd.DataFrame, 
        method: Literal["pearson", "spearman"] = "spearman",
        axis: int = 1,
    ) -> pd.Series:
    ic = df_prd.corrwith(df_ref, axis=axis, method=method,)
    return ic

def t_test(
        samples: pd.Series,
        mu_h0: float = 0.0,
    ) -> float:
    mu = samples.mean()
    sd = samples.std()
    N = samples.count()
    t_val = (mu - mu_h0) / (sd / np.sqrt(N))
    return t_val


def save_alpha_result(
        file_path: str,
        df_prd: pd.Series,
        df_ref: pd.Series,
        gamma: float,
        halflife: int,
        lookback: int,
        horizon: int,
        method: str,
        const_pred: float,
        xshape: tuple,
        yshape: tuple,
        assets: list[str],
        features_mac: list[str],
        features_ret: list[str],
        target: str,
    ) -> None:
    assert isinstance(df_prd, pd.DataFrame), type(df_prd)
    assert isinstance(df_ref, pd.DataFrame), type(df_ref)
    assert df_prd.shape == df_ref.shape, (df_prd.shape, df_ref.shape)
    assert isinstance(assets, str), type(assets)
    assert isinstance(features_mac, str), type(features_mac)
    assert isinstance(features_ret, str), type(features_ret)
    # Eval Scores

    ics = ic_score(df_prd, df_ref, method="spearman", axis=1)  # cross-sectional ranking (strong negative is also good)
    icp = ic_score(df_prd, df_ref, method="pearson", axis=1)
    t_ics = t_test(ics, mu_h0=0.0)  # t-test for ic score
    t_icp = t_test(icp, mu_h0=0.0)
    nrmse = nrmse_score(df_prd, df_ref, axis=0)  # timing
    r2 = r2_score(df_prd, df_ref, axis=0)
    
    df_row = pd.DataFrame([{
        "ics": np.nan_to_num(ics.mean(), nan=0.0).round(4).item(),
        "icp": np.nan_to_num(icp.mean(), nan=0.0).round(4).item(),
        "tics": np.nan_to_num(t_ics, nan=0.0).round(4).item(),
        "ticp": np.nan_to_num(t_icp, nan=0.0).round(4).item(),
        "r2": r2.mean().round(4).item(),
        "nrmse": nrmse.mean().round(4).item(),
        "gamma": gamma,
        "halflife": halflife,
        "lookback": lookback,
        "horizon": horizon,
        "method": method,
        "constpred": const_pred if method == "C" else None,
        "T": yshape[0],
        "N": yshape[1],
        "F": xshape[1] // yshape[1],
        "assets": assets,
        "macrof": "" if method == "C" else features_mac,
        "returnf": "" if method == "C" else features_ret,
        "target": target,
    }])
    df_row.to_csv(
        file_path, 
        mode="a", 
        header=not os.path.exists(file_path), 
        index=False
    )

