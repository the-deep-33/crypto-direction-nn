"""
Izdvojena Python skripta namenjena transformaciji sirovih
cena Bitcoin-a u diferencirane logaritamske prinose za
dalju obradu.
"""

import numpy as np


def add_log_returns(df, price_col="close", out_col="log_return"):
    df = df.copy()
    df[out_col] = np.log(df[price_col]).diff()
    return df


def build_features(df, n_lags=5, dropna=True):
    df = add_log_returns(df)

    feature_cols = []
    for k in range(n_lags):
        name = f"r_lag_{k}"
        df[name] = df["log_return"].shift(k)
        feature_cols.append(name)

    future = df["log_return"].shift(-1)
    df["label"] = (future > 0).astype(float)
    df.loc[future.isna(), "label"] = np.nan

    if dropna:
        df = df.dropna(subset=feature_cols + ["label"]).reset_index(drop=True)

    return df, feature_cols
