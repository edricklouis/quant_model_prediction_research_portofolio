import pandas as pd
import numpy as np

# ------ Volume Spike Ratio ------
def add_volume_spike_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()
    
    df['volume_spike_ratio'] = (df['volume'] / df['volume_sma_21']) - 1
    
    return df

# ------ Normalized 'volume' Using Z-score ------
def add_z_score_volume(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()

    rolling_std = df.groupby('ticker')['volume'].transform(
        lambda x: x.rolling(window=21).std()
    )

    df['z_score_volume_21'] = (df['volume'] - df['volume_sma_21']) / (rolling_std + 1e-8)

    return df

# ------ 'close' to 'vwap_21' Ratio ------
def add_vwap_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()

    df['close_vwap_21_ratio'] = (df['close'] / df['vwap_21']) - 1

    return df

# ------ EMA Ratio by 'close' ------
def add_ema_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()
    ema_cols = ['ema_9', 'ema_21', 'ema_50', 'ema_200']

    for col in ema_cols:
        new_col = f'{col}_ratio'

        if col in df.columns:
            df[new_col] = (df['close'] / df[col]) - 1
        else:
            df[new_col] = np.nan

    return df

# ------ 'macd_histogram' Normalized by 'close' ------
def add_macd_hist_norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()

    df['macd_hist_norm_close'] = df['macd_histogram'] / df['close']

    return df

# ------ Normalized 'atr_14' Using Z-score ------
def add_z_score_atr_14(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()
    atr_norm = df['atr_14'] / df['close']

    df['z_score_atr_14'] = (
        atr_norm
        .groupby(df['ticker'])
        .transform(lambda x: (x - x.mean()) / x.std())
    )

    return df

# ------ Band Position (BB) ------
def add_bb_pct_b(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()

    upper = df['bb_upper_20']
    lower = df['bb_lower_20']
    close = df['close']

    denom = upper - lower
    bb_pct_b = ((close - lower) / denom)

    df['bb_pct_b'] = (
        bb_pct_b
        .groupby(df['ticker'])
        .transform(lambda x: x)
    )
    
    return df

# ------ Highest Ratio by 'close' ------
def add_close_highest_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()
    
    df['close_highest_50_ratio'] = (df['close'] / df['highest_50']) - 1
    df['close_highest_200_ratio'] = (df['close'] / df['highest_200']) - 1
    
    return df

# ------ Normalized 'garman_klass_variance' Using Z-score & Rank ------
def add_z_score_rank_gk_var_21(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()

    rolling_mean = (
        df.groupby('ticker')['garman_klass_variance']
        .transform(lambda x: x.rolling(21).mean())
    )

    rolling_std = (
        df.groupby('ticker')['garman_klass_variance']
        .transform(lambda x: x.rolling(21).std())
    )

    df['z_score_gk_var_21'] = (df['garman_klass_variance'] - rolling_mean) / rolling_std
    df['z_score_gk_var_21_rank_pct'] = (
        df.groupby('date')['z_score_gk_var_21']
        .rank(pct=True, method='average', ascending=True)
    )

    return df
