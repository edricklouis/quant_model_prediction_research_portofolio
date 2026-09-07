import pandas as pd
import numpy as np

# ------ Exponential Moving Average (EMA) ------
def add_ema_features(df: pd.DataFrame, round_digits) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()
    ema_periods = [9, 21, 50, 200]

    for period in ema_periods:
        ema_col = f'ema_{period}'

        df[ema_col] = (
            df.groupby('ticker')['close']
              .transform(lambda x: x.ewm(span=period, adjust=False, min_periods=period).mean())
              .round(round_digits)
        )

    return df

# ------ Relative Strength Index (RSI) ------
def add_rsi_features(df: pd.DataFrame, round_digits) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()
    period = 14

    def compute_rsi_tv(close_series, period=14):
        delta = close_series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        rma_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        rma_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

        rs = rma_gain / rma_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['rsi_14'] = (
        df.groupby('ticker')['close']
          .transform(lambda x: compute_rsi_tv(x, period))
          .round(round_digits)
    )

    df['rsi_sma_14'] = (
        df.groupby('ticker')['rsi_14']
          .transform(lambda x: x.rolling(period, min_periods=period).mean())
          .round(round_digits)
    )

    return df

# ------ Moving Average Convergence Divergence (MACD) ------
def add_macd_features(df: pd.DataFrame, round_digits) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()

    ema_fast = (
        df.groupby('ticker')['close']
          .transform(lambda x: x.ewm(span=12, adjust=False, min_periods=12).mean())
    )
    ema_slow = (
        df.groupby('ticker')['close']
          .transform(lambda x: x.ewm(span=26, adjust=False, min_periods=26).mean())
    )
    macd_line = ema_fast - ema_slow

    macd_signal = (
        macd_line.groupby(df['ticker'])
        .transform(lambda x: x.ewm(span=9, adjust=False, min_periods=9).mean())
    )

    macd_hist = macd_line - macd_signal

    df['macd_12_26'] = macd_line.round(round_digits)
    df['macd_signal_ema_9'] = macd_signal.round(round_digits)
    df['macd_histogram'] = macd_hist.round(round_digits)

    return df

# ------ Average True Range (ATR) ------
def add_atr_features(df: pd.DataFrame, round_digits) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()

    close_prev = df.groupby('ticker')['close'].shift(1)

    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - close_prev).abs()
    tr3 = (df['low'] - close_prev).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df['atr_14'] = (
        tr
        .groupby(df['ticker'])
        .transform(lambda x: x.ewm(alpha=1/14, adjust=False).mean())
        .round(round_digits)
    )

    def invalidate_first_n(group):
        group.iloc[:14] = pd.NA

        return group

    df['atr_14'] = (
        df
        .groupby('ticker')['atr_14']
        .apply(invalidate_first_n)
        .reset_index(level=0, drop=True)
    )

    return df

# ------ Bollinger Bands ------
def add_bollinger_bands_features(df: pd.DataFrame, round_digits) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()

    close = df['close']

    mid = close.groupby(df['ticker']).transform(lambda x: x.rolling(20).mean())
    std = close.groupby(df['ticker']).transform(lambda x: x.rolling(20).std(ddof=0))

    upper = mid + 2 * std
    lower = mid - 2 * std

    df['bb_mid_20'] = mid.round(round_digits)
    df['bb_upper_20'] = upper.round(round_digits)
    df['bb_lower_20'] = lower.round(round_digits)

    return df

# ------ Highest 'high' ------ 
def add_highest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()

    highest_50 = (
        df.groupby('ticker')['high']
        .transform(lambda x: x.rolling(window=50, min_periods=50).max())
    )
    highest_200 = (
        df.groupby('ticker')['high']
        .transform(lambda x: x.rolling(window=200, min_periods=200).max())
    )

    df['highest_50'] = highest_50
    df['highest_200'] = highest_200

    return df

# ------ Chaikin Money Flow (CMF) ------ 
def add_cmf_features(df: pd.DataFrame, round_digits) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()

    ad_val = (df['close'] - df['low']) - (df['high'] - df['close'])
    hl_range = df['high'] - df['low']
    mf_multiplier = np.where(hl_range == 0, 0, ad_val / hl_range)
    mf_volume = mf_multiplier * df['volume']

    rolling_mfv = mf_volume.groupby(df['ticker']).transform(lambda x: x.rolling(5).sum())
    rolling_vol = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(5).sum())
    cmf = (rolling_mfv / rolling_vol)

    df['cmf_5'] = cmf.round(round_digits)
    df['cmf_5_delta'] = (
        df.groupby('ticker')['cmf_5']
          .transform(lambda x: x.diff())
          .round(round_digits)
    )

    return df

# ------ Money Flow Index (MFI) ------
def add_mfi_features(df: pd.DataFrame, round_digits: int) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()

    typical_price = (df['high'] + df['low'] + df['close']) / 3
    rmf = typical_price * df['volume']

    tp_diff = typical_price.groupby(df['ticker']).diff()

    pos_flow = pd.Series(np.where(tp_diff > 0, rmf, 0), index=df.index)
    neg_flow = pd.Series(np.where(tp_diff < 0, rmf, 0), index=df.index)

    pos_mf_sum = pos_flow.groupby(df['ticker']).transform(
        lambda x: x.rolling(window=14).sum()
    )
    neg_mf_sum = neg_flow.groupby(df['ticker']).transform(
        lambda x: x.rolling(window=14).sum()
    )

    total_mf = pos_mf_sum + neg_mf_sum
    mfi = 100 * (pos_mf_sum / total_mf)

    df['mfi_14'] = mfi.round(round_digits)

    return df
