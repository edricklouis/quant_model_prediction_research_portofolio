import pandas as pd

# ------ Volume SMA ------
def add_volume_sma(df: pd.DataFrame, round_digits) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()
    
    df['volume_sma_21'] = df.groupby('ticker')['volume'].transform(
        lambda x: x.rolling(window=21).mean()
    ).round(round_digits)
    
    return df

# ------ Volume Weighted Average Price (VWAP) ------
def add_rolling_vwap(df: pd.DataFrame, round_digits) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()

    typical_price = (df['high'] + df['low'] + df['close']) / 3
    price_volume = typical_price * df['volume']

    pv_sum = price_volume.groupby(df['ticker']).transform(
        lambda x: x.rolling(window=21).sum()
    )
    vol_sum = df['volume'].groupby(df['ticker']).transform(
        lambda x: x.rolling(window=21).sum()
    )

    df['vwap_21'] = pv_sum / vol_sum
    df['vwap_21'] = df['vwap_21'].round(round_digits)

    return df
