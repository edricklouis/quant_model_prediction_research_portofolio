import pandas as pd

# ------ Rolling Standard Deviation ------
def add_rolling_std_21(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()
    daily_return = df.groupby('ticker')['close'].pct_change()
    
    rolling_std_21 = (
        daily_return
        .groupby(df['ticker'])
        .rolling(window=21)
        .std()
        .reset_index(level=0, drop=True)
    )
    
    df['rolling_std_return_21'] = rolling_std_21

    return df

# ------ Mean Reversion (Z-score) ------
def add_z_score_21(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()
    daily_return = df.groupby('ticker')['close'].pct_change()

    rolling_mean_return_21 = (
        daily_return
        .groupby(df['ticker'])
        .rolling(window=21)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df['z_score_return_21'] = (daily_return - rolling_mean_return_21) / df['rolling_std_return_21']

    return df
