import pandas as pd
import numpy as np

# ------ Price Level ------
def add_price_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()
    
    bins = [0, 200, 500, 2000, 5000, float('inf')]
    labels = [1, 2, 3, 4, 5]
    
    df['price_level'] = pd.cut(df['close'], bins=bins, labels=labels, right=False)
    df['price_level'] = df['price_level'].astype('Int32')
    
    return df

# ------ Current Return ------
def add_current_return(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()
    current_return = df.groupby('ticker')['close'].pct_change()

    df['current_return'] = current_return

    return df

# ------ Lagged Returns ------
def add_lagged_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()
    daily_return = df.groupby('ticker')['close'].pct_change()

    lag_list = [1, 2, 3, 5, 10, 21]

    for lag in lag_list:
        df[f'lag_return_{lag}'] = (
            daily_return
            .groupby(df['ticker'])
            .shift(lag)
        )

    return df

# ------ Current Volatility ------
def add_current_volatility(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()
    current_volatility = ((df['high'] - df['low']) / df['close'])

    df['current_volatility'] = current_volatility

    return df

# ------ Lagged Volatility ------
def add_lagged_volatility(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()
    daily_volatility = ((df['high'] - df['low']) / df['close'])

    for lag in range(1, 4):
        df[f'lag_volatility_{lag}'] = (
            daily_volatility
            .groupby(df['ticker'])
            .shift(lag)
        )

    return df

# ------ Rolling Skewness & Kurtosis Return ------
def add_rolling_skew_kurt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()
    daily_return = df.groupby('ticker')['close'].pct_change()

    df['rolling_skew_21'] = (
        daily_return
        .groupby(df['ticker'])
        .rolling(window=21)
        .skew()
        .reset_index(level=0, drop=True)
    )

    df['rolling_kurt_21'] = (
        daily_return
        .groupby(df['ticker'])
        .rolling(window=21)
        .kurt()
        .reset_index(level=0, drop=True)
    )
    
    return df

# ------ Garman Klass Variance ------
def add_garman_klass_variance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()

    log_hl = np.log(df['high'] / df['low'])
    log_oc = np.log(df['close'] / df['open'])

    coefficient = 2 * np.log(2) - 1

    gk_var = 0.5 * (log_hl ** 2) - coefficient * (log_oc ** 2)

    df['garman_klass_variance'] = np.maximum(gk_var, 0.0)

    return df
