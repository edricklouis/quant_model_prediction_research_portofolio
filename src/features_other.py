import pandas as pd
import numpy as np

# ------ Month ------
def add_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()

    _months = df['date'].dt.month

    df['month_sin'] = np.sin(2 * np.pi * _months / 12)
    df['month_cos'] = np.cos(2 * np.pi * _months / 12)
    
    return df

# ------ Regime ------
def add_regime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()

    bull_trend = df['ema_9'] > df['ema_21']
    bear_trend = df['ema_9'] <= df['ema_21']
    
    price_above_ma = df['close'] > df['ema_9']
    price_below_ma = df['close'] <= df['ema_9']

    high_volume = df['volume'] > df['volume_sma_21'].fillna(0)

    conditions = [
        (bull_trend & price_above_ma & high_volume),
        (bull_trend),
        (bear_trend & price_below_ma & high_volume),
        (bear_trend)
    ]
    
    choices = [3, 2, 0, 1]
    
    regime = np.select(conditions, choices, default=np.nan)
    df['regime'] = pd.Series(regime, index=df.index).astype('Int32')
    
    return df

# ------ RSI Regime ------
def add_rsi_regime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()

    def classify_rsi_int(value):
        if pd.isna(value):
            return pd.NA
        if value < 30:
            return 0
        elif value < 50:
            return 1
        elif value < 70:
            return 2
        else:
            return 3

    df['rsi_14_regime'] = df['rsi_14'].apply(classify_rsi_int).astype('Int32')
    df['rsi_sma_14_regime'] = df['rsi_sma_14'].apply(classify_rsi_int).astype('Int32')

    return df

# ------ Hidden Demand & Supply ------
def add_hidden_demand_supply(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()

    hidden_demand = (df['current_return'] < 0) & (df['cmf_5_delta'] > 0)
    hidden_supply = (df['current_return'] > 0) & (df['cmf_5_delta'] < 0)

    is_any_null = df['current_return'].isna() | df['cmf_5_delta'].isna()

    df['hidden_demand'] = hidden_demand.astype('Int32')
    df['hidden_supply'] = hidden_supply.astype('Int32')

    df.loc[is_any_null, ['hidden_demand', 'hidden_supply']] = pd.NA

    return df

# ------ Money Flow Magnitude ------
def add_money_flow_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()

    df['money_flow_magnitude'] = (df['cmf_5'] * df['volume_spike_ratio'])
    
    return df
