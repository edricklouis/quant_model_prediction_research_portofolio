import pandas as pd
import numpy as np

# ------ Next Close Labeling ------
def add_next_close_label(df: pd.DataFrame, round_digits) -> pd.DataFrame:
    df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)

    df['next_close'] = df.groupby('ticker')['close'].shift(-1)
    df['next_close'] = df['next_close'].round(round_digits)
    df['next_close_pct'] = (df['next_close'] - df['close']) / df['close']
    df['next_close_pct'] = df['next_close_pct'].round(round_digits)

    return df

# ------ Rank Labeling ------
def add_rank_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)

    df['rank'] = (
        df
        .groupby('date')['next_close_pct']
        .rank(method='min', ascending=False)
        .astype('Int32')
    )
    df['rank_pct'] = (
        df
        .groupby('date')['next_close_pct']
        .rank(pct=True, method='average', ascending=True)
    ).round(4)

    conditions = [
        (df['rank_pct'] > 0.90),
        (df['rank_pct'] > 0.50),
        (df['rank_pct'] > 0.10),
        (True)
    ]
    choices = [3, 2, 1, 0]
    df['rank_class'] = np.select(conditions, choices, default=1)
    df['rank_class'] = df['rank_class'].astype('Int32')
    
    return df

# ------ Triple Barrier Method (TBM) Labeling ------
# def add_tbm_label(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)

#     daily_return = df.groupby('ticker')['close'].pct_change()
#     volatility = daily_return.groupby(df['ticker']).transform(
#         lambda x: x.rolling(window=21).std()
#     )
    
#     dynamic_threshold = volatility * 2

#     up = df['next_close_pct'] >= dynamic_threshold
#     neutral = (df['next_close_pct'] > -dynamic_threshold) & (df['next_close_pct'] < dynamic_threshold)
#     down = df['next_close_pct'] <= -dynamic_threshold
    
#     conditions = [up, neutral, down]
#     choices = [2, 1, 0]

#     df['label_tbm'] = np.select(conditions, choices, default=np.nan)
#     df['label_tbm'] = df['label_tbm'].astype('Int32')
    
#     return df
