import pandas as pd

# ------ Decision ------
def add_decision_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date']).copy()

    daily_rank_pct_predicted = (
        df
        .groupby('date')['rank_pct_predicted']
        .rank(method='first', ascending=False)
    )

    df['decision'] = (daily_rank_pct_predicted <= 5).astype('Int32')

    return df

# ------ Filter Decision Positive ------
def filter_decision_positive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by=['date', 'rank_pct_predicted'], ascending=[True, False]).copy()
    df = df[df['decision'] == 1]

    return df

# ------ Growth Balance ------
def add_growth_balance(df: pd.DataFrame, starting_balance) -> pd.DataFrame:
    df = df.sort_values(by=['date', 'rank_pct_predicted'], ascending=[True, False]).copy()

    balance = float(starting_balance)
    balances = []

    for _, row in df.iterrows():
        if balance <= 1_000_000:
            balances.append(balance)
            continue

        invested_balance = balance * 0.20
        pnl = invested_balance * row['next_close_pct']
        balance = balance + pnl

        balances.append(balance)

    df['balance'] = balances
    df['balance'] = df['balance'].round(0)    

    return df

# ------ Total PnL Percentage ------
def add_total_pnl_pct(
    df: pd.DataFrame, 
    date, 
    invested_value: str, 
    round_digits
) -> pd.DataFrame:
    base_price = df.loc[df['date'] == date, invested_value].iloc[0]
    
    df['total_pnl_pct'] = (
        (df[invested_value] - base_price) / base_price
    ).round(round_digits)

    return df
