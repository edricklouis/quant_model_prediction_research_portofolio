import pandas as pd

# ------ Get Data Predict ------
def get_data_predict(df: pd.DataFrame, predict_date: str) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    target_date = pd.to_datetime(predict_date)
    df_data_predict = df[df['date'] == target_date].reset_index(drop=True)

    return df_data_predict

# ------ Remove First Year ------
def remove_first_year_per_ticker(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    cleaned_dfs = []

    for ticker, group in df.groupby('ticker', group_keys=False):
        start_date = group['date'].min()
        cutoff_date = start_date + pd.DateOffset(years=1)
        group_cleaned = group[group['date'] > cutoff_date]
        cleaned_dfs.append(group_cleaned)

    df_cleaned = pd.concat(cleaned_dfs, ignore_index=True)

    return df_cleaned

# ------ Remove Duplicate ------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=['ticker', 'date'], keep='first').reset_index(drop=True)

    return df

# ------ Remove Null Values ------
def remove_null_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna().reset_index(drop=True)
    
    return df

# ------ Split Data Train & Test ------
def split_train_val_test(
    df: pd.DataFrame,
    start_train: str,
    end_train: str,
    start_val: str,
    end_val: str,
    start_test: str,
    end_test: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    df_data_train = df[(df['date'] >= pd.to_datetime(start_train)) & (df['date'] <= pd.to_datetime(end_train))].reset_index(drop=True)
    df_data_val = df[(df['date'] >= pd.to_datetime(start_val)) & (df['date'] <= pd.to_datetime(end_val))].reset_index(drop=True)
    df_data_test  = df[(df['date'] >= pd.to_datetime(start_test))  & (df['date'] <= pd.to_datetime(end_test))].reset_index(drop=True)

    return df_data_train, df_data_val, df_data_test

# ------ Remove Data With Same OHLC Value ------
def remove_same_ohlc_rows(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df[~((df['open'] == df['high']) &
                    (df['high'] == df['low']) &
                    (df['low'] == df['close']))].reset_index(drop=True)

    return df_clean

# ------ Remove Data With 0 Volume ------
def remove_zero_volume_rows(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df[df['volume'] > 0].reset_index(drop=True)

    return df_clean

# ------ Drop Columns ------
def drop_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    df = df.copy()
    existing_cols = [col for col in columns_to_drop if col in df.columns]

    df = df.drop(columns=existing_cols)

    return df
