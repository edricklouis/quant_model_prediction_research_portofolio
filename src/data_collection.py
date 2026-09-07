import yfinance as yf
import pandas as pd

# ------ Retrieve Tickers Data ------
def fetch_tickers_data(tickers, start_date, end_date, interval):
    all_data = []
    success_tickers = []
    total = len(tickers)

    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{total}] Downloading {ticker} ...")

        try:
            df_data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=False
            )

            if df_data.empty:
                print(f"⚠️ Data {ticker} tidak ditemukan!")
                continue

            if isinstance(df_data.columns, pd.MultiIndex):
                df_data.columns = [col[0] for col in df_data.columns]

            df_data = df_data.reset_index()
            df_data["ticker"] = ticker
            df_data = df_data.rename(columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume"
            })
            df_data = df_data[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]]

            numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
            df_data[numeric_cols] = (
                df_data[numeric_cols]
                .apply(pd.to_numeric, errors="coerce")
                .astype("float64")
                .round(2)
            )
            df_data = df_data.dropna(subset=numeric_cols)
            all_data.append(df_data)
            success_tickers.append(ticker)

        except Exception as e:
            print(f"❌ Gagal download {ticker}: {e}")

    df_all_data = pd.concat(all_data, ignore_index=True)

    return df_all_data, success_tickers

# ------ Retrieve Index Data ------
def fetch_index_data(index, start_date, end_date, interval):
    all_data = []
    total = len(index)

    for i, ticker in enumerate(index, start=1):
        print(f"[{i}/{total}] Downloading {ticker} ...")

        try:
            df_data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=False
            )

            if df_data.empty:
                print(f"⚠️ Data {ticker} tidak ditemukan!")
                continue

            if isinstance(df_data.columns, pd.MultiIndex):
                df_data.columns = [col[0] for col in df_data.columns]

            df_data = df_data.reset_index()
            df_data["ticker"] = ticker
            df_data = df_data.rename(columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume"
            })
            df_data = df_data[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]]

            numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
            df_data[numeric_cols] = (
                df_data[numeric_cols]
                .apply(pd.to_numeric, errors="coerce")
                .astype("float64")
                .round(2)
            )
            df_data = df_data.dropna(subset=numeric_cols)
            all_data.append(df_data)

        except Exception as e:
            print(f"❌ Gagal download {ticker}: {e}")

    df_all_data_index = pd.concat(all_data, ignore_index=True)

    if not all_data:
        return pd.DataFrame()

    return df_all_data_index
