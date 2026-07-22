import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/titan_raw.csv")

df["Date"] = pd.to_datetime(df["Date"])

df["Daily_Return"] = df["Close"].pct_change()

df["MA5"] = df["Close"].rolling(window=5).mean()

df["MA20"] = df["Close"].rolling(window=20).mean()

df["MA5_to_MA20"] = df["MA5"] / df["MA20"]

df["Price_to_MA20"] = df["Close"] / df["MA20"]

df["Lag1_Return"] = df["Daily_Return"].shift(1)

df["Lag2_Return"] = df["Daily_Return"].shift(2)

df["Volatility_5d"] = df["Daily_Return"].rolling(window=5).std()

df["Volume_Change"] = df["Volume"].pct_change()

df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.dropna(inplace=True)

df.to_csv("data/processed/titan_features.csv", index=False)

print("Processed dataset saved successfully!")
print(df.head())
print(df.columns)