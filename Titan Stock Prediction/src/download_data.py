import yfinance as yf

ticker = yf.Ticker("TITAN.NS")

df = ticker.history(period="10y")

df.to_csv("data/raw/titan_raw.csv")

print("Titan stock data downloaded successfully!")