import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/titan_raw.csv")

df["Date"] = pd.to_datetime(df["Date"])

df["MA_20"] = df["Close"].rolling(window=20).mean()
df["MA_50"] = df["Close"].rolling(window=50).mean()

plt.figure(figsize=(14,7))

plt.plot(df["Date"], df["Close"], label="Close Price")

plt.plot(df["Date"], df["MA_20"], label="20-Day MA")

plt.plot(df["Date"], df["MA_50"], label="50-Day MA")

plt.title("Titan Stock Price with Moving Averages")

plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()

plt.grid(True)

plt.savefig("plots/moving_average.png", dpi=300, bbox_inches="tight")


plt.show()