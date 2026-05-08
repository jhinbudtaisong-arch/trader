import backtrader as bt
import yfinance as yf
import pandas as pd




class SmaCrossStrategy(bt.Strategy):
    params = (
        ("fast", 10),
        ("slow", 30),
    )

    def __init__(self):
        sma_fast = bt.ind.SMA(period=self.params.fast)
        sma_slow = bt.ind.SMA(period=self.params.slow)
        self.crossover = bt.ind.CrossOver(sma_fast, sma_slow)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        else:
            if self.crossover < 0:
                self.sell()


df = yf.download(
    "BTC-USD",
    start="2023-01-01",
    end="2025-01-01",
    interval="1d",
    auto_adjust=True,
)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[["Open", "High", "Low", "Close", "Volume"]]
df.dropna(inplace=True)

data = bt.feeds.PandasData(dataname=df)

start_price = df["Close"].iloc[0]
end_price = df["Close"].iloc[-1]

buy_hold_return = (end_price / start_price - 1) * 100

print(f"Buy & Hold Return: {buy_hold_return:.2f}%")

cerebro = bt.Cerebro()
cerebro.addstrategy(SmaCrossStrategy)
cerebro.adddata(data)

cerebro.broker.setcash(10000)
cerebro.broker.setcommission(commission=0.001)

cerebro.addsizer(bt.sizers.PercentSizer, percents=95)
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
cerebro.broker.set_slippage_perc(perc=0.001)
print("Starting Portfolio Value:", cerebro.broker.getvalue())


results = cerebro.run()
strategy = results[0]

print("Final Portfolio Value:", cerebro.broker.getvalue())

print("\n--- Trade Analysis ---")
print(strategy.analyzers.trades.get_analysis())

print("\n--- Drawdown ---")
print(strategy.analyzers.drawdown.get_analysis())

print("\n--- Sharpe Ratio ---")
print(strategy.analyzers.sharpe.get_analysis())

print("\n--- Returns ---")
print(strategy.analyzers.returns.get_analysis())

print("Final Portfolio Value:", cerebro.broker.getvalue())

cerebro.plot()