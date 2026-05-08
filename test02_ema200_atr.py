import backtrader as bt
import yfinance as yf
import pandas as pd


class EmaAtrStrategy(bt.Strategy):
    params = (
        ("ema_period", 200),
        ("volume_period", 20),
        ("atr_period", 14),
        ("atr_mult", 2.0),
        ("rr", 2.0),
        ("pullback_period", 5),
    )

    def __init__(self):
        self.ema200 = bt.ind.EMA(self.data.close, period=self.params.ema_period)
        self.vol_ma = bt.ind.SMA(self.data.volume, period=self.params.volume_period)
        self.atr = bt.ind.ATR(self.data, period=self.params.atr_period)

        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.take_profit = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.entry_price = order.executed.price
                risk = self.atr[0] * self.params.atr_mult
                self.stop_price = self.entry_price - risk
                self.take_profit = self.entry_price + risk * self.params.rr

                print(
                    f"BUY | Price: {self.entry_price:.2f} | "
                    f"SL: {self.stop_price:.2f} | TP: {self.take_profit:.2f}"
                )

            elif order.issell():
                print(f"SELL | Price: {order.executed.price:.2f}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print("ORDER CANCELED / MARGIN / REJECTED")

        self.order = None

    def next(self):
        if self.order:
            return

        close = self.data.close[0]
        open_ = self.data.open[0]
        low = self.data.low[0]
        high = self.data.high[0]
        volume = self.data.volume[0]

        green_candle = close > open_
        trend_up = close > self.ema200[0]
        volume_ok = volume > self.vol_ma[0]

        recent_low = min(self.data.low.get(size=self.params.pullback_period))
        had_pullback = recent_low < self.ema200[0] or close > recent_low * 1.03

        if not self.position:
            if trend_up and green_candle and volume_ok and had_pullback:
                self.order = self.buy()

        else:
            if low <= self.stop_price:
                self.order = self.sell()
            elif high >= self.take_profit:
                self.order = self.sell()
            elif close < self.ema200[0]:
                self.order = self.sell()


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

start_price = df["Close"].iloc[0]
end_price = df["Close"].iloc[-1]
buy_hold_return = (end_price / start_price - 1) * 100
print(f"Buy & Hold Return: {buy_hold_return:.2f}%")

data = bt.feeds.PandasData(dataname=df)

cerebro = bt.Cerebro()
cerebro.addstrategy(EmaAtrStrategy)
cerebro.adddata(data)

cerebro.broker.setcash(10000)
cerebro.broker.setcommission(commission=0.001)
cerebro.broker.set_slippage_perc(perc=0.001)

cerebro.addsizer(bt.sizers.PercentSizer, percents=95)

cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

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

cerebro.plot()