import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np


# =========================================================
# 1) Calculate approximate rolling Volume Profile
# =========================================================

def calculate_volume_profile_for_window(window_df, bins=24, value_area_pct=0.70):
    """
    Approximate Volume Profile:
    - Uses typical price per candle
    - Allocates whole candle volume to one price bin
    - Not tick-accurate, but enough for logic testing
    """

    low_price = window_df["Low"].min()
    high_price = window_df["High"].max()

    if high_price <= low_price:
        return np.nan, np.nan, np.nan, "UNKNOWN"

    prices = (window_df["High"] + window_df["Low"] + window_df["Close"]) / 3
    volumes = window_df["Volume"]

    bin_edges = np.linspace(low_price, high_price, bins + 1)
    volume_by_bin = np.zeros(bins)

    for price, volume in zip(prices, volumes):
        idx = np.searchsorted(bin_edges, price, side="right") - 1
        idx = max(0, min(idx, bins - 1))
        volume_by_bin[idx] += volume

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    poc_idx = int(np.argmax(volume_by_bin))
    poc = bin_centers[poc_idx]

    total_volume = volume_by_bin.sum()
    target_volume = total_volume * value_area_pct

    selected = {poc_idx}
    selected_volume = volume_by_bin[poc_idx]

    left = poc_idx - 1
    right = poc_idx + 1

    while selected_volume < target_volume and (left >= 0 or right < bins):
        left_vol = volume_by_bin[left] if left >= 0 else -1
        right_vol = volume_by_bin[right] if right < bins else -1

        if right_vol >= left_vol:
            selected.add(right)
            selected_volume += right_vol
            right += 1
        else:
            selected.add(left)
            selected_volume += left_vol
            left -= 1

    selected_bins = sorted(selected)
    val = bin_centers[selected_bins[0]]
    vah = bin_centers[selected_bins[-1]]

    # Detect rough B / P / D profile
    upper_volume = volume_by_bin[int(bins * 0.66):].sum()
    middle_volume = volume_by_bin[int(bins * 0.33):int(bins * 0.66)].sum()
    lower_volume = volume_by_bin[:int(bins * 0.33)].sum()

    close = window_df["Close"].iloc[-1]
    price_range = high_price - low_price
    close_position = (close - low_price) / price_range if price_range > 0 else 0.5

    if middle_volume > upper_volume and middle_volume > lower_volume:
        profile_type = "D"  # balance
    elif lower_volume > upper_volume and close_position > 0.55:
        profile_type = "P"  # rally / short covering
    elif upper_volume > lower_volume and close_position < 0.45:
        profile_type = "B"  # selloff / long liquidation
    else:
        profile_type = "NEUTRAL"

    return poc, vah, val, profile_type


def add_rolling_volume_profile(df, lookback=60, bins=24):
    poc_list = []
    vah_list = []
    val_list = []
    profile_list = []

    for i in range(len(df)):
        if i < lookback:
            poc_list.append(np.nan)
            vah_list.append(np.nan)
            val_list.append(np.nan)
            profile_list.append("UNKNOWN")
            continue

        window_df = df.iloc[i - lookback:i]
        poc, vah, val, profile_type = calculate_volume_profile_for_window(
            window_df,
            bins=bins,
            value_area_pct=0.70,
        )

        poc_list.append(poc)
        vah_list.append(vah)
        val_list.append(val)
        profile_list.append(profile_type)

    df["POC"] = poc_list
    df["VAH"] = vah_list
    df["VAL"] = val_list
    df["ProfileCode"] = profile_list

    code_map = {
        "UNKNOWN": 0,
        "D": 1,
        "P": 2,
        "B": 3,
        "NEUTRAL": 4,
    }

    df["ProfileType"] = df["ProfileCode"].map(code_map).fillna(0)

    return df


# =========================================================
# 2) Custom Backtrader Data Feed
# =========================================================

class PandasVolumeProfileData(bt.feeds.PandasData):
    lines = ("poc", "vah", "val", "profile_type",)

    params = (
        ("datetime", None),
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
        ("openinterest", None),
        ("poc", "POC"),
        ("vah", "VAH"),
        ("val", "VAL"),
        ("profile_type", "ProfileType"),
    )


# =========================================================
# 3) Strategy: EMA200 + Volume Profile + B/P/D + ATR Trailing
# =========================================================

class VolumeProfileBPDStrategy(bt.Strategy):
    params = (
        ("ema_fast", 50),
        ("ema_slow", 200),
        ("ema_slope_lookback", 10),
        ("volume_period", 20),
        ("volume_mult", 1.0),
        ("atr_period", 14),
        ("atr_mult", 3.0),
        ("cooldown_bars", 7),
        ("max_distance_from_poc", 0.15),
    )

    def __init__(self):
        self.ema50 = bt.ind.EMA(self.data.close, period=self.params.ema_fast)
        self.ema200 = bt.ind.EMA(self.data.close, period=self.params.ema_slow)
        self.vol_ma = bt.ind.SMA(self.data.volume, period=self.params.volume_period)
        self.atr = bt.ind.ATR(self.data, period=self.params.atr_period)

        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.highest_since_entry = None
        self.cooldown = 0

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.entry_price = order.executed.price
                self.highest_since_entry = order.executed.price
                self.stop_price = self.entry_price - self.atr[0] * self.params.atr_mult

                print(
                    f"BUY | Price: {self.entry_price:.2f} | "
                    f"POC: {self.data.poc[0]:.2f} | "
                    f"VAH: {self.data.vah[0]:.2f} | "
                    f"VAL: {self.data.val[0]:.2f} | "
                    f"Profile: {int(self.data.profile_type[0])} | "
                    f"SL: {self.stop_price:.2f}"
                )

            elif order.issell():
                sell_price = order.executed.price
                print(f"SELL | Price: {sell_price:.2f}")

                if self.entry_price and sell_price < self.entry_price:
                    self.cooldown = self.params.cooldown_bars

                self.entry_price = None
                self.stop_price = None
                self.highest_since_entry = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print("ORDER CANCELED / MARGIN / REJECTED")

        self.order = None

    def next(self):
        if self.order:
            return

        min_bars = self.params.ema_slow + self.params.ema_slope_lookback
        if len(self) < min_bars:
            return

        close = self.data.close[0]
        open_ = self.data.open[0]
        high = self.data.high[0]
        low = self.data.low[0]
        volume = self.data.volume[0]

        poc = self.data.poc[0]
        vah = self.data.vah[0]
        val = self.data.val[0]
        profile_type = int(self.data.profile_type[0])

        if np.isnan(poc) or np.isnan(vah) or np.isnan(val):
            return

        if self.cooldown > 0:
            self.cooldown -= 1

        # Profile codes:
        # 1 = D balance
        # 2 = P rally
        # 3 = B selloff
        # 4 = Neutral

        green_candle = close > open_

        trend_up = close > self.ema200[0]
        ema_stack_up = close > self.ema50[0] > self.ema200[0]
        ema_slope_up = self.ema200[0] > self.ema200[-self.params.ema_slope_lookback]

        volume_expansion = volume > self.vol_ma[0] * self.params.volume_mult

        above_value_area = close > vah
        above_poc = close > poc

        not_d_profile = profile_type != 1
        not_b_profile = profile_type != 3

        not_too_far_from_poc = close < poc * (1 + self.params.max_distance_from_poc)

        # Fake breakout:
        # previous close was above VAH, now close falls back below VAH
        fake_breakout = self.data.close[-1] > self.data.vah[-1] and close < vah

        if not self.position:
            if (
                self.cooldown == 0
                and trend_up
                and ema_stack_up
                and ema_slope_up
                and green_candle
                and volume_expansion
                and above_value_area
                and above_poc
                and not_d_profile
                and not_b_profile
                and not_too_far_from_poc
                and not fake_breakout
            ):
                self.order = self.buy()

        else:
            # ATR trailing stop
            if high > self.highest_since_entry:
                self.highest_since_entry = high

            trailing_stop = self.highest_since_entry - self.atr[0] * self.params.atr_mult

            if trailing_stop > self.stop_price:
                self.stop_price = trailing_stop

            exit_by_trailing = low <= self.stop_price
            exit_back_inside_value = close < vah
            exit_below_poc = close < poc
            exit_below_ema200 = close < self.ema200[0]
            exit_fake_breakout = fake_breakout

            if (
                exit_by_trailing
                or exit_back_inside_value
                or exit_below_poc
                or exit_below_ema200
                or exit_fake_breakout
            ):
                self.order = self.sell()


# =========================================================
# 4) Main Backtest
# =========================================================

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

df = add_rolling_volume_profile(
    df,
    lookback=30,
    bins=24,
)

df.dropna(inplace=True)

start_price = df["Close"].iloc[0]
end_price = df["Close"].iloc[-1]
buy_hold_return = (end_price / start_price - 1) * 100
print(f"Buy & Hold Return: {buy_hold_return:.2f}%")

data = PandasVolumeProfileData(dataname=df)

cerebro = bt.Cerebro()
cerebro.addstrategy(VolumeProfileBPDStrategy)
cerebro.adddata(data)

cerebro.broker.setcash(10000)
cerebro.broker.setcommission(commission=0.001)
cerebro.broker.set_slippage_perc(perc=0.001)

# สำคัญ: test03 ลด risk ก่อน
cerebro.addsizer(bt.sizers.PercentSizer, percents=30)

cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

print("Starting Portfolio Value:", cerebro.broker.getvalue())

results = cerebro.run()
strategy = results[0]

final_value = cerebro.broker.getvalue()
strategy_return = (final_value / 10000 - 1) * 100

print(f"Final Portfolio Value: {final_value}")
print(f"Strategy Return: {strategy_return:.2f}%")

print("\n--- Trade Analysis ---")
trades = strategy.analyzers.trades.get_analysis()
print(trades)

print("\n--- Drawdown ---")
drawdown = strategy.analyzers.drawdown.get_analysis()
print(drawdown)

print("\n--- Sharpe Ratio ---")
print(strategy.analyzers.sharpe.get_analysis())

print("\n--- Returns ---")
print(strategy.analyzers.returns.get_analysis())

print("\n--- Simple Summary ---")
total_trades = trades.get("total", {}).get("closed", 0)
won_trades = trades.get("won", {}).get("total", 0)
lost_trades = trades.get("lost", {}).get("total", 0)

win_rate = (won_trades / total_trades * 100) if total_trades else 0
max_dd = drawdown.get("max", {}).get("drawdown", 0)

print(f"Total Trades: {total_trades}")
print(f"Won Trades: {won_trades}")
print(f"Lost Trades: {lost_trades}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Max Drawdown: {max_dd:.2f}%")
print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
print(f"Strategy Return: {strategy_return:.2f}%")

cerebro.plot()