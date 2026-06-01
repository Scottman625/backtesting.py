import numpy as np
import pandas as pd
from typing import Dict, Tuple


class VectorizedIndicators:
    """
    事件迴圈開始前，一次性計算所有股票的所有指標。
    輸出 (時間 × 股票數) 矩陣，供事件驅動層 O(1) 查詢。
    """

    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.assets = sorted(data.keys())
        closes = pd.DataFrame({a: data[a]["Close"] for a in self.assets})
        self.closes = closes.values
        self.index = closes.index
        self.T, self.N = self.closes.shape

    @classmethod
    def from_panel(cls, df: pd.DataFrame, date_col: str = "date", asset_col: str = "stock_id"):
        """從含 date / stock_id 的 panel DataFrame 建立實例。"""
        data: Dict[str, pd.DataFrame] = {}
        for asset in sorted(df[asset_col].unique()):
            asset_df = (
                df.loc[df[asset_col] == asset, ["Open", "High", "Low", "Close", "Volume"]]
                .assign(**{date_col: df.loc[df[asset_col] == asset, date_col].values})
                .set_index(date_col)
                .sort_index()
            )
            data[str(asset)] = asset_df
        return cls(data)

    def sma(self, window: int) -> np.ndarray:
        """一次計算所有股票的 SMA，回傳 (T × N) 矩陣。"""
        closes = pd.DataFrame(self.closes, index=self.index, columns=self.assets)
        return closes.rolling(window, min_periods=window).mean().values

    def rsi(self, period: int = 14) -> np.ndarray:
        """所有股票的 RSI，(T × N) 矩陣。"""
        delta = np.diff(self.closes, axis=0)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        avg_gain = np.full((self.T, self.N), np.nan)
        avg_loss = np.full((self.T, self.N), np.nan)

        if self.T <= period:
            return avg_gain

        avg_gain[period] = gain[:period].mean(axis=0)
        avg_loss[period] = loss[:period].mean(axis=0)

        for i in range(period + 1, self.T):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period

        rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
        return 100 - (100 / (1 + rs))

    def crossover_signals(
        self, fast_window: int, slow_window: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        雙均線交叉訊號，輸出 (T × N) Boolean 矩陣。
        entries[t, n] = True 代表第 t 天第 n 支股票產生買入訊號。
        """
        fast = self.sma(fast_window)
        slow = self.sma(slow_window)

        entries = (fast[1:] > slow[1:]) & (fast[:-1] <= slow[:-1])
        exits = (fast[1:] < slow[1:]) & (fast[:-1] >= slow[:-1])

        entries = np.vstack([np.zeros((1, self.N), dtype=bool), entries])
        exits = np.vstack([np.zeros((1, self.N), dtype=bool), exits])
        return entries, exits
