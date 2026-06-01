import numpy as np
import pandas as pd


class VectorizedAnalytics:
    """
    回測結束後，用 NumPy 一次性計算所有績效指標。
    """

    @staticmethod
    def equity_curve(portfolio_df: pd.DataFrame) -> pd.Series:
        return portfolio_df["total_value"]

    @staticmethod
    def sharpe_ratio(
        equity: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        returns = equity.pct_change().dropna().values
        excess = returns - risk_free_rate / periods_per_year
        if len(excess) == 0 or excess.std() == 0:
            return 0.0
        return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())

    @staticmethod
    def max_drawdown(equity: pd.Series) -> float:
        values = equity.values
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return float(drawdown.min())

    @staticmethod
    def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
        if len(equity) < 2 or equity.iloc[0] == 0:
            return 0.0
        n_years = len(equity) / periods_per_year
        return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1)

    @staticmethod
    def batch_sharpe(equity_matrix: np.ndarray, periods_per_year: int = 252) -> np.ndarray:
        """
        同時計算 M 組參數掃描結果的 Sharpe Ratio。
        equity_matrix shape: (T, M)
        """
        returns = np.diff(equity_matrix, axis=0) / equity_matrix[:-1]
        mean_r = returns.mean(axis=0)
        std_r = returns.std(axis=0)
        std_r = np.where(std_r == 0, 1e-10, std_r)
        return np.sqrt(periods_per_year) * mean_r / std_r
