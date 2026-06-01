"""
全模式 benchmark：七種執行路徑比較，皆使用 SmaCross(fast=10, slow=30)。

  執行路徑               說明                                      層級
  ─────────────────────────────────────────────────────────────────────
  event_driven          backtesting.py 原始事件迴圈               舊基準
                        策略在 next() 內即時計算 SMA、下單

  vectorized_signals    backtesting.py 混合模式                   正確性基準
                        訊號預算（Layer 1）+ 事件執行（Layer 2）

  broker                MultiBT 委派 vectorized_signals           =上行

  portfolio/fast        MultiBT Numba @njit kernel（快取後）      本專案主打
  portfolio/python      MultiBT Python UnifiedPortfolio           完整 audit log
  portfolio/next_open   Numba kernel + next_open 成交價           接近 broker
  rebalance             MultiBT 每日等權再平衡                    組合語意

用法:
    python benchmark_vectorize.py --stocks 20 --days 80
    python benchmark_vectorize.py --stocks 100 --days 500

首次執行時，Numba JIT 暖機 1-3s；第二次起快取生效。
event_driven 速度最慢（O(T²) 指標計算），資料量大時請有耐心。
"""

import argparse
import sys
import time
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backtesting.py"))
sys.path.insert(0, str(ROOT))

from backtesting import Backtest, Strategy
from multibt import MultiBT
from multibt.core.numba_kernels import NUMBA_AVAILABLE
from multibt.vectorized import VectorizedIndicators

FAST, SLOW = 10, 30
CASH = 1_000_000
COMMISSION = 0.001


# ──────────────────────────────────────────────────────────────────────────────
# 策略定義
# ──────────────────────────────────────────────────────────────────────────────


class EventDrivenSmaCross(Strategy):
    """
    event_driven 模式策略：在 next() 內即時計算 SMA，觸發買賣。

    特性：
    - _batch_at(t) 只給當天的一列，因此策略自己維護 _history 緩衝區，
      每次 next() 將當日收盤價 append 後重新算 rolling SMA。
    - 整體複雜度 O(T × W × N)，W = slow 視窗大小，N = 股票數。
    - 這就是 vectorized_signals（Layer 1 向量化預計算）要消除的瓶頸。
    """

    fast = FAST
    slow = SLOW

    def init(self):
        self._history: dict[str, list] = {}   # stock_id -> list of Close prices
        self._holding: dict[str, bool] = {}   # stock_id -> True if long

    def next(self, batch=None):
        if batch is None or (hasattr(batch, "empty") and batch.empty):
            return
        for stock_id in batch["stock_id"].unique():
            row = batch[batch["stock_id"] == stock_id]
            if row.empty:
                continue
            close = float(row["Close"].iloc[-1])
            buf = self._history.setdefault(stock_id, [])
            buf.append(close)

            # 至少需要 slow+1 個資料點才能偵測交叉
            if len(buf) < self.slow + 1:
                continue

            # 用最近 slow+1 筆計算目前與前一步的 SMA
            window = buf[-(self.slow + 1):]
            fast_sma  = np.mean(window[-(self.fast):])
            slow_sma  = np.mean(window[-(self.slow):])
            prev_fast = np.mean(window[-(self.fast + 1):-1])
            prev_slow = np.mean(window[-(self.slow + 1):-1])

            is_holding = self._holding.get(stock_id, False)
            golden = (not is_holding
                      and fast_sma > slow_sma
                      and prev_fast <= prev_slow)
            death  = (is_holding
                      and fast_sma < slow_sma
                      and prev_fast >= prev_slow)

            if golden:
                self.buy(stock=stock_id)
                self._holding[stock_id] = True
            elif death:
                for trade in list(self.trades):
                    if trade.stock == stock_id and trade.is_long:
                        trade.close(stock_id)
                self._holding[stock_id] = False


class VectorizedMultiSmaCross(Strategy):
    """
    vectorized_signals / broker 模式佔位策略。
    訊號由引擎在迴圈外預計算，next() 不做任何事。
    """

    fast = FAST
    slow = SLOW

    def init(self):
        pass

    def next(self, batch=None):
        pass


# ──────────────────────────────────────────────────────────────────────────────
# 資料產生
# ──────────────────────────────────────────────────────────────────────────────


def make_multi_asset_data(n_stocks: int, n_days: int, seed: int = 42) -> dd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rows = []
    for i in range(n_stocks):
        stock_id = f"STK{i:03d}"
        prices = 100.0 + np.cumsum(rng.normal(0, 1, n_days))
        prices = np.maximum(prices, 1.0)
        for d, p in zip(dates, prices):
            rows.append({
                "date": d, "stock_id": stock_id,
                "Open": p - 0.1, "High": p * 1.01,
                "Low": p * 0.99, "Close": p, "Volume": 1000,
            })
    return dd.from_pandas(pd.DataFrame(rows), npartitions=4)


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark
# ──────────────────────────────────────────────────────────────────────────────


def _run(fn) -> tuple[float, object]:
    t0 = time.perf_counter()
    result = fn()
    return time.perf_counter() - t0, result


def _row(label: str, elapsed: float, trades: int, ret: float,
         note: str, base_t: float) -> str:
    xfactor = f"{base_t / elapsed:6.1f}x" if elapsed > 0 else "  ---"
    label_col = f"[{label}]"
    return (
        f"{label_col:<28} {elapsed:7.3f}s  {xfactor}  "
        f"trades={trades:4d}  return={ret:+8.2f}%  {note}"
    )


def benchmark(n_stocks: int, n_days: int, skip_event_driven: bool = False):
    sep = "=" * 80
    print(sep)
    print(
        f"  MultiBT 全模式 benchmark | SmaCross({FAST},{SLOW}) | "
        f"{n_stocks} 股 x {n_days} 天"
    )
    print(f"  Numba: {NUMBA_AVAILABLE}")
    print(sep)

    data_dask = make_multi_asset_data(n_stocks, n_days)
    panel = data_dask.compute()

    timings: dict[str, float] = {}
    trd: dict[str, int] = {}
    ret: dict[str, float] = {}

    header = (
        f"\n{'路徑':<28} {'時間':>7}   {'加速':>7}  "
        f"{'trades':>10}  {'return':>12}  說明"
    )
    print(header)
    print("-" * 80)

    # ── event_driven（最慢，原始基準）────────────────────────────────────────
    if not skip_event_driven:
        bt_ev = Backtest(data_dask, EventDrivenSmaCross,
                         cash=CASH, commission=COMMISSION)
        t, res = _run(lambda: bt_ev.run(mode="event_driven"))
        timings["event_driven"] = t
        trd["event_driven"] = int(res.get("# Trades", 0))
        ret["event_driven"] = float(res.get("Return [%]", 0))
        print(_row("event_driven", t, trd["event_driven"], ret["event_driven"],
                   "原始事件迴圈（O(T^2 x N) 指標計算）",
                   t))  # 以自身為基準先印
    else:
        print(f"{'[event_driven]':<28} {'SKIP':>7}   {'---':>7}  "
              f"{'---':>10}  {'---':>12}  (--skip-event-driven)")

    # ── vectorized_signals（正確性基準）──────────────────────────────────────
    bt_vec = Backtest(data_dask, VectorizedMultiSmaCross,
                      cash=CASH, commission=COMMISSION)
    t, res = _run(lambda: bt_vec.run(mode="vectorized_signals",
                                     fast=FAST, slow=SLOW))
    timings["vectorized_signals"] = t
    trd["vectorized_signals"] = int(res["# Trades"])
    ret["vectorized_signals"] = float(res["Return [%]"])

    # ── broker（MultiBT 委派 vectorized_signals）──────────────────────────────
    t, res = _run(lambda: MultiBT(
        panel, cash=CASH, commission=COMMISSION,
        fast_param=FAST, slow_param=SLOW,
    ).run(mode="broker"))
    timings["broker"] = t
    trd["broker"] = int(res["metrics"].get("# Trades", 0))
    ret["broker"] = float(res["metrics"].get("Return [%]", 0))

    # ── portfolio / Numba fast（首次，含 JIT 暖機）───────────────────────────
    t, res = _run(lambda: MultiBT(
        panel, cash=CASH, commission=COMMISSION,
        fast_param=FAST, slow_param=SLOW, fast=True,
    ).run(mode="portfolio", trade_on="close"))
    timings["portfolio_fast_warm"] = t
    m = res["metrics"]
    trd["portfolio_fast_warm"] = int(m.get("n_trades", 0))
    ret["portfolio_fast_warm"] = float(m.get("return_pct", 0))

    # ── portfolio / Numba fast（第二次，JIT 快取後）──────────────────────────
    t, res = _run(lambda: MultiBT(
        panel, cash=CASH, commission=COMMISSION,
        fast_param=FAST, slow_param=SLOW, fast=True,
    ).run(mode="portfolio", trade_on="close"))
    timings["portfolio_fast"] = t
    m = res["metrics"]
    trd["portfolio_fast"] = int(m.get("n_trades", 0))
    ret["portfolio_fast"] = float(m.get("return_pct", 0))

    # ── portfolio / Python slow（完整 audit log）──────────────────────────────
    t, res = _run(lambda: MultiBT(
        panel, cash=CASH, commission=COMMISSION,
        fast_param=FAST, slow_param=SLOW, fast=False,
    ).run(mode="portfolio", trade_on="close"))
    timings["portfolio_python"] = t
    m = res["metrics"]
    trd["portfolio_python"] = int(m.get("n_trades", 0))
    ret["portfolio_python"] = float(m.get("return_pct", 0))

    # ── portfolio / next_open（較接近 broker 成交價）─────────────────────────
    t, res = _run(lambda: MultiBT(
        panel, cash=CASH, commission=COMMISSION,
        fast_param=FAST, slow_param=SLOW, fast=True,
    ).run(mode="portfolio", trade_on="next_open"))
    timings["portfolio_next_open"] = t
    m = res["metrics"]
    trd["portfolio_next_open"] = int(m.get("n_trades", 0))
    ret["portfolio_next_open"] = float(m.get("return_pct", 0))

    # ── rebalance（不同執行語意）──────────────────────────────────────────────
    t, res = _run(lambda: MultiBT(
        panel, cash=CASH, commission=COMMISSION,
        fast_param=FAST, slow_param=SLOW, fast=True,
    ).run(mode="rebalance"))
    timings["rebalance"] = t
    m = res["metrics"]
    trd["rebalance"] = int(m.get("n_trades", 0))
    ret["rebalance"] = float(m.get("return_pct", 0))

    # ── 輸出 ──────────────────────────────────────────────────────────────────
    base = timings["vectorized_signals"]

    rows = [
        ("event_driven",        "原始事件迴圈，O(T^2*N) 指標計算",          True),
        ("vectorized_signals",  "Layer1 向量化 + Layer2 事件執行 [正確性基準]", False),
        ("broker",              "MultiBT 委派 vectorized_signals",           False),
        ("portfolio_fast_warm", f"Numba @njit {'(含 JIT 暖機)' if NUMBA_AVAILABLE else '(無Numba)'}",  False),
        ("portfolio_fast",      "Numba @njit（JIT 快取後）",                  False),
        ("portfolio_python",    "Python UnifiedPortfolio（完整 audit log）", False),
        ("portfolio_next_open", "Numba + next_open 成交價（接近 broker）",   False),
        ("rebalance",           "每日等權再平衡（不同執行語意）",             False),
    ]

    for name, note, is_slow in rows:
        if name == "event_driven" and skip_event_driven:
            continue
        if name not in timings:
            continue
        elapsed = timings[name]
        print(_row(name, elapsed, trd[name], ret[name], note, base))

    # ── 一致性比較 ─────────────────────────────────────────────────────────────
    print()
    print(sep)
    print("  一致性比較（vectorized_signals 為報酬基準）")
    print(f"  {'路徑':<26}  {'trades':>8}  {'return':>9}  {'diff':>9}  說明")
    print("-" * 80)
    ref_ret = ret["vectorized_signals"]
    ref_trd = trd["vectorized_signals"]

    consistency_rows = [
        ("vectorized_signals",  "[基準]"),
        ("broker",              "應與基準相同"),
        ("portfolio_fast",      "成交語意不同，return 差距是預期行為"),
        ("portfolio_python",    "同上"),
        ("portfolio_next_open", "成交價較接近 broker，但 sizing/FIFO 仍不同"),
        ("rebalance",           "不同執行語意，不作報酬比較"),
    ]

    for name, note in consistency_rows:
        if name not in trd:
            continue
        diff = ret[name] - ref_ret
        match = "[OK]" if abs(diff) < 0.1 else "    "
        print(
            f"  {name:<26}  {trd[name]:>8}  {ret[name]:>+8.2f}%  {diff:>+8.2f}pp  {match} {note}"
        )

    print()
    print("  event_driven: O(T^2) 指標計算，與 vectorized_signals 的差距是 Layer1 優化的意義")
    print("  portfolio vs broker: 撮合語意不同，非 bug（見 doc/portfolio_execution.md）")
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MultiBT 全模式 benchmark（SmaCross 策略）"
    )
    parser.add_argument("--stocks", type=int, default=20,
                        help="股票數量（預設 20）")
    parser.add_argument("--days", type=int, default=80,
                        help="交易天數（預設 80）")
    parser.add_argument("--skip-event-driven", action="store_true",
                        help="跳過最慢的 event_driven（大資料量時使用）")
    args = parser.parse_args()
    benchmark(args.stocks, args.days, skip_event_driven=args.skip_event_driven)
