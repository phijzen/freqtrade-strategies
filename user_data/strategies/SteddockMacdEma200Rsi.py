from __future__ import annotations

from typing import Dict, Final

import pandas as pd
import talib.abstract as ta

from freqtrade.strategy import IStrategy
from freqtrade.strategy import IntParameter


_RSI_ENTRY_LOW: Final[int] = 40
_RSI_ENTRY_HIGH: Final[int] = 65
_RSI_EXIT: Final[int] = 40

def _create_int_parameter(default: int, space: str, upper_bound: int = 5, lower_bound: int = -5) -> IntParameter:
    return IntParameter(default+lower_bound, default+upper_bound, default=default, space=space)


class SteddockMacdEma200Rsi(IStrategy):
    """
    BTC/USDC 1h trend-following strategy.

    Entry:
    - Price above EMA200
    - EMA200 is rising
    - Bullish MACD crossover
    - RSI between 40 and 65

    Exit:
    - Bearish MACD crossover
    - RSI crosses down through 40
    - Price closes below EMA200

    ROI is effectively disabled so profitable trends
    can continue running until an exit condition occurs.
    """

    # --- Core configuration ---

    timeframe: str = "1h"
    can_short: bool = False

    # Effectively disable ROI-based profit taking.
    # This allows strong trends to continue beyond +5%.
    minimal_roi: Dict[str, float] = {
        "0": 100.0,
    }

    # Maximum loss / emergency protection.
    stoploss: float = -0.10

    startup_candle_count: int = 210
    process_only_new_candles: bool = True

    use_exit_signal: bool = True
    exit_profit_only: bool = False
    ignore_roi_if_entry_signal: bool = False

    # --- Indicator parameters ---

    ema_length: int = 200

    # RSI entry band:
    # Only enter when RSI is between 40 and 65.
    rsi_entry_low: IntParameter = _create_int_parameter(_RSI_ENTRY_LOW, space="buy")
    rsi_entry_high: IntParameter = _create_int_parameter(_RSI_ENTRY_HIGH, space="buy")
    rsi_exit: IntParameter = _create_int_parameter(_RSI_EXIT, space="sell")

    # RSI exit:
    # Exit when RSI crosses DOWN through 40.
    rsi_exit: IntParameter = IntParameter(
        RSI_EXIT-5, RSI_EXIT+5, default=RSI_EXIT, space="sell"
    )

    def populate_indicators(
        self,
        dataframe: pd.DataFrame,
        metadata: Dict,
    ) -> pd.DataFrame:

        # --- EMA200 ---

        dataframe["ema200"] = ta.EMA(
            dataframe,
            timeperiod=self.ema_length,
        )

        dataframe["ema200_prev"] = dataframe["ema200"].shift(1)

        # --- MACD ---

        macd = ta.MACD(dataframe)

        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        dataframe["macd_prev"] = dataframe["macd"].shift(1)
        dataframe["macdsignal_prev"] = dataframe["macdsignal"].shift(1)

        # --- RSI ---

        dataframe["rsi"] = ta.RSI(
            dataframe,
            timeperiod=14,
        )

        dataframe["rsi_prev"] = dataframe["rsi"].shift(1)

        return dataframe

    def populate_entry_trend(
        self,
        dataframe: pd.DataFrame,
        metadata: Dict,
    ) -> pd.DataFrame:

        dataframe.loc[
            (
                # --------------------------------------------------
                # 1. Trend regime
                # --------------------------------------------------
                (dataframe["close"] > dataframe["ema200"])
                & (
                    dataframe["ema200"]
                    > dataframe["ema200_prev"]
                )

                # --------------------------------------------------
                # 2. Bullish MACD crossover
                # --------------------------------------------------
                & (
                    dataframe["macd"]
                    > dataframe["macdsignal"]
                )
                & (
                    dataframe["macd_prev"]
                    <= dataframe["macdsignal_prev"]
                )

                # --------------------------------------------------
                # 3. RSI between 40 and 65
                # --------------------------------------------------
                & (
                    dataframe["rsi"]
                    >= self.rsi_entry_low.value
                )
                & (
                    dataframe["rsi"]
                    <= self.rsi_entry_high.value
                )
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(
        self,
        dataframe: pd.DataFrame,
        metadata: Dict,
    ) -> pd.DataFrame:

        dataframe.loc[
            (
                # --------------------------------------------------
                # 1. Bearish MACD crossover
                # --------------------------------------------------
                (
                    (dataframe["macd"] < dataframe["macdsignal"])
                    & (
                        dataframe["macd_prev"]
                        >= dataframe["macdsignal_prev"]
                    )
                )

                # --------------------------------------------------
                # 2. RSI crosses DOWN through 40
                # --------------------------------------------------
                | (
                    (dataframe["rsi"] < self.rsi_exit.value)
                    & (
                        dataframe["rsi_prev"]
                        >= self.rsi_exit.value
                    )
                )

                # --------------------------------------------------
                # 3. Price closes below EMA200
                # --------------------------------------------------
                | (
                    dataframe["close"]
                    < dataframe["ema200"]
                )
            ),
            "exit_long",
        ] = 1

        return dataframe

