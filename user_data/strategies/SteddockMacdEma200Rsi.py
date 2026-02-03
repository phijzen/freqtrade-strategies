from __future__ import annotations

from typing import Dict, List

import pandas as pd
import talib.abstract as ta

from freqtrade.strategy import IStrategy
from freqtrade.strategy import IntParameter
from freqtrade.persistence import Trade


class SteddockMacdEma200Rsi(IStrategy):
    """
    Stable BTC/USDC trend-following strategy:
    - EMA200 defines market regime
    - MACD cross confirms momentum
    - RSI filters entries and exits
    """

    # --- Core configuration ---
    timeframe: str = "1h"
    can_short: bool = False

    minimal_roi: Dict[str, float] = {
        "0": 0.05,  # safety net, exits mostly indicator-driven
    }

    stoploss: float = -0.10

    startup_candle_count: int = 210

    process_only_new_candles: bool = True

    # Prevent immediate re-entries
    cooldown_period: int = 5

    use_exit_signal: bool = True
    exit_profit_only: bool = False
    ignore_roi_if_entry_signal: bool = False

    # --- Indicator parameters (typed & tweakable) ---
    ema_length: int = 200

    rsi_entry_low: IntParameter = IntParameter(45, 50, default=45, space="buy")
    rsi_entry_high: IntParameter = IntParameter(50, 55, default=55, space="buy")

    rsi_exit: int = 30

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: Dict
    ) -> pd.DataFrame:
        # EMA200
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=self.ema_length)
        dataframe["ema200_prev"] = dataframe["ema200"].shift(1)

        # MACD
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        dataframe["macd_prev"] = dataframe["macd"].shift(1)
        dataframe["macdsignal_prev"] = dataframe["macdsignal"].shift(1)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: Dict
    ) -> pd.DataFrame:
        """
        Enter long when:
        - Price above EMA200
        - EMA200 rising
        - MACD crosses above signal
        - RSI within controlled band
        """

        dataframe.loc[
            (
                # Regime filter
                (dataframe["close"] > dataframe["ema200"])
                & (dataframe["ema200"] > dataframe["ema200_prev"])

                # MACD bullish cross
                & (dataframe["macd"] > dataframe["macdsignal"])
                & (dataframe["macd_prev"] <= dataframe["macdsignal_prev"])

                # RSI timing filter
                & (dataframe["rsi"] >= self.rsi_entry_low.value)
                & (dataframe["rsi"] <= self.rsi_entry_high.value)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: Dict
    ) -> pd.DataFrame:
        """
        Exit long when ANY:
        - MACD crosses below signal
        - RSI < 30
        - Price closes below EMA200 (regime violation)
        """

        dataframe.loc[
            (
                # MACD bearish cross
                (
                    (dataframe["macd"] < dataframe["macdsignal"])
                    & (dataframe["macd_prev"] >= dataframe["macdsignal_prev"])
                )
                |
                # RSI breakdown
                (dataframe["rsi"] < self.rsi_exit)
                |
                # Regime violation
                (dataframe["close"] < dataframe["ema200"])
            ),
            "exit_long",
        ] = 1

        return dataframe

