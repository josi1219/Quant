import logging
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import MetaTrader5 as mt5

from src.config import cfg
from src.data.loader import _clean
from src.features.pipeline import build_features_mtf
from src.labels.triple_barrier import _estimate_daily_volatility, _compute_vol_regime_multiplier
from src.live.mt5_client import MT5Client

logger = logging.getLogger(__name__)

class LiveExecutor:
    def __init__(self, model_dir="models", symbol="EURUSD"):
        self.model_dir = Path(model_dir)
        self.symbol = symbol
        self.client = MT5Client(symbol=symbol)
        
        self.model = None
        self.feature_names = []
        self.frac_d = 0.15
        self.last_trade_time = None
        
        # Raw Data Caches
        self.cached_m5 = None
        self.cached_h1 = None
        
        self._load_model()
        
    def _load_model(self):
        """Load the trained LightGBM model and feature specifications."""
        lgbm_path = self.model_dir / "eurusd_model.lgbm"
        spec_path = self.model_dir / "eurusd_model_feature_spec.json"
        
        # We need the full scikit-learn wrapper if it was saved as pickle, 
        # but let's load native booster for pure prediction if possible, 
        # or load the pickle.
        pkl_path = self.model_dir / "eurusd_model.pkl"
        
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info("Loaded model from %s", pkl_path)
        else:
            self.model = lgb.Booster(model_file=str(lgbm_path))
            logger.info("Loaded booster from %s", lgbm_path)
            
        with open(spec_path, "r") as f:
            spec = json.load(f)
            self.feature_names = spec["feature_names"]
            self.frac_d = spec.get("pipeline_metadata", {}).get("frac_d", 0.15)
            
        logger.info("Loaded feature specifications (%d features, frac_d=%.3f)", len(self.feature_names), self.frac_d)

    def connect(self) -> bool:
        """Connect to the MT5 client."""
        return self.client.connect()

    def disconnect(self):
        self.client.disconnect()

    def check_connection(self) -> bool:
        """Check MT5 terminal connection and attempt reconnect if lost."""
        if not mt5.terminal_info():
            logger.warning("MT5 terminal connection lost. Attempting to reconnect...")
            return self.connect()
        return True

    def manage_open_trades(self):
        """Enforce vertical barrier by closing trades that exceed max holding period."""
        bot_positions = self.client.get_bot_positions(magic=1219)
        if not bot_positions:
            return

        max_minutes = cfg.labels.max_holding_period * 5
        current_time = datetime.now()

        for pos in bot_positions:
            pos_time = datetime.fromtimestamp(pos.time)
            elapsed_minutes = (current_time - pos_time).total_seconds() / 60.0
            
            if elapsed_minutes > max_minutes:
                logger.warning("Trade %s open for %.1f mins (Max: %d). ENFORCING VERTICAL BARRIER.", pos.ticket, elapsed_minutes, max_minutes)
                self.client.close_position(ticket=pos.ticket, pos_type=pos.type, volume=pos.volume, magic=1219)

    def execute_iteration(self):
        """Main loop iteration to be called at the close of every M5 candle."""
        logger.info("="*50)
        logger.info("Running execution iteration...")
        
        # Enforce vertical barrier before looking for new setups
        self.manage_open_trades()
        
        # 1. Fetch Data (Incremental Caching)
        if self.cached_m5 is None or self.cached_h1 is None:
            # First run: Fetch large history to warm up the cache
            logger.info("Initializing raw data cache... This might take a few minutes to download from the broker!")
            
            # Progressive fallback to handle MT5 broker pagination limits
            for attempt_count in [25000, 10000, 5000]:
                logger.info(f"Attempting to fetch {attempt_count} M5 bars...")
                new_m5 = self.client.get_historical_data(mt5.TIMEFRAME_M5, count=attempt_count)
                new_h1 = self.client.get_historical_data(mt5.TIMEFRAME_H1, count=attempt_count // 10)
                
                if not new_m5.empty and not new_h1.empty:
                    logger.info(f"Successfully fetched {attempt_count} bars for cache initialization!")
                    break
            
            self.cached_m5 = new_m5 if not new_m5.empty else None
            self.cached_h1 = new_h1 if not new_h1.empty else None
        else:
            # Subsequent runs: Fetch only recent candles and append to cache
            logger.info("Updating raw data cache (Fetching 10 M5 bars, 2 H1 bars)...")
            new_m5 = self.client.get_historical_data(mt5.TIMEFRAME_M5, count=10)
            new_h1 = self.client.get_historical_data(mt5.TIMEFRAME_H1, count=2)
            
            # Combine and drop duplicates based on the timestamp index
            self.cached_m5 = pd.concat([self.cached_m5, new_m5]).loc[~pd.concat([self.cached_m5, new_m5]).index.duplicated(keep='last')]
            self.cached_h1 = pd.concat([self.cached_h1, new_h1]).loc[~pd.concat([self.cached_h1, new_h1]).index.duplicated(keep='last')]
            
            # Truncate to maintain stable memory usage
            self.cached_m5 = self.cached_m5.iloc[-25000:]
            self.cached_h1 = self.cached_h1.iloc[-2500:]
        
        if self.cached_m5 is None or self.cached_m5.empty or self.cached_h1 is None or self.cached_h1.empty:
            logger.error("Failed to fetch historical data. Skipping iteration.")
            return

        # Working copies for this iteration
        df_m5 = self.cached_m5.copy()
        df_h1 = self.cached_h1.copy()

        # Drop the current unclosed M5 candle so we only build volume bars from closed candles.
        # CRITICAL: We do NOT drop the unclosed H1 candle. We let multi_timeframe.py's shift(1)
        # handle it, which ensures the previously completed H1 candle correctly aligns with
        # the current M5 timestamp, avoiding a 1-hour lag in features.
        df_m5 = df_m5.iloc[:-1]
        
        # Clean data identically to training to prevent spread hallucinations
        df_m5 = _clean(df_m5, cfg.data)
        df_h1 = _clean(df_h1, cfg.data)
        
        if df_m5.empty or df_h1.empty:
            logger.error("Data empty after cleaning. Skipping iteration.")
            return
        
        current_time = df_m5.index[-1]
        current_price = df_m5['close'].iloc[-1]
        logger.info("Latest closed M5 candle: %s, Close: %.5f", current_time, current_price)

        # 2. Build Features
        try:
            X, _ = build_features_mtf(df_m5, df_h1, frac_d=self.frac_d)
        except Exception as e:
            logger.error("Feature pipeline failed: %s", e)
            return
            
        if X.empty:
            logger.error("Feature matrix is empty after warmup drops. Need more historical data.")
            return

        # 3. Generate Prediction
        # Extract the very last row (most recent closed candle)
        last_row = X.iloc[[-1]]

        # Validate feature columns match the saved model spec before slicing.
        # A mismatch means the live pipeline diverged from training (pipeline code change,
        # NaN column drop, or feature rename). Fail loudly rather than silently.
        missing_cols = [f for f in self.feature_names if f not in last_row.columns]
        if missing_cols:
            logger.error(
                "FEATURE MISMATCH: %d features expected by model are missing from live "
                "pipeline output: %s",
                len(missing_cols),
                missing_cols[:10],
            )
            return
        extra_cols = [c for c in last_row.columns if c not in self.feature_names]
        if extra_cols:
            logger.warning(
                "Live pipeline produced %d extra features not in model spec (ignored): %s",
                len(extra_cols),
                extra_cols[:5],
            )

        # Ensure column order matches exactly
        last_row = last_row[self.feature_names]
        
        # Synchronization: Prevent trading on stale volume bars
        last_bar_time = last_row.index[-1]
        if not hasattr(self, "last_processed_bar_time"):
            self.last_processed_bar_time = last_bar_time
            logger.info("Initialized last volume bar time: %s. Waiting for a new volume bar to complete...", last_bar_time)
            return

        if last_bar_time <= self.last_processed_bar_time:
            logger.info("Volume bar %s already processed (No new volume bar completed yet). Waiting...", last_bar_time)
            return

        self.last_processed_bar_time = last_bar_time
        logger.info("NEW VOLUME BAR COMPLETED! (Started at %s). Generating prediction...", last_bar_time)
        
        # lgb.Booster.predict() returns shape (N, num_class) for multiclass problems directly.
        # The legacy pkl path (LGBMClassifier) uses predict_proba. Both produce the same layout.
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(last_row)[0]
        else:
            probs = self.model.predict(last_row)[0]  # shape (3,) for a single row
            
        # Classes: 0 -> Sell (-1), 1 -> Hold (0), 2 -> Buy (1)
        prob_sell = probs[0]
        prob_hold = probs[1]
        prob_buy = probs[2]
        
        logger.info("Probabilities -> Sell: %.3f | Hold: %.3f | Buy: %.3f", prob_sell, prob_hold, prob_buy)

        # 4. Meta-Labeler Logic (Confidence Threshold)
        threshold = cfg.meta_label.confidence_threshold
        signal = 0
        
        # Argmax first to get the model's actual prediction
        pred_class = np.argmax(probs)
        
        if pred_class == 2 and prob_buy > threshold:
            signal = 1
            confidence = prob_buy
        elif pred_class == 0 and prob_sell > threshold:
            signal = -1
            confidence = prob_sell
        else:
            logger.info("No signal generated (pred_class=%d, prob_sell=%.3f, prob_buy=%.3f)", pred_class, prob_sell, prob_buy)
            return

        # 5. Check Trade Restrictions
        open_positions = self.client.get_open_positions(magic=1219)
        logger.info("Current open positions: %d / %d", open_positions, cfg.training.max_open_trades)
        
        if open_positions >= cfg.training.max_open_trades:
            logger.warning("Max open trades reached. Vetoing new %s signal.", "BUY" if signal == 1 else "SELL")
            return
            
        # Cooldown check
        if self.last_trade_time is not None:
            time_since_last_trade = (datetime.now() - self.last_trade_time).total_seconds() / 60
            cooldown_minutes = cfg.training.cooldown_bars * 5
            if time_since_last_trade < cooldown_minutes:
                logger.warning("Trade cooldown active (%.1f / %d mins). Vetoing signal.", time_since_last_trade, cooldown_minutes)
                return

        # 6. Calculate Dynamic Take Profit & Stop Loss
        # Use volume-bar close prices (same series the model was trained on) for
        # volatility estimation so that TP/SL distances match training calibration.
        # X.index holds volume-bar end timestamps which are valid M5 timestamps.
        vb_close = df_m5['close'].reindex(X.index).dropna()
        if len(vb_close) < 30:
            logger.error("Too few volume-bar close prices (%d) to estimate volatility.", len(vb_close))
            return
        daily_vol = _estimate_daily_volatility(vb_close, lookback=cfg.labels.vol_lookback).iloc[-1]

        if cfg.labels.dynamic_barriers:
            vm = _compute_vol_regime_multiplier(vb_close, cfg.labels.vol_regime_fast, cfg.labels.vol_regime_slow).iloc[-1]
        else:
            vm = 1.0
            
        pt_dist = cfg.labels.pt_multiplier * vm * daily_vol
        sl_dist = cfg.labels.sl_multiplier * vm * daily_vol
        
        # 7. Pre-execution Safety Checks (Spread) and Live Pricing
        tick = self.client.get_tick()
        if tick is None:
            logger.error("Failed to get tick for live pricing. Vetoing signal.")
            return
            
        pip_size = 0.01 if "JPY" in self.symbol else 0.0001
        current_spread = (tick.ask - tick.bid) / pip_size
        if current_spread > cfg.live.max_spread_pips:
            logger.warning("Spread %.1f pips > %.1f pips (max). Vetoing signal to avoid slippage.", current_spread, cfg.live.max_spread_pips)
            return

        if signal == 1: # Buy at Ask
            pt_price = tick.ask + pt_dist
            sl_price = tick.ask - sl_dist
            logger.info("🚀 EXECUTING BUY | Confidence: %.3f | PT: %.5f | SL: %.5f", confidence, pt_price, sl_price)
        else: # Sell at Bid
            pt_price = tick.bid - pt_dist
            sl_price = tick.bid + sl_dist
            logger.info("🚀 EXECUTING SELL | Confidence: %.3f | PT: %.5f | SL: %.5f", confidence, pt_price, sl_price)

        # Query the symbol's required price precision
        symbol_info = mt5.symbol_info(self.symbol)
        digits = symbol_info.digits if symbol_info is not None else 5

        # Round to correct decimals to prevent MT5 invalid price errors
        pt_price = round(pt_price, digits)
        sl_price = round(sl_price, digits)

        # 8. Send Order with Dynamic Position Sizing
        if confidence >= cfg.meta_label.full_position_threshold:
            lot_size = 0.1
        elif confidence >= cfg.meta_label.half_position_threshold:
            lot_size = 0.05
        elif confidence >= cfg.meta_label.quarter_position_threshold:
            lot_size = 0.02
        else:
            lot_size = 0.01  # Minimum base size
            
        success = self.client.execute_trade(signal=signal, volume=lot_size, pt_price=pt_price, sl_price=sl_price, tick=tick, magic=1219)
        if success:
            logger.info("Trade successfully dispatched!")
            self.last_trade_time = datetime.now()
        else:
            logger.error("Trade dispatch failed.")
