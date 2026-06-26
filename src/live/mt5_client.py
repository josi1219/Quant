import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MT5Client:
    def __init__(self, symbol="EURUSD"):
        self.symbol = symbol

    def connect(self) -> bool:
        """Connect to the local MT5 terminal."""
        if not mt5.initialize():
            logger.error("initialize() failed, error code = %s", mt5.last_error())
            return False
            
        # Ensure symbol is available
        if_visible = mt5.symbol_select(self.symbol, True)
        if not if_visible:
            logger.error("symbol_select(%s, True) failed", self.symbol)
            return False
            
        logger.info("Connected to MT5 terminal successfully.")
        return True

    def disconnect(self):
        """Disconnect from MT5."""
        mt5.shutdown()

    def get_historical_data(self, timeframe, count: int) -> pd.DataFrame:
        """Fetch historical data and format to standard pandas DataFrame."""
        if count <= 5000:
            rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                logger.error("Failed to copy rates for %s. Error: %s", self.symbol, mt5.last_error())
                return pd.DataFrame()
        else:
            # Fetch in chunks of 5000 to prevent MT5 IPC timeouts on large broker downloads
            all_rates = []
            remaining = count
            current_pos = 0
            while remaining > 0:
                chunk_size = min(5000, remaining)
                logger.info("Fetching chunk of %d bars starting from position %d (timeframe: %s)...", chunk_size, current_pos, timeframe)
                chunk = mt5.copy_rates_from_pos(self.symbol, timeframe, current_pos, chunk_size)
                if chunk is None or len(chunk) == 0:
                    logger.error("Failed to copy rates chunk for %s at pos %d. Error: %s", self.symbol, current_pos, mt5.last_error())
                    break
                # chunk contains [oldest...newest] for that specific window.
                # Since we are iterating back in time (current_pos increases),
                # we prepend the older chunks to maintain overall chronological order.
                all_rates.insert(0, chunk)
                current_pos += chunk_size
                remaining -= chunk_size
                
            if not all_rates:
                return pd.DataFrame()
            
            rates = np.concatenate(all_rates)

        df = pd.DataFrame(rates)
        
        # Format columns to match our CSV format exactly
        # MT5 native columns: time, open, high, low, close, tick_volume, spread, real_volume
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Keep only what we need for pipeline
        df = df[['open', 'high', 'low', 'close', 'tick_volume', 'spread']]
        
        return df

    def get_bot_positions(self, magic: int = 1219):
        """Return a list of open positions for the symbol opened by this bot."""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []
        return [p for p in positions if p.magic == magic]

    def get_open_positions(self, magic: int = 1219) -> int:
        """Return the number of open positions for the symbol by this bot."""
        return len(self.get_bot_positions(magic))

    def get_tick(self):
        """Get the current live tick for the symbol."""
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            logger.error("Failed to get tick for %s", self.symbol)
        return tick

    def execute_trade(self, signal: int, volume: float, pt_price: float, sl_price: float, tick=None, magic: int = 1219):
        """
        Execute a market order.
        signal: 1 for Buy, -1 for Sell
        volume: lot size
        pt_price: Take Profit price level
        sl_price: Stop Loss price level
        magic: Expert Advisor magic number
        """
        if tick is None:
            tick = self.get_tick()
            if tick is None:
                return False

        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logger.error("Symbol info not found for %s", self.symbol)
            return False
            
        filling_mode = symbol_info.filling_mode
        if filling_mode & 1:
            fill_type = mt5.ORDER_FILLING_FOK
        elif filling_mode & 2:
            fill_type = mt5.ORDER_FILLING_IOC
        else:
            fill_type = mt5.ORDER_FILLING_RETURN

        action_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
        price = tick.ask if signal == 1 else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": action_type,
            "price": price,
            "sl": sl_price,
            "tp": pt_price,
            "deviation": 20, # Max deviation in points
            "magic": magic,
            "comment": "Antigravity MTF Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": fill_type,
        }

        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            logger.error("No response from MT5 terminal (order_send returned None).")
            return False
            
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("Order failed! Retcode: %s, Error: %s", result.retcode, result.comment)
            return False
            
        logger.info("Order executed successfully! Deal ticket: %s", result.deal)
        return True

    def close_position(self, ticket: int, pos_type: int, volume: float, magic: int = 1219) -> bool:
        """Close an existing open position."""
        tick = self.get_tick()
        if tick is None:
            return False

        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return False
            
        filling_mode = symbol_info.filling_mode
        if filling_mode & 1:
            fill_type = mt5.ORDER_FILLING_FOK
        elif filling_mode & 2:
            fill_type = mt5.ORDER_FILLING_IOC
        else:
            fill_type = mt5.ORDER_FILLING_RETURN

        action_type = mt5.ORDER_TYPE_SELL if pos_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if action_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": action_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": magic,
            "comment": "Close - Vertical Barrier",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": fill_type,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("Failed to close position %s. Retcode: %s", ticket, getattr(result, 'retcode', 'None'))
            return False
            
        logger.info("Successfully closed position %s due to vertical barrier!", ticket)
        return True
