"""
Nexus Shadow-Quant — Centralized Logger
=========================================
Captures everything to a structured log file:
- Predictions (direction, confidence, target prices)
- Trades (entry/exit, PnL, reason)
- Model performance (accuracy, training time)
- System events (startup, errors, timing)
- Quant engine outputs (regime, Hurst, OFI)

Usage:
    from nexus_logger import NexusLogger
    logger = NexusLogger()
    logger.log_prediction(prediction_dict)
    logger.log_trade(trade_dict)
"""

import os
import logging
import json
import time
from datetime import datetime
from functools import wraps

import config


class NexusLogger:
    """Centralized structured logger for the Nexus pipeline."""
    
    _instance = None  # Singleton
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        os.makedirs(config.LOG_DIR, exist_ok=True)
        
        # Main log file with timestamp rotation
        self.log_path = os.path.join(config.LOG_DIR, "nexus_session.log")
        self.events_path = os.path.join(config.LOG_DIR, "nexus_events.jsonl")
        
        # Setup file handler
        self.logger = logging.getLogger("nexus")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        # Clear old handlers
        self.logger.handlers.clear()
        
        # File handler — detailed
        fh = logging.FileHandler(self.log_path, mode='a', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(fh)
        
        # Console handler — info only
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(ch)
        
        # Session start marker
        self.session_start = datetime.now()
        self.logger.info("=" * 70)
        self.logger.info(f"NEXUS SESSION START — {self.session_start.isoformat()}")
        self.logger.info(f"Version: {config.VERSION}")
        self.logger.info("=" * 70)
        
        self._initialized = True
    
    # ========== STRUCTURED EVENT LOG ==========
    
    def _log_event(self, event_type: str, data: dict):
        """Append structured JSON event to events file."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "session": self.session_start.isoformat(),
            "type": event_type,
            **data
        }
        try:
            with open(self.events_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event, default=lambda x: float(x) if hasattr(x, 'item') else str(x)) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write event: {e}")
    
    # ========== PREDICTION LOGGING ==========
    
    def log_prediction(self, prediction: dict, duration_ms: float = 0):
        """Log an AI prediction with all details."""
        direction = prediction.get('direction', 'UNKNOWN')
        confidence = prediction.get('confidence', 0)
        target_1h = prediction.get('target_price_1h', prediction.get('target_price', 0))
        target_2h = prediction.get('target_price_2h', 0)
        hurst = prediction.get('hurst', 0)
        
        self.logger.info(
            f"PREDICTION | {direction} | Conf: {confidence:.1f}% | "
            f"1H: ${target_1h:,.2f} | 2H: ${target_2h:,.2f} | "
            f"Hurst: {hurst:.3f} | Time: {duration_ms:.0f}ms"
        )
        
        self._log_event("prediction", {
            "direction": direction,
            "confidence": confidence,
            "target_1h": target_1h,
            "target_2h": target_2h,
            "hurst": hurst,
            "duration_ms": duration_ms,
            "full_data": {k: v for k, v in prediction.items() 
                         if isinstance(v, (int, float, str, bool))}
        })
    
    # ========== TRADE LOGGING ==========
    
    def log_trade_open(self, direction: str, entry_price: float, 
                       size: float, leverage: int, confidence: float):
        """Log a new trade opening."""
        self.logger.info(
            f"TRADE OPEN | {direction} @ ${entry_price:,.2f} | "
            f"Size: ${size:,.2f} | Lev: {leverage}x | Conf: {confidence:.1f}%"
        )
        self._log_event("trade_open", {
            "direction": direction,
            "entry_price": entry_price,
            "size_usd": size,
            "leverage": leverage,
            "confidence": confidence
        })
    
    def log_trade_close(self, trade_record: dict):
        """Log a trade closing with full details."""
        pnl = trade_record.get('pnl_usd', 0)
        reason = trade_record.get('close_reason', 'UNKNOWN')
        direction = trade_record.get('direction', '?')
        
        emoji = "✅" if pnl > 0 else "❌"
        self.logger.info(
            f"{emoji} TRADE CLOSE | {direction} | Reason: {reason} | "
            f"PnL: ${pnl:+,.2f} ({trade_record.get('pnl_pct', 0):+.1f}%) | "
            f"Balance: ${trade_record.get('balance_after', 0):,.2f}"
        )
        self._log_event("trade_close", trade_record)
    
    # ========== TRAINING LOGGING ==========
    
    def log_training(self, duration_sec: float, accuracy: float, 
                     samples: int, features: int):
        """Log model training event."""
        self.logger.info(
            f"TRAINING | Duration: {duration_sec:.1f}s | "
            f"Accuracy: {accuracy:.1f}% | "
            f"Samples: {samples:,} | Features: {features}"
        )
        self._log_event("training", {
            "duration_sec": duration_sec,
            "accuracy": accuracy,
            "samples": samples,
            "features": features
        })
    
    # ========== QUANT ENGINE LOGGING ==========
    
    def log_quant_analysis(self, analysis: dict, duration_ms: float = 0):
        """Log quant engine output."""
        regime = analysis.get('regime', {}).get('regime', 'UNKNOWN')
        regime_conf = analysis.get('regime', {}).get('confidence', 0)
        
        self.logger.info(
            f"QUANT | Regime: {regime} ({regime_conf:.0f}%) | "
            f"Models: {len(analysis)} | Time: {duration_ms:.0f}ms"
        )
        
        # Log simplified version (avoid huge nested dicts)
        safe_data = {}
        for k, v in analysis.items():
            if isinstance(v, dict):
                safe_data[k] = {kk: vv for kk, vv in v.items() 
                               if isinstance(vv, (int, float, str, bool))}
            elif isinstance(v, (int, float, str, bool)):
                safe_data[k] = v
        
        self._log_event("quant_analysis", {
            "regime": regime,
            "regime_confidence": regime_conf,
            "duration_ms": duration_ms,
            "model_count": len(analysis),
            "data": safe_data
        })
    
    # ========== PRICE LOGGING ==========
    
    def log_price(self, price: float, source: str = "live"):
        """Log a price update."""
        self.logger.debug(f"PRICE | ${price:,.2f} | Source: {source}")
        self._log_event("price", {"price": price, "source": source})
    
    # ========== SYSTEM EVENTS ==========
    
    def log_system(self, message: str, level: str = "INFO"):
        """Log a system event."""
        getattr(self.logger, level.lower(), self.logger.info)(f"SYSTEM | {message}")
        self._log_event("system", {"message": message, "level": level})
    
    def log_error(self, message: str, exception: Exception = None):
        """Log an error."""
        err_msg = f"ERROR | {message}"
        if exception:
            err_msg += f" | {type(exception).__name__}: {str(exception)}"
        self.logger.error(err_msg)
        self._log_event("error", {
            "message": message,
            "exception": str(exception) if exception else None
        })
    
    # ========== TIMING DECORATOR ==========
    
    @staticmethod
    def timed(label: str):
        """Decorator to time function execution and log it."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                
                logger = NexusLogger()
                logger.logger.debug(f"TIMING | {label} | {elapsed_ms:.1f}ms")
                logger._log_event("timing", {"label": label, "duration_ms": elapsed_ms})
                
                return result
            return wrapper
        return decorator


# Convenience singleton access
def get_logger() -> NexusLogger:
    return NexusLogger()
