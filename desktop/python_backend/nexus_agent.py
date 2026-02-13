"""
Dr. Nexus — AI-Powered BTC Quant Analyst
==========================================
Uses OpenAI GPT-4o to provide intelligent market analysis
based on a live JSON snapshot of all app state.
"""

import json
import logging
import os
import time
import uuid
import threading
import requests
from datetime import datetime
from typing import Optional, Dict, List

import config
import nexus_memory

# ============================================================
#  API KEY — loaded from environment / .env file
# ============================================================
_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


# ============================================================
#  SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are **Dr. Nexus** 🧬 — the built-in AI quant analyst for Nexus Shadow-Quant, a professional BTC/USDT trading intelligence platform.

## Your Identity
You are a senior quantitative finance researcher with deep expertise in:
- Cryptocurrency market microstructure
- Machine learning for time-series forecasting
- Risk management and portfolio optimization
- Technical analysis and statistical arbitrage

Your tone is **professional but approachable** — like a senior quant mentor explaining things to a smart junior trader. Use precise financial terminology but make it understandable.

## The Platform's Algorithm Stack (explain these when relevant)

### XGBoost Ensemble Predictor
- Trained on 6+ years of BTC 1-minute candles (3.15M+ data points)
- **35 engineered features**, ALL scale-invariant (returns/ratios, not raw prices)
- Feature categories: Returns-based OHLCV (6), Technical indicators (4), Fourier cycles (2), Market microstructure (3), Quant features (3), Advanced quant (2), Multi-timeframe trend (6), Volume profile (2), Cross-asset correlation (7)
- Prediction horizon: 15 minutes (optimized for actionable timeframe)
- Statistically validated: 50.71% accuracy with positive Sharpe ratio (0.88)

### LSTM Neural Network (Production-Grade)
- **512 hidden units, 3 layers** with LayerNorm, dropout 0.3 (~50MB VRAM, RTX 3060+)
- FC head: 512→128→ReLU→Dropout→1
- Starts with 0 weight in ensemble, earns weight only when validation > 52%
- Trained on 30-step sliding window sequences of all 35 features

### Cross-Asset Correlation Engine
- Real-time data from 3 Binance pairs: ETH/USDT, PAXG/USDT (gold), ETH/BTC
- ETH often leads BTC by 1-5 min; ETH/BTC ratio measures BTC dominance; Gold signals macro fear
- 7 features: eth_ret_5, eth_ret_15, eth_vol_ratio, ethbtc_ret_5, ethbtc_trend, gold_ret_15, gold_ret_60

### Two-Agent Feedback Loop
- Trader logs every closed trade outcome (pnl_usd, regime, confidence) to trade_feedback.json
- Predictor reads feedback, adjusts confidence ±10% based on regime-specific win rate
- Adaptive threshold auto-adjusts (40-75%) using PnL-weighted composite score
- A $100 loss matters 10x more than a $10 loss in performance assessment

### Quant Engine (Institutional Analysis)
- **Regime Detection**: Hidden Markov Model classifying market into TRENDING/MEAN_REVERTING/VOLATILE/CHAOTIC states
- **FFT Cycle Analysis**: Fourier decomposition to identify dominant market cycles (short/mid/long)
- **Order Flow Simulation**: Synthetic microstructure analysis modeling buy/sell pressure
- **Hawkes Process**: Self-exciting point process for modeling volatility clustering and jump risk
- **Wasserstein Drift**: Optimal Transport metric measuring distribution shift between recent and historical returns

### Hurst Exponent
- H > 0.5: Trending/persistent market → follow the trend
- H ≈ 0.5: Random walk → chaotic, skip trades
- H < 0.5: Mean-reverting → expect reversals

### Paper Trading Risk Management
- **Signal Confirmation**: 2 consecutive same-direction predictions required
- No direction stacking (can't open 2 longs simultaneously)
- **Trailing Stop Loss**: Activates >0.3% in profit, locks 50% of gains
- Circuit breaker at 20% drawdown
- Kelly Criterion position sizing (half-Kelly, capped at 25%)
- **Multi-position**: Up to 3 concurrent positions with margin allocation
- **Adaptive entry threshold**: Auto-adjusts based on PnL-weighted performance
- TP/SL on every position (1.5:1 reward-to-risk)
- Max hold time: 1 hour

### BTC Quarterly Seasonality (Coinglass Data — use for seasonal bias)
Historical quarterly returns (%) — one of the strongest patterns in crypto:

| Year | Q1       | Q2       | Q3       | Q4       |
|------|----------|----------|----------|----------|
| 2026 | -22.73%  | —        | —        | —        |
| 2025 | -11.82%  | +29.74%  | +6.31%   | -23.07%  |
| 2024 | +68.68%  | -11.92%  | +0.96%   | +47.73%  |
| 2023 | +71.77%  | +7.19%   | -11.54%  | +56.9%   |
| 2022 | -1.46%   | -56.2%   | -2.57%   | -14.75%  |
| 2021 | +103.17% | -40.36%  | +25.01%  | +5.45%   |
| 2020 | -10.83%  | +42.33%  | +17.97%  | +168.02% |
| 2019 | +8.74%   | +159.36% | -22.86%  | -13.54%  |
| 2018 | -49.7%   | -7.71%   | +3.61%   | -42.16%  |
| 2017 | +11.89%  | +123.86% | +80.41%  | +215.07% |
| 2016 | -3.06%   | +62.06%  | -9.41%   | +58.17%  |
| 2015 | -24.14%  | +7.57%   | -10.05%  | +81.24%  |
| 2014 | -37.42%  | +40.43%  | -39.74%  | -16.7%   |
| 2013 | +539.96% | -3.97%   | +40.6%   | +479.59% |
| **AVG** | **+45.93%** | **+27.11%** | **+6.05%** | **+77.07%** |
| **MED** | **-2.26%** | **+7.57%** | **+0.96%** | **+47.73%** |

Key seasonal insights to reference:
- **Q4 is historically the BEST quarter** (avg +77%, median +48%) — bullish seasonal bias
- **Q1 is the WEAKEST** (median -2.26%) — often post-Q4 correction
- **Q2 is strong** (avg +27%) — often starts the mid-year rally
- **Q3 is flat/weak** (median +0.96%) — summer doldrums
- **Post-halving years** (2013, 2017, 2021, 2025) tend to have explosive Q4s
- **Bear market years** (2018, 2022) show negative across ALL quarters
- Use the `seasonality.current_quarter` from the state JSON to tell the user where we are in the cycle

## Your Data (LIVE — Updated Every Request)
Below is the platform's current state. This is real-time data, not historical.

```json
{STATE_JSON}
```

## Response Guidelines
1. **Always reference actual numbers** from the JSON (price $X, confidence Y%, Hurst Z, etc.)
2. Use **tables** when comparing data (positions, stats, etc.) — format them properly in markdown
3. Use **bold** for key insights and ⚠️ for risk warnings
4. Structure responses with headers (##) for different analysis sections
5. Keep analysis **actionable** — what does the data mean for trading decisions?
6. When asked about the algorithm, explain which specific components are producing the current signals
7. If positions are open, always comment on their risk/reward status
8. Reference the **feedback loop** when discussing performance (adaptive threshold, regime scoring)
9. When discussing cross-asset data, explain how ETH/Gold/ETH-BTC relate to BTC outlook
10. End important analyses with a quick **📊 Summary** section

## Formatting Rules
- Use markdown tables for structured data
- Use `code blocks` for numbers and technical values
- Use bullet points for lists
- Use > blockquotes for key takeaways
- Use --- horizontal rules between major sections

⚠️ **Disclaimer**: This is an educational research tool. Not financial advice. Past performance does not guarantee future results."""


# ============================================================
#  STATE AGGREGATOR
# ============================================================

def build_state_snapshot(predictor=None, trader=None, collector=None) -> Dict:
    """
    Build a comprehensive JSON snapshot of all app state.
    Compact but complete — injected into system prompt.
    """
    now = datetime.now()
    quarter = (now.month - 1) // 3 + 1
    state = {
        "ts": now.strftime("%Y-%m-%d %H:%M:%S"),
        "v": config.VERSION,
        "seasonality": {
            "current_quarter": f"Q{quarter} {now.year}",
            "month": now.strftime("%B"),
            "note": f"Q{quarter} historical avg: " + {1: "+45.93%", 2: "+27.11%", 3: "+6.05%", 4: "+77.07%"}[quarter],
        },
    }
    
    # ── Prediction + Market Data ──────────────────────
    try:
        if predictor and hasattr(predictor, 'get_prediction') and predictor.is_trained:
            pred = predictor.get_prediction()
            state["market"] = {
                "btc_price": pred.get("current_price", 0),
                "direction": pred.get("direction", "?"),
                "confidence_pct": pred.get("confidence", 0),
                "target_15m": pred.get("target_price", 0),
                "target_30m": pred.get("target_price_2h", 0),
                "hurst": round(pred.get("hurst", 0.5), 4),
                "regime": pred.get("regime_label", "?"),
                "xgb_prob": pred.get("xgb_prob", 0),
                "lstm_prob": pred.get("lstm_prob", 0),
                "ensemble": pred.get("ensemble_weights", ""),
                "verified": pred.get("verified", False),
            }
            
            # Quant signals
            q = predictor.last_quant_analysis or {}
            if q:
                regime = q.get("regime", {})
                cyc = q.get("fft_cycles", {})
                flow = q.get("order_flow", {})
                jump = q.get("jump_risk", {})
                state["quant"] = {
                    "regime": regime.get("regime", "?"),
                    "regime_conf": round(regime.get("confidence", 0), 2),
                    "cycle_short": round(cyc.get("short", {}).get("strength", 0), 3),
                    "cycle_mid": round(cyc.get("mid", {}).get("strength", 0), 3),
                    "cycle_long": round(cyc.get("long", {}).get("strength", 0), 3),
                    "buy_pressure": round(flow.get("buy_pressure", 0.5), 3),
                    "jump_risk": jump.get("level", "?"),
                }
            
            # Alt signals
            alt = predictor.last_alt_signals or {}
            if alt:
                state["alt_signals"] = {
                    "fear_greed": alt.get("fear_greed_index", "?"),
                    "sentiment": alt.get("overall_sentiment", "?"),
                }
            
            # Accuracy
            state["accuracy"] = {
                "validated_pct": predictor.last_validation_accuracy,
                "stat_verified": predictor.is_statistically_verified,
            }
    except Exception as e:
        logging.warning(f"Dr. Nexus state: prediction err: {e}")
        state["market"] = {"error": str(e)[:100]}
    
    # ── Portfolio + Positions ─────────────────────────
    try:
        if trader:
            s = trader.get_stats()
            state["portfolio"] = {
                "balance": round(s.get("balance", 10000), 2),
                "start_bal": s.get("starting_balance", 10000),
                "pnl_usd": round(s.get("total_pnl", 0), 2),
                "pnl_pct": round(s.get("total_pnl_pct", 0), 2),
                "trades": s.get("total_trades", 0),
                "wins": s.get("winning_trades", 0),
                "losses": s.get("losing_trades", 0),
                "win_rate": round(s.get("win_rate", 0), 1),
                "sharpe": round(s.get("sharpe_ratio", 0), 2),
                "max_dd_pct": round(s.get("max_drawdown_pct", 0), 2),
                "profit_factor": round(s.get("profit_factor", 0), 2),
                "kelly": round(s.get("kelly_fraction", 0.01), 3),
                "bot_on": trader.is_running,
                "leverage": s.get("leverage", 10),
            }
            
            # Open positions (full detail)
            price = getattr(trader, '_last_price', None)
            positions = []
            for pos in trader.positions:
                p = price or pos.entry_price
                positions.append({
                    "dir": pos.direction,
                    "entry": round(pos.entry_price, 2),
                    "size_usd": round(pos.size_usd, 0),
                    "margin_usd": round(getattr(pos, 'margin', pos.size_usd / 10), 2),
                    "leverage": getattr(pos, 'leverage', 10),
                    "upnl_usd": round(pos.unrealized_pnl(p), 2),
                    "upnl_pct": round(pos.unrealized_pnl_pct(p), 2),
                    "tp": round(pos.tp_price, 2) if pos.tp_price else None,
                    "sl": round(pos.sl_price, 2) if pos.sl_price else None,
                    "liq": round(pos.liq_price, 2) if hasattr(pos, 'liq_price') and pos.liq_price else None,
                    "hold_min": round((datetime.now() - pos.entry_time).total_seconds() / 60, 1),
                })
            state["positions"] = positions
            
            # Recent closed trades (last 5)
            if trader.trade_history:
                recent = trader.trade_history[-5:]
                state["last_trades"] = [
                    {
                        "dir": t.get("direction"),
                        "pnl": round(t.get("pnl_usd", 0), 2),
                        "pnl_pct": round(t.get("pnl_pct", 0), 2),
                        "reason": t.get("close_reason", ""),
                        "entry": round(t.get("entry_price", 0), 2),
                        "exit": round(t.get("exit_price", 0), 2),
                    }
                    for t in recent
                ]
    except Exception as e:
        logging.warning(f"Dr. Nexus state: trader err: {e}")
        state["portfolio"] = {"error": str(e)[:100]}
    
    return state


# ============================================================
#  AI CHAT (OpenAI GPT-4o)
# ============================================================

def _call_openai(system_prompt: str, user_message: str, api_key: str,
                 conversation_history: List[Dict] = None) -> str:
    """Call OpenAI API with GPT-4o, supporting multi-turn conversations."""
    url = "https://api.openai.com/v1/chat/completions"
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Inject recent conversation history for continuity
    if conversation_history:
        for msg in conversation_history[-8:]:  # Last 8 messages max
            role = "user" if msg.get("role") == "user" else "assistant"
            content = msg.get("content", "")[:500]  # Truncate for token budget
            messages.append({"role": role, "content": content})
    
    # Current user message
    messages.append({"role": "user", "content": user_message})
    
    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2000,
    }
    
    resp = requests.post(
        url,
        json=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=45,
    )
    resp.raise_for_status()
    data = resp.json()
    
    choices = data.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "No response generated.")
    
    return "No response generated."


# ============================================================
#  KNOWLEDGE EXTRACTION (async post-chat)
# ============================================================

_EXTRACT_PROMPT = """Analyze the following conversation exchange. If there are key insights worth remembering, extract them as JSON.
Only extract genuinely useful information (market observations, trading lessons, user preferences, model observations).
If nothing is worth saving, return {"insights": []}.

User said: {user_msg}
Dr. Nexus replied: {agent_reply}

Return ONLY valid JSON:
{{"insights": [
  {{"category": "market_insight|trading_lesson|user_preference|model_observation|risk_note|general", "content": "concise fact"}}
]}}"""

def _extract_knowledge_async(user_msg: str, agent_reply: str, api_key: str, msg_id: int):
    """Background thread: extract knowledge from conversation."""
    try:
        prompt = _EXTRACT_PROMPT.format(
            user_msg=user_msg[:300],
            agent_reply=agent_reply[:500]
        )
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",  # Cheap model for extraction
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 300,
            },
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=15,
        )
        if resp.status_code == 200:
            text = resp.json()["choices"][0]["message"]["content"]
            # Parse JSON from response
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(text)
            for insight in data.get("insights", []):
                nexus_memory.save_knowledge(
                    category=insight.get("category", "general"),
                    content=insight.get("content", ""),
                    source_msg_id=msg_id,
                )
                logging.info(f"Dr. Nexus learned: [{insight.get('category')}] {insight.get('content', '')[:60]}")
    except Exception as e:
        logging.debug(f"Knowledge extraction skipped: {e}")


# Active session tracking
_current_session_id = None

def get_or_create_session() -> str:
    """Get the current session ID or create a new one."""
    global _current_session_id
    if not _current_session_id:
        _current_session_id = str(uuid.uuid4())[:8]
    return _current_session_id

def new_session() -> str:
    """Start a new chat session."""
    global _current_session_id
    _current_session_id = str(uuid.uuid4())[:8]
    return _current_session_id


def chat(user_message: str, predictor=None, trader=None, collector=None,
         session_id: str = None) -> Dict:
    """
    Main chat function — builds state, crafts prompt, calls GPT-4o.
    Now with persistent memory and multi-turn conversation.
    Returns {"reply": str, "provider": str, "state_snapshot": dict, "session_id": str}
    """
    # 0. Get or create session
    sid = session_id or get_or_create_session()
    
    # 1. Save user message to memory
    user_msg_id = nexus_memory.save_message(sid, "user", user_message)
    
    # 2. Build live state snapshot
    state = build_state_snapshot(predictor, trader, collector)
    state_json = json.dumps(state, indent=2, default=str)
    
    # 3. Build system prompt with injected state + knowledge
    knowledge_context = nexus_memory.get_knowledge_summary()
    full_prompt = SYSTEM_PROMPT.replace("{STATE_JSON}", state_json)
    if knowledge_context:
        full_prompt += f"\n\n{knowledge_context}"
    
    # 4. Load conversation history for multi-turn
    conv_history = nexus_memory.get_recent_messages(session_id=sid, limit=10)
    # Remove the message we just saved (it'll be sent as current user message)
    conv_history = [m for m in conv_history if m['id'] != user_msg_id]
    
    # 5. Prefer user's key from config/.env, fallback to built-in key
    api_key = getattr(config, 'OPENAI_API_KEY', None) or os.environ.get('OPENAI_API_KEY', '')
    
    if not api_key:
        # Fallback: read .env directly
        try:
            env_path = os.path.join(config.DATA_ROOT, ".env")
            if os.path.exists(env_path):
                with open(env_path, 'r', encoding='utf-8', errors='replace') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("OPENAI_API_KEY="):
                            api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    
    if not api_key:
        # Last resort: hardcoded key (may be over quota)
        api_key = _OPENAI_API_KEY
    
    if not api_key:
        return {
            "reply": "⚠️ No OpenAI API key found. Please add one in **Settings** or contact the developer.",
            "provider": "none",
            "state_snapshot": state,
            "session_id": sid,
        }
    
    # 6. Call GPT-4o with conversation history
    try:
        reply = _call_openai(full_prompt, user_message, api_key,
                             conversation_history=conv_history)
        
        # Save agent reply to memory
        agent_msg_id = nexus_memory.save_message(sid, "agent", reply, provider="gpt-4o")
        
        # Async knowledge extraction (don't block the response)
        t = threading.Thread(
            target=_extract_knowledge_async,
            args=(user_message, reply, api_key, agent_msg_id),
            daemon=True
        )
        t.start()
        
        return {
            "reply": reply,
            "provider": "gpt-4o",
            "state_snapshot": state,
            "session_id": sid,
        }
    except requests.exceptions.HTTPError as e:
        error_detail = ""
        try:
            error_detail = e.response.json().get("error", {}).get("message", str(e))
        except Exception:
            error_detail = str(e)
        logging.error(f"Dr. Nexus API error: {error_detail}")
        return {
            "reply": f"⚠️ API Error: {error_detail}",
            "provider": "error",
            "state_snapshot": state,
            "session_id": sid,
        }
    except Exception as e:
        logging.error(f"Dr. Nexus chat error: {e}")
        return {
            "reply": f"⚠️ Error: {str(e)}",
            "provider": "error",
            "state_snapshot": state,
            "session_id": sid,
        }
