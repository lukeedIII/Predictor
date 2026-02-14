"""
AssGPU Mini-Game Engine
=======================
Crypto-idle mining game inside Nexus Shadow-Quant.
Players transfer trading profits (one-way) to buy AssGPU cards
that mine ASS coin — a simulated token with oscillating price.

GPU Tiers: 10s → 20s → 30s → 40s → 50s
Auto-merge: 4x Tier1→Tier2, then 2x per tier
"""

import json
import os
import time
import math
import random
import logging
from typing import Dict, List, Optional
from datetime import datetime

import config

logger = logging.getLogger(__name__)

# ─── GPU Tier Definitions ────────────────────────────

GPU_TIERS = {
    1: {"name": "10 Series", "label": "10s", "color": "#00E676", "base_cost": 250,   "mining_rate": 1.0},
    2: {"name": "20 Series", "label": "20s", "color": "#2979FF", "base_cost": 600,   "mining_rate": 5.0},
    3: {"name": "30 Series", "label": "30s", "color": "#AA00FF", "base_cost": 1500,  "mining_rate": 15.0},
    4: {"name": "40 Series", "label": "40s", "color": "#FF9100", "base_cost": 4000,  "mining_rate": 50.0},
    5: {"name": "50 Series", "label": "50s", "color": "#FF1744", "base_cost": 10000, "mining_rate": 180.0},
}

# How many cards of the SAME tier needed to auto-merge into the NEXT tier
# Tier 1→2: 4 cards, Tier 2→3: 2, Tier 3→4: 2, Tier 4→5: 2
# Total Tier 1 cards needed: T2=4, T3=8, T4=16, T5=32
MERGE_THRESHOLD = {
    1: 4,   # 4x 10 Series → 1x 20 Series
    2: 2,   # 2x 20 Series → 1x 30 Series
    3: 2,   # 2x 30 Series → 1x 40 Series
    4: 2,   # 2x 40 Series → 1x 50 Series
}

# ASS coin constants
ASS_INITIAL_PRICE = 5.0
ASS_MIN_PRICE = 1.0
ASS_MAX_PRICE = 20.0
ASS_MEAN_PRICE = 7.0
ASS_VOLATILITY = 0.06  # 6% per tick — crypto-level volatility
ASS_MEAN_REVERSION = 0.03  # gentle pull toward mean (slow drift)

TICK_INTERVAL = 30  # seconds between ticks
PRICE_HISTORY_MAX = 500  # max price points to keep


class GpuGame:
    """AssGPU mining game engine."""

    def __init__(self, state_path: str = None):
        self.state_path = state_path or getattr(config, 'GPU_GAME_STATE_PATH',
            os.path.join(getattr(config, 'DATA_DIR', 'data'), 'gpu_game_state.json'))
        
        # Game state
        self.game_balance_usd: float = 0.0      # USD available in game
        self.ass_balance: float = 0.0            # ASS coins held
        self.ass_price: float = ASS_INITIAL_PRICE
        self.ass_price_history: List[Dict] = []  # [{ts, price}]
        self.cards: List[Dict] = []              # [{id, tier, created_ts}]
        self.total_transferred: float = 0.0      # lifetime USD transferred in
        self.total_mined: float = 0.0            # lifetime ASS mined
        self.total_sold_ass: float = 0.0         # lifetime ASS sold
        self.next_card_id: int = 1
        self.last_tick_ts: float = 0.0
        self.created_ts: str = datetime.utcnow().isoformat()
        
        self._load_state()

    # ─── Persistence ──────────────────────────────────

    def _load_state(self):
        """Load game state from JSON file."""
        if not os.path.exists(self.state_path):
            logger.info("[GPU-GAME] No saved state — starting fresh")
            self._init_price_history()
            return
        
        try:
            with open(self.state_path, 'r') as f:
                data = json.load(f)
            
            self.game_balance_usd = data.get('game_balance_usd', 0.0)
            self.ass_balance = data.get('ass_balance', 0.0)
            self.ass_price = data.get('ass_price', ASS_INITIAL_PRICE)
            self.ass_price_history = data.get('ass_price_history', [])
            self.cards = data.get('cards', [])
            self.total_transferred = data.get('total_transferred', 0.0)
            self.total_mined = data.get('total_mined', 0.0)
            self.total_sold_ass = data.get('total_sold_ass', 0.0)
            self.next_card_id = data.get('next_card_id', 1)
            self.last_tick_ts = data.get('last_tick_ts', 0.0)
            self.created_ts = data.get('created_ts', datetime.utcnow().isoformat())
            
            logger.info(f"[GPU-GAME] State loaded: {len(self.cards)} cards, "
                       f"${self.game_balance_usd:.2f} USD, {self.ass_balance:.4f} ASS")
        except Exception as e:
            logger.error(f"[GPU-GAME] Failed to load state: {e}")
            self._init_price_history()

    def _init_price_history(self):
        """Initialize with a few price points."""
        now = time.time()
        for i in range(20):
            ts = now - (20 - i) * TICK_INTERVAL
            noise = random.gauss(0, ASS_VOLATILITY * ASS_MEAN_PRICE)
            price = max(ASS_MIN_PRICE, min(ASS_MAX_PRICE, ASS_INITIAL_PRICE + noise * (i / 20)))
            self.ass_price_history.append({"ts": ts, "price": round(price, 4)})
        self.ass_price = self.ass_price_history[-1]["price"]

    def _save_state(self):
        """Persist game state to JSON."""
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            data = {
                'game_balance_usd': round(self.game_balance_usd, 2),
                'ass_balance': round(self.ass_balance, 6),
                'ass_price': round(self.ass_price, 4),
                'ass_price_history': self.ass_price_history[-PRICE_HISTORY_MAX:],
                'cards': self.cards,
                'total_transferred': round(self.total_transferred, 2),
                'total_mined': round(self.total_mined, 6),
                'total_sold_ass': round(self.total_sold_ass, 6),
                'next_card_id': self.next_card_id,
                'last_tick_ts': self.last_tick_ts,
                'created_ts': self.created_ts,
            }
            with open(self.state_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[GPU-GAME] Failed to save state: {e}")

    # ─── ASS Coin Price Simulation ────────────────────

    def _update_ass_price(self):
        """Mean-reverting random walk for ASS coin price (XRP-style)."""
        # Brownian motion with mean reversion
        noise = random.gauss(0, ASS_VOLATILITY)
        mean_pull = ASS_MEAN_REVERSION * (ASS_MEAN_PRICE - self.ass_price) / ASS_MEAN_PRICE
        
        # Occasional "pump" or "dump" (5% chance per tick)
        if random.random() < 0.05:
            noise *= 3  # amplified move
        
        new_price = self.ass_price * (1 + noise + mean_pull)
        self.ass_price = round(max(ASS_MIN_PRICE, min(ASS_MAX_PRICE, new_price)), 4)
        
        self.ass_price_history.append({
            "ts": time.time(),
            "price": self.ass_price
        })
        
        # Trim history
        if len(self.ass_price_history) > PRICE_HISTORY_MAX:
            self.ass_price_history = self.ass_price_history[-PRICE_HISTORY_MAX:]

    # ─── Mining ───────────────────────────────────────

    def _mine(self, elapsed_seconds: float):
        """Calculate and credit mined ASS coins based on active GPUs."""
        if not self.cards or elapsed_seconds <= 0:
            return 0.0
        
        hours = elapsed_seconds / 3600.0
        total_mined = 0.0
        
        for card in self.cards:
            tier_info = GPU_TIERS.get(card['tier'])
            if tier_info:
                mined = tier_info['mining_rate'] * hours
                total_mined += mined
        
        self.ass_balance += total_mined
        self.total_mined += total_mined
        return total_mined

    # ─── Card Cost (oscillates with ASS price) ────────

    def get_card_cost(self, tier: int = 1) -> float:
        """Card cost in USD, oscillating ±20% with ASS coin price."""
        tier_info = GPU_TIERS.get(tier)
        if not tier_info:
            return 0.0
        
        # Price ratio: how far ASS is from mean as a fraction
        price_ratio = self.ass_price / ASS_MEAN_PRICE
        # Dampen the effect to ±20%
        cost_multiplier = 0.8 + 0.4 * price_ratio  # ranges ~0.8 to ~1.2
        return round(tier_info['base_cost'] * cost_multiplier, 2)

    # ─── Game Actions ─────────────────────────────────

    def transfer_in(self, amount: float) -> Dict:
        """Transfer USD from trading wallet to game (one-way)."""
        if amount <= 0:
            return {"ok": False, "error": "Amount must be positive"}
        
        self.game_balance_usd += amount
        self.total_transferred += amount
        self._save_state()
        
        logger.info(f"[GPU-GAME] Transferred ${amount:.2f} → game balance now ${self.game_balance_usd:.2f}")
        return {"ok": True, "game_balance": round(self.game_balance_usd, 2)}

    def buy_card(self) -> Dict:
        """Buy a 10 Series (tier 1) AssGPU card."""
        cost = self.get_card_cost(tier=1)
        
        if self.game_balance_usd < cost:
            return {"ok": False, "error": f"Need ${cost:.2f}, have ${self.game_balance_usd:.2f}"}
        
        self.game_balance_usd -= cost
        card = {
            "id": self.next_card_id,
            "tier": 1,
            "created_ts": datetime.utcnow().isoformat(),
        }
        self.cards.append(card)
        self.next_card_id += 1
        
        tier_info = GPU_TIERS[1]
        logger.info(f"[GPU-GAME] Bought {tier_info['label']} card #{card['id']} for ${cost:.2f}")
        
        # Auto-merge after buying
        merges = self._auto_merge()
        self._save_state()
        
        return {
            "ok": True, "card": card, "cost": cost,
            "game_balance": round(self.game_balance_usd, 2),
            "merges": merges,
        }

    def _auto_merge(self) -> List[Dict]:
        """Automatically merge cards when threshold is reached.
        
        Merge thresholds:
          - 4x Tier 1 (10 Series) → 1x Tier 2 (20 Series)
          - 2x Tier 2 (20 Series) → 1x Tier 3 (30 Series)
          - 2x Tier 3 (30 Series) → 1x Tier 4 (40 Series)
          - 2x Tier 4 (40 Series) → 1x Tier 5 (50 Series)
        
        Cascades: if merging Tier 1s creates enough Tier 2s, those merge too.
        """
        all_merges = []
        changed = True
        
        while changed:
            changed = False
            for tier in range(1, 5):  # check tiers 1-4
                threshold = MERGE_THRESHOLD.get(tier, 2)
                cards_of_tier = [c for c in self.cards if c['tier'] == tier]
                
                while len(cards_of_tier) >= threshold:
                    # Take the oldest N cards of this tier
                    consumed = cards_of_tier[:threshold]
                    consumed_ids = [c['id'] for c in consumed]
                    
                    new_tier = tier + 1
                    new_card = {
                        "id": self.next_card_id,
                        "tier": new_tier,
                        "created_ts": datetime.utcnow().isoformat(),
                    }
                    
                    # Remove consumed cards, add new one
                    self.cards = [c for c in self.cards if c['id'] not in consumed_ids]
                    self.cards.append(new_card)
                    self.next_card_id += 1
                    
                    tier_info = GPU_TIERS[new_tier]
                    logger.info(f"[GPU-GAME] Auto-merged {threshold}x Tier {tier} → {tier_info['label']} card #{new_card['id']}")
                    all_merges.append({
                        "from_tier": tier,
                        "to_tier": new_tier,
                        "consumed": consumed_ids,
                        "new_card": new_card,
                    })
                    
                    changed = True
                    cards_of_tier = [c for c in self.cards if c['tier'] == tier]
        
        return all_merges

    def merge_cards(self, card_id_1: int, card_id_2: int) -> Dict:
        """Legacy manual merge — now handled by auto-merge."""
        # Auto-merge handles everything now
        merges = self._auto_merge()
        self._save_state()
        if merges:
            return {"ok": True, "merges": merges}
        return {"ok": False, "error": "No cards ready to merge (auto-merge handles this)"}

    def sell_ass(self, amount: float) -> Dict:
        """Sell ASS coins at current price → game USD balance."""
        if amount <= 0:
            return {"ok": False, "error": "Amount must be positive"}
        if amount > self.ass_balance:
            return {"ok": False, "error": f"Not enough ASS (have {self.ass_balance:.4f})"}
        
        usd_received = amount * self.ass_price
        self.ass_balance -= amount
        self.game_balance_usd += usd_received
        self.total_sold_ass += amount
        self._save_state()
        
        logger.info(f"[GPU-GAME] Sold {amount:.4f} ASS @ ${self.ass_price:.2f} = ${usd_received:.2f}")
        return {
            "ok": True,
            "amount_sold": round(amount, 6),
            "price": self.ass_price,
            "usd_received": round(usd_received, 2),
            "game_balance": round(self.game_balance_usd, 2),
            "ass_balance": round(self.ass_balance, 6),
        }

    # ─── Tick (called every ~30s) ─────────────────────

    def tick(self) -> Dict:
        """Update game state: mine coins, update ASS price."""
        now = time.time()
        elapsed = now - self.last_tick_ts if self.last_tick_ts > 0 else TICK_INTERVAL
        
        # Cap elapsed to 1 hour max (prevents huge offline mining)
        elapsed = min(elapsed, 3600)
        
        mined = self._mine(elapsed)
        self._update_ass_price()
        self.last_tick_ts = now
        self._save_state()
        
        return {
            "mined": round(mined, 6),
            "ass_price": self.ass_price,
            "ass_balance": round(self.ass_balance, 6),
            "elapsed": round(elapsed, 1),
        }

    # ─── Full State for Frontend ──────────────────────

    def get_state(self) -> Dict:
        """Return full game state for the frontend."""
        # Enrich cards with tier info
        enriched_cards = []
        for card in self.cards:
            tier_info = GPU_TIERS.get(card['tier'], {})
            enriched_cards.append({
                **card,
                "name": tier_info.get('name', '??'),
                "label": tier_info.get('label', '??'),
                "color": tier_info.get('color', '#888'),
                "mining_rate": tier_info.get('mining_rate', 0),
            })
        
        # Mining stats
        total_mining_rate = sum(
            GPU_TIERS.get(c['tier'], {}).get('mining_rate', 0) for c in self.cards
        )
        
        # Count cards per tier & merge progress
        tier_counts = {}
        for t in range(1, 6):
            count = sum(1 for c in self.cards if c['tier'] == t)
            threshold = MERGE_THRESHOLD.get(t, None)
            tier_counts[str(t)] = {
                "count": count,
                "merge_threshold": threshold,
                "merge_progress": f"{count}/{threshold}" if threshold else "MAX",
            }
        
        return {
            "game_balance_usd": round(self.game_balance_usd, 2),
            "ass_balance": round(self.ass_balance, 6),
            "ass_price": self.ass_price,
            "ass_value_usd": round(self.ass_balance * self.ass_price, 2),
            "cards": enriched_cards,
            "card_cost": self.get_card_cost(1),
            "total_mining_rate": round(total_mining_rate, 2),
            "total_transferred": round(self.total_transferred, 2),
            "total_mined": round(self.total_mined, 6),
            "total_sold_ass": round(self.total_sold_ass, 6),
            "ass_price_history": self.ass_price_history[-100:],
            "tier_counts": tier_counts,
            "merge_thresholds": {str(k): v for k, v in MERGE_THRESHOLD.items()},
            "gpu_tiers": {
                str(k): {
                    "name": v["name"],
                    "label": v["label"],
                    "color": v["color"],
                    "base_cost": v["base_cost"],
                    "mining_rate": v["mining_rate"],
                }
                for k, v in GPU_TIERS.items()
            },
        }
