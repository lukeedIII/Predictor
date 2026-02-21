/**
 * GpuFarm — AssGPU Mini-Game Tab
 * ==============================
 * Buy GPUs and mine ASS coin. Cards auto-merge when thresholds are met.
 * 4x Tier1→Tier2, 2x Tier2→Tier3, 2x Tier3→Tier4, 2x Tier4→Tier5
 */
import { useState, useEffect, useCallback, useRef } from 'react';

const API = 'http://127.0.0.1:8420';

// ─── Types ───────────────────────────────────────
interface GpuCard {
    id: number;
    tier: number;
    name: string;
    label: string;
    color: string;
    mining_rate: number;
    created_ts: string;
}

interface TierInfo {
    name: string;
    label: string;
    color: string;
    base_cost: number;
    mining_rate: number;
}

interface TierCount {
    count: number;
    merge_threshold: number | null;
    merge_progress: string;
}

interface GameState {
    game_balance_usd: number;
    ass_balance: number;
    ass_price: number;
    ass_value_usd: number;
    cards: GpuCard[];
    card_cost: number;
    total_mining_rate: number;
    total_transferred: number;
    total_mined: number;
    total_sold_ass: number;
    ass_price_history: { ts: number; price: number }[];
    gpu_tiers: Record<string, TierInfo>;
    tier_counts?: Record<string, TierCount>;
    merge_thresholds?: Record<string, number>;
}

// ─── Tier visual config ──────────────────────────
const TIER_GLOW: Record<number, string> = {
    1: '0 0 20px rgba(0,230,118,0.4)',
    2: '0 0 20px rgba(41,121,255,0.4)',
    3: '0 0 25px rgba(170,0,255,0.4)',
    4: '0 0 30px rgba(255,145,0,0.5)',
    5: '0 0 40px rgba(255,23,68,0.6), 0 0 80px rgba(255,215,0,0.3)',
};

const TIER_BG: Record<number, string> = {
    1: 'linear-gradient(135deg, #0d2818 0%, #1a3a2a 100%)',
    2: 'linear-gradient(135deg, #0d1528 0%, #1a2a3a 100%)',
    3: 'linear-gradient(135deg, #1a0d28 0%, #2a1a3a 100%)',
    4: 'linear-gradient(135deg, #281a0d 0%, #3a2a1a 100%)',
    5: 'linear-gradient(135deg, #280d0d 0%, #3a1a1a 100%)',
};

const TIER_COLORS: Record<number, string> = {
    1: '#00E676', 2: '#2979FF', 3: '#AA00FF', 4: '#FF9100', 5: '#FF1744',
};

const MERGE_THRESHOLDS: Record<number, number> = {
    1: 4, 2: 2, 3: 2, 4: 2,
};

// ─── Price sparkline (simple SVG) ────────────
const Sparkline = ({ data }: { data: { ts: number; price: number }[] }) => {
    if (data.length < 2) return null;
    const prices = data.map(d => d.price);
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    const range = max - min || 1;
    const w = 240, h = 60;
    const points = prices.map((p, i) =>
        `${(i / (prices.length - 1)) * w},${h - ((p - min) / range) * h}`
    ).join(' ');
    const trending = prices[prices.length - 1] >= prices[0];
    return (
        <svg width={w} height={h} style={{ opacity: 0.9 }}>
            <polyline fill="none" stroke={trending ? '#00E676' : '#FF1744'} strokeWidth="2" points={points} />
        </svg>
    );
};

// ─── Merge progress bar ──────────────────────
const MergeProgress = ({ state }: { state: GameState | null }) => {
    const tiers = state?.tier_counts;
    if (!tiers) return null;

    return (
        <div className="gpu-merge-progress">
            <h3 style={{ margin: '0 0 10px', fontSize: 14, opacity: 0.7 }}>Auto-Merge Progress</h3>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                {[1, 2, 3, 4].map(tier => {
                    const info = tiers[String(tier)];
                    if (!info) return null;
                    const threshold = info.merge_threshold || MERGE_THRESHOLDS[tier] || 2;
                    const count = info.count;
                    const pct = Math.min((count / threshold) * 100, 100);
                    const tierName = state?.gpu_tiers?.[String(tier)]?.label || `T${tier}`;
                    const nextName = state?.gpu_tiers?.[String(tier + 1)]?.label || `T${tier + 1}`;
                    const color = TIER_COLORS[tier + 1] || '#888';
                    return (
                        <div key={tier} style={{
                            flex: '1 1 120px', background: 'rgba(255,255,255,0.03)',
                            borderRadius: 8, padding: '8px 12px', border: '1px solid rgba(255,255,255,0.06)',
                        }}>
                            <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 4 }}>
                                {threshold}× {tierName} → {nextName}
                            </div>
                            <div style={{
                                height: 6, background: 'rgba(255,255,255,0.08)', borderRadius: 3,
                                overflow: 'hidden', marginBottom: 3,
                            }}>
                                <div style={{
                                    width: `${pct}%`, height: '100%', background: color,
                                    borderRadius: 3, transition: 'width 0.3s ease',
                                }} />
                            </div>
                            <div style={{ fontSize: 12, fontWeight: 600, color }}>
                                {count}/{threshold}
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export default function GpuFarm() {
    const [state, setState] = useState<GameState | null>(null);
    const [transferAmt, setTransferAmt] = useState('');
    const [sellAmt, setSellAmt] = useState('');
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

    // ─── Fetch game state ────────────────────────
    const fetchState = useCallback(async () => {
        try {
            const res = await fetch(`${API}/api/game/state`);
            if (res.ok) {
                const data = await res.json();
                setState(data);
            }
        } catch { /* backend not ready */ }
    }, []);

    useEffect(() => {
        void fetchState();
        intervalRef.current = setInterval(fetchState, 5000);
        return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
    }, [fetchState]);

    // ─── Feedback helpers ────────────────────────
    const flash = (msg: string, type: 'ok' | 'err') => {
        if (type === 'ok') { setSuccess(msg); setError(''); }
        else { setError(msg); setSuccess(''); }
        setTimeout(() => { setSuccess(''); setError(''); }, 3000);
    };

    // ─── Actions ─────────────────────────────────
    const doTransfer = async () => {
        const amt = parseFloat(transferAmt);
        if (!amt || amt <= 0) return flash('Enter a valid amount', 'err');
        try {
            const res = await fetch(`${API}/api/game/transfer`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ amount: amt }),
            });
            if (!res.ok) { const d = await res.json(); return flash(d.detail || 'Transfer failed', 'err'); }
            flash(`Transferred $${amt.toFixed(2)} to game wallet`, 'ok');
            setTransferAmt('');
            fetchState();
        } catch { flash('Transfer failed', 'err'); }
    };

    const doBuy = async () => {
        try {
            const res = await fetch(`${API}/api/game/buy-card`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
            });
            if (!res.ok) { const d = await res.json(); return flash(d.detail || 'Buy failed', 'err'); }
            const data = await res.json();
            const mergeCount = data.merges?.length || 0;
            const msg = mergeCount > 0
                ? `Bought 10s for $${data.cost?.toFixed(2)} — ${mergeCount} auto-merge${mergeCount > 1 ? 's' : ''} 🔥`
                : `Bought 10 Series card for $${data.cost?.toFixed(2)}`;
            flash(msg, 'ok');
            fetchState();
        } catch { flash('Buy failed', 'err'); }
    };

    const doSellAss = async () => {
        const amt = parseFloat(sellAmt);
        if (!amt || amt <= 0) return flash('Enter a valid amount', 'err');
        try {
            const res = await fetch(`${API}/api/game/sell-ass`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ amount: amt }),
            });
            if (!res.ok) { const d = await res.json(); return flash(d.detail || 'Sell failed', 'err'); }
            const data = await res.json();
            flash(`Sold ${data.amount_sold} ASS for $${data.usd_received?.toFixed(2)}`, 'ok');
            setSellAmt('');
            fetchState();
        } catch { flash('Sell failed', 'err'); }
    };


    if (!state) return (
        <div className="page gpu-page" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100%' }}>
            <div style={{ textAlign: 'center', opacity: 0.5 }}>
                <div className="spinner" />
                <p style={{ marginTop: 12 }}>Loading GPU Farm...</p>
            </div>
        </div>
    );

    return (
        <div className="page gpu-page">
            {/* ─── Feedback toasts ─── */}
            {error && <div className="gpu-toast gpu-toast-err">{error}</div>}
            {success && <div className="gpu-toast gpu-toast-ok">{success}</div>}

            {/* ─── Header Stats ─── */}
            <div className="gpu-header">
                <div className="gpu-stat-row">
                    <div className="gpu-stat">
                        <span className="gpu-stat-label">Game Balance</span>
                        <span className="gpu-stat-value">${state.game_balance_usd.toFixed(2)}</span>
                    </div>
                    <div className="gpu-stat">
                        <span className="gpu-stat-label">ASS Balance</span>
                        <span className="gpu-stat-value">{state.ass_balance.toFixed(4)} ASS</span>
                        <span className="gpu-stat-sub">≈ ${state.ass_value_usd.toFixed(2)}</span>
                    </div>
                    <div className="gpu-stat">
                        <span className="gpu-stat-label">ASS Price</span>
                        <span className="gpu-stat-value" style={{ color: state.ass_price >= 7 ? '#00E676' : '#FF9100' }}>
                            ${state.ass_price.toFixed(2)}
                        </span>
                    </div>
                    <div className="gpu-stat">
                        <span className="gpu-stat-label">Mining Rate</span>
                        <span className="gpu-stat-value">{state.total_mining_rate.toFixed(1)} ASS/hr</span>
                    </div>
                </div>
                <div className="gpu-chart-row">
                    <Sparkline data={state.ass_price_history} />
                </div>
            </div>

            {/* ─── Action Panels ─── */}
            <div className="gpu-actions">
                {/* Transfer */}
                <div className="gpu-action-panel">
                    <h3>💰 Transfer to Game</h3>
                    <p className="gpu-action-desc">One-way transfer from trading wallet</p>
                    <div className="gpu-input-row">
                        <span className="gpu-input-prefix">$</span>
                        <input
                            type="number" placeholder="Amount" value={transferAmt}
                            onChange={e => setTransferAmt(e.target.value)}
                            className="gpu-input"
                        />
                        <button onClick={doTransfer} className="gpu-btn gpu-btn-transfer">Transfer</button>
                    </div>
                </div>

                {/* Buy */}
                <div className="gpu-action-panel">
                    <h3>🛒 Buy 10 Series</h3>
                    <p className="gpu-action-desc">Current price: <strong>${state.card_cost.toFixed(2)}</strong></p>
                    <button onClick={doBuy} className="gpu-btn gpu-btn-buy" disabled={state.game_balance_usd < state.card_cost}>
                        Buy AssGPU 10s
                    </button>
                    <p className="gpu-action-desc" style={{ marginTop: 6, fontSize: 11, opacity: 0.5 }}>
                        Cards auto-merge: 4× 10s→20s, 2× 20s→30s, 2× 30s→40s, 2× 40s→50s
                    </p>
                </div>

                {/* Sell ASS */}
                <div className="gpu-action-panel">
                    <h3>📈 Sell ASS Coin</h3>
                    <p className="gpu-action-desc">Current rate: ${state.ass_price.toFixed(2)}/ASS</p>
                    <div className="gpu-input-row">
                        <input
                            type="number" placeholder="Amount" value={sellAmt}
                            onChange={e => setSellAmt(e.target.value)}
                            className="gpu-input"
                        />
                        <button onClick={() => setSellAmt(state.ass_balance.toFixed(4))} className="gpu-btn-max">MAX</button>
                        <button onClick={doSellAss} className="gpu-btn gpu-btn-sell" disabled={state.ass_balance <= 0}>Sell</button>
                    </div>
                </div>
            </div>

            {/* ─── Auto-Merge Progress ─── */}
            <MergeProgress state={state} />

            {/* ─── GPU Card Grid ─── */}
            <div className="gpu-grid-header">
                <h2>Your GPU Farm</h2>
                <span className="gpu-grid-count">{state.cards.length} card{state.cards.length !== 1 ? 's' : ''}</span>
            </div>

            {state.cards.length === 0 ? (
                <div className="gpu-empty">
                    <p>No GPU cards yet. Transfer funds and buy your first 10 Series!</p>
                </div>
            ) : (
                <div className="gpu-grid">
                    {state.cards
                        .sort((a, b) => b.tier - a.tier || a.id - b.id)
                        .map(card => (
                            <div
                                key={card.id}
                                className="gpu-card"
                                style={{
                                    background: TIER_BG[card.tier] || '#111',
                                    borderColor: card.color,
                                    boxShadow: TIER_GLOW[card.tier],
                                }}
                            >
                                <div className="gpu-card-tier" style={{ color: card.color }}>
                                    {card.label}
                                </div>
                                <div className="gpu-card-name">{card.name}</div>
                                <div className="gpu-card-chip" style={{ background: card.color }} />
                                <div className="gpu-card-rate">
                                    ⛏ {card.mining_rate} ASS/hr
                                </div>
                                <div className="gpu-card-id">#{card.id}</div>
                            </div>
                        ))}
                </div>
            )}

            {/* ─── Stats Footer ─── */}
            <div className="gpu-footer">
                <span>Total Transferred: ${state.total_transferred.toFixed(2)}</span>
                <span>Total Mined: {state.total_mined.toFixed(2)} ASS</span>
                <span>Total Sold: {state.total_sold_ass.toFixed(2)} ASS</span>
            </div>
        </div>
    );
}
