import { useState } from 'react';
import { useApi, apiPost } from '../hooks/useApi';
import { useLivePositions, useLiveStats, useLiveBotStatus } from '../stores/liveStore';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import {
    IconChart, IconLineChart, IconList, IconPlay, IconSquare,
    IconArrowUp, IconArrowDown, IconXCircle,
} from '../components/Icons';

/* ─── Types ──────────────────────────────────────── */
type Position = {
    index: number;
    direction: string;
    entry_price: number;
    size_usd: number;
    leverage: number;
    margin: number;
    unrealized_pnl: number;
    unrealized_pnl_pct: number;
    tp_price: number;
    sl_price: number;
    elapsed_seconds: number;
};

type StatsData = {
    balance: number;
    total_pnl: number;
    total_pnl_pct: number;
    win_rate: number;
    winning_trades: number;
    losing_trades: number;
    total_trades: number;
    sharpe_ratio: number;
    max_drawdown_pct: number;
    kelly_fraction: number;
    profit_factor: number;
};

type TradeData = { trades: Array<Record<string, unknown>> };
type EquityData = { points: Array<{ timestamp: string; balance: number }> };

/* ─── Helpers ────────────────────────────────────── */
function formatDuration(seconds: number): string {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${m}m`;
}

/* ─── Position Card ──────────────────────────────── */
function PositionCard({ pos, onClose }: { pos: Position; onClose: () => void }) {
    const pnlColor = pos.unrealized_pnl >= 0 ? 'var(--positive)' : 'var(--negative)';
    const isLong = pos.direction === 'LONG';

    return (
        <div className="card animate-in relative overflow-hide">
            {/* Direction glow */}
            <div
                className="position-glow"
                style={{ background: isLong ? 'var(--positive)' : 'var(--negative)' }}
            />

            <div className="flex-start">
                <div>
                    <span className={`badge ${isLong ? 'badge-long' : 'badge-short'}`}>
                        {isLong ? '↑' : '↓'} {pos.direction}
                    </span>
                    <div className="stat-sub mt-6">{formatDuration(pos.elapsed_seconds)}</div>
                </div>
                <button className="btn btn-danger" onClick={onClose} style={{ padding: '6px 12px', fontSize: 11 }}>
                    Close
                </button>
            </div>

            <div className="grid-4 mt-16 gap-12">
                <div>
                    <div className="stat-label">ENTRY</div>
                    <div className="stat-value xs mono">${pos.entry_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
                </div>
                <div>
                    <div className="stat-label">SIZE</div>
                    <div className="stat-value xs mono">${pos.size_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
                    <div className="stat-sub">{pos.leverage}x · ${pos.margin.toFixed(0)}m</div>
                </div>
                <div>
                    <div className="stat-label">PNL</div>
                    <div className="stat-value xs mono" style={{ color: pnlColor }}>
                        ${pos.unrealized_pnl.toFixed(2)} ({pos.unrealized_pnl_pct > 0 ? '+' : ''}{pos.unrealized_pnl_pct.toFixed(1)}%)
                    </div>
                </div>
                <div>
                    <div className="stat-label">TP / SL</div>
                    <div className="tp-sl-row">
                        <span className="text-positive">${pos.tp_price?.toLocaleString()}</span>
                        {' / '}
                        <span className="text-negative">${pos.sl_price?.toLocaleString()}</span>
                    </div>
                </div>
            </div>
        </div>
    );
}

/* ─── Stats Panel ────────────────────────────────── */
function StatsPanel({ stats }: { stats: StatsData | null }) {
    if (!stats) return <div className="card skeleton skeleton-h200" />;

    const balColor = stats.total_pnl >= 0 ? 'var(--positive)' : 'var(--negative)';
    const wrColor = stats.win_rate >= 50 ? 'var(--positive)' : stats.win_rate >= 40 ? 'var(--warning)' : 'var(--negative)';

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title"><IconChart size={14} style={{ marginRight: 5, verticalAlign: -2 }} /> Performance</span>
            </div>

            {/* Balance hero */}
            <div className="balance-hero">
                <div className="stat-label">PORTFOLIO VALUE</div>
                <div className="stat-value" style={{ fontSize: 28, color: balColor }}>
                    ${stats.balance.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </div>
                <div className="stat-sub" style={{ color: balColor }}>
                    {stats.total_pnl_pct >= 0 ? '+' : ''}{stats.total_pnl_pct.toFixed(2)}% All-Time
                </div>
            </div>

            {/* Stats grid */}
            <div className="grid-3 gap-12">
                <div className="text-center">
                    <div className="stat-label">WIN RATE</div>
                    <div className="stat-value xs" style={{ color: wrColor }}>{stats.win_rate.toFixed(1)}%</div>
                    <div className="stat-sub">{stats.winning_trades}W / {stats.losing_trades}L</div>
                </div>
                <div className="text-center">
                    <div className="stat-label">SHARPE</div>
                    <div className="stat-value xs" style={{ color: stats.sharpe_ratio > 0 ? 'var(--positive)' : 'var(--negative)' }}>
                        {stats.sharpe_ratio.toFixed(2)}
                    </div>
                </div>
                <div className="text-center">
                    <div className="stat-label">DRAWDOWN</div>
                    <div className="stat-value xs text-negative">
                        {stats.max_drawdown_pct.toFixed(1)}%
                    </div>
                </div>
            </div>

            <div className="grid-3 gap-12 mt-12">
                <div className="text-center">
                    <div className="stat-label">TRADES</div>
                    <div className="stat-value xs">{stats.total_trades}</div>
                </div>
                <div className="text-center">
                    <div className="stat-label">KELLY</div>
                    <div className="stat-value xs mono">{(stats.kelly_fraction * 100).toFixed(1)}%</div>
                </div>
                <div className="text-center">
                    <div className="stat-label">PROFIT F.</div>
                    <div className="stat-value xs" style={{ color: stats.profit_factor > 1 ? 'var(--positive)' : 'var(--negative)' }}>
                        {stats.profit_factor === Infinity ? '∞' : stats.profit_factor.toFixed(2)}
                    </div>
                </div>
            </div>
        </div>
    );
}

/* ─── Equity Curve ───────────────────────────────── */
function EquityCurve({ points }: { points: Array<{ timestamp: string; balance: number }> }) {
    if (points.length < 2) {
        return (
            <div className="card">
                <div className="card-header">
                    <span className="card-title"><IconLineChart size={14} style={{ marginRight: 5, verticalAlign: -2 }} /> Equity Curve</span>
                </div>
                <div className="empty-state p-30">
                    <div className="text-32"><IconLineChart size={32} style={{ color: 'var(--text-3)' }} /></div>
                    <div className="text-3 mt-8">Waiting for trade data...</div>
                </div>
            </div>
        );
    }

    const data = points.map(p => ({
        time: new Date(p.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' }),
        balance: p.balance,
    }));

    const start = points[0].balance;
    const gradientId = data[data.length - 1].balance >= start ? 'eqGradUp' : 'eqGradDown';
    const lineColor = data[data.length - 1].balance >= start ? '#34D399' : '#F87171';

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title"><IconLineChart size={14} style={{ marginRight: 5, verticalAlign: -2 }} /> Equity Curve</span>
            </div>
            <ResponsiveContainer width="100%" height={240}>
                <AreaChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                    <defs>
                        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor={lineColor} stopOpacity={0.25} />
                            <stop offset="100%" stopColor={lineColor} stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <XAxis dataKey="time" tick={{ fill: '#566A84', fontSize: 10 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: '#566A84', fontSize: 10 }} axisLine={false} tickLine={false}
                        tickFormatter={(v: number) => `$${(v / 1000).toFixed(1)}k`} width={55} />
                    <Tooltip
                        contentStyle={{
                            background: 'rgba(13,17,23,0.95)',
                            border: '1px solid rgba(255,255,255,0.08)',
                            borderRadius: 8,
                            fontFamily: 'var(--font-mono)',
                            fontSize: 12,
                        }}
                        formatter={(v: number | undefined) => [`$${(v ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2 })}`, 'Balance']}
                    />
                    <Area type="monotone" dataKey="balance" stroke={lineColor} strokeWidth={2}
                        fill={`url(#${gradientId})`} dot={false} animationDuration={800} />
                </AreaChart>
            </ResponsiveContainer>
        </div>
    );
}

/* ─── Trade History ──────────────────────────────── */
function TradeHistory({ trades }: { trades: Array<Record<string, unknown>> }) {
    if (trades.length === 0) {
        return (
            <div className="card">
                <div className="card-header">
                    <span className="card-title"><IconList size={14} style={{ marginRight: 5, verticalAlign: -2 }} /> Trade History</span>
                </div>
                <div className="empty-state p-20">No trades yet</div>
            </div>
        );
    }

    return (
        <div className="card" style={{ padding: 0 }}>
            <div className="card-header" style={{ padding: '16px 20px 8px' }}>
                <span className="card-title"><IconList size={14} style={{ marginRight: 5, verticalAlign: -2 }} /> Trade History</span>
                <span className="text-12 text-3">{trades.length} trades</span>
            </div>
            <div className="table-scroll">
                <table className="data-table">
                    <thead>
                        <tr>
                            <th>Dir</th>
                            <th>Entry</th>
                            <th>Exit</th>
                            <th>PnL</th>
                            <th>Reason</th>
                        </tr>
                    </thead>
                    <tbody>
                        {trades.slice().reverse().map((t, i) => {
                            const pnl = t.pnl_usd as number || 0;
                            const pnlPct = t.pnl_pct as number || 0;
                            const pnlColor = pnl >= 0 ? 'var(--positive)' : 'var(--negative)';
                            return (
                                <tr key={i}>
                                    <td>
                                        <span className={`badge ${t.direction === 'LONG' ? 'badge-long' : 'badge-short'}`} style={{ padding: '2px 8px', fontSize: 11 }}>
                                            {t.direction === 'LONG' ? '↑' : '↓'}
                                        </span>
                                    </td>
                                    <td className="tp-sl-row">
                                        ${Number(t.entry_price).toLocaleString(undefined, { minimumFractionDigits: 0 })}
                                    </td>
                                    <td className="tp-sl-row">
                                        ${Number(t.exit_price).toLocaleString(undefined, { minimumFractionDigits: 0 })}
                                    </td>
                                    <td style={{ color: pnlColor, fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 600 }}>
                                        ${pnl.toFixed(2)} ({pnlPct > 0 ? '+' : ''}{pnlPct.toFixed(1)}%)
                                    </td>
                                    <td className="text-11 text-3">{String(t.close_reason || '—')}</td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

/* ─── Page ───────────────────────────────────────── */
export default function PaperTrading() {
    // Granular selectors — only re-render when these slices change
    const rawPositions = useLivePositions();
    const rawStats = useLiveStats();
    const botRunning = useLiveBotStatus();
    // Large/infrequent data stays on REST polling
    const { data: tradeData } = useApi<TradeData>('/api/trade-history?limit=50', 10000);
    const { data: equityData } = useApi<EquityData>('/api/equity-history', 15000);

    const [loading, setLoading] = useState<string | null>(null);

    const positions = (rawPositions || []) as unknown as Position[];
    const stats = rawStats as unknown as StatsData | null;
    const trades = tradeData?.trades || [];
    const equityPoints = equityData?.points || [];

    const doAction = async (action: string, body?: object) => {
        setLoading(action);
        try {
            await apiPost(`/api/${action}`, body);
            // WS will auto-push updated state within 2s — no manual refresh needed
        } catch (e) {
            console.error(e);
        }
        setLoading(null);
    };

    return (
        <div className="animate-in">
            {/* Header */}
            <div className="page-header">
                <div className="page-title">Paper Trading</div>
                <div className="page-subtitle">
                    <span className="status-dot" style={{ background: botRunning ? 'var(--positive)' : 'var(--negative)' }} /> {botRunning ? 'Auto-Trading Active' : 'Standby'} · {positions.length} Open Positions
                </div>
            </div>

            {/* Controls */}
            <div className="controls-row animate-in animate-in-1">
                {botRunning ? (
                    <button className="btn btn-danger" onClick={() => doAction('bot/stop')} disabled={loading !== null}>
                        <IconSquare size={14} style={{ marginRight: 5, verticalAlign: -2 }} /> Stop Bot
                    </button>
                ) : (
                    <button className="btn btn-primary" onClick={() => doAction('bot/start')} disabled={loading !== null}>
                        <IconPlay size={14} style={{ marginRight: 5, verticalAlign: -2 }} /> Start Bot
                    </button>
                )}
                <button className="btn btn-success" onClick={() => doAction('trade/open', { direction: 'LONG' })} disabled={loading !== null}>
                    <IconArrowUp size={14} style={{ marginRight: 5, verticalAlign: -2 }} /> Manual Long
                </button>
                <button className="btn btn-danger" onClick={() => doAction('trade/open', { direction: 'SHORT' })} disabled={loading !== null}>
                    <IconArrowDown size={14} style={{ marginRight: 5, verticalAlign: -2 }} /> Manual Short
                </button>
                {positions.length > 0 && (
                    <button className="btn btn-warning" onClick={() => doAction('trade/close-all')} disabled={loading !== null}>
                        <IconXCircle size={14} style={{ marginRight: 5, verticalAlign: -2 }} /> Close All
                    </button>
                )}
            </div>

            {/* Positions */}
            {positions.length > 0 ? (
                <div className="positions-stack">
                    {positions.map(pos => (
                        <PositionCard
                            key={pos.index}
                            pos={pos}
                            onClose={() => doAction('trade/close', { position_index: pos.index, reason: 'MANUAL' })}
                        />
                    ))}
                </div>
            ) : (
                <div className="card animate-in animate-in-2 text-center p-32 mb-20">
                    <div className="text-32">💤</div>
                    <div className="text-3">No open positions — start the bot or trade manually</div>
                </div>
            )}

            {/* Stats + Equity */}
            <div className="grid-2 animate-in animate-in-3 mb-20">
                <StatsPanel stats={stats} />
                <EquityCurve points={equityPoints} />
            </div>

            {/* Trade History */}
            <div className="animate-in animate-in-4">
                <TradeHistory trades={trades} />
            </div>
        </div>
    );
}
