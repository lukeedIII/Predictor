import { useLiveBotStatus, useLivePositions, useLiveStats } from '../stores/liveStore';
import { useApi, apiPost } from '../hooks/useApi';
import { useState } from 'react';
import { IconPlay, IconStop, IconRefresh, IconArrowUp, IconArrowDown } from '../components/Icons';
import { toast } from '../stores/toastStore';

type Trade = {
    id?: number;
    direction: string;
    entry_price: number;
    exit_price?: number;
    pnl?: number;
    pnl_usd?: number;
    pnl_pct?: number;
    timestamp?: string;
    exit_time?: string;
    close_reason?: string;
};

export default function PaperTrading() {
    const botRunning = useLiveBotStatus();
    const positions = useLivePositions() as any[];
    const stats = useLiveStats() as Record<string, any> | null;
    const { data: history } = useApi<{ trades: Trade[] }>('/api/paper/trades', 10_000);
    const trades = history?.trades ?? [];
    const [busy, setBusy] = useState(false);

    const act = async (action: string) => {
        setBusy(true);
        try {
            await apiPost(`/api/paper/${action}`);
            toast.success(`Bot ${action}`);
        } catch (e: any) {
            toast.error(e.message);
        } finally {
            setBusy(false);
        }
    };

    const fmtMoney = (v: number) => `$${v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    const fmtPct = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;

    // Stats
    const balance = stats?.balance ?? 10000;
    const available = stats?.available_balance ?? balance;
    const marginInUse = stats?.margin_in_use ?? 0;
    const totalPnl = stats?.total_pnl ?? 0;
    const unrealizedPnl = stats?.unrealized_pnl ?? 0;
    const winRate = stats?.win_rate ?? 0;
    const sharpe = stats?.sharpe_ratio ?? 0;
    const maxDD = stats?.max_drawdown_pct ?? 0;
    const profitFactor = stats?.profit_factor ?? 0;
    const totalTrades = stats?.total_trades ?? 0;
    const totalFees = stats?.total_fees ?? 0;
    const kelly = stats?.kelly_fraction ?? 0.02;
    const leverage = stats?.leverage ?? 10;
    const posCount = stats?.positions_count ?? positions.length;
    const maxConcurrent = stats?.max_concurrent ?? 3;

    return (
        <div className="flex-col gap-10">
            {/* ─── Capital Management Header ──────── */}
            <div className="card card-compact animate-in">
                {/* Row 1: Controls + Balance Breakdown */}
                <div className="trading-controls" style={{ flexWrap: 'wrap', gap: 8 }}>
                    {!botRunning ? (
                        <button className="btn btn-success btn-sm" onClick={() => act('start')} disabled={busy}>
                            <IconPlay style={{ width: 12, height: 12 }} /> Start
                        </button>
                    ) : (
                        <button className="btn btn-danger btn-sm" onClick={() => act('stop')} disabled={busy}>
                            <IconStop style={{ width: 12, height: 12 }} /> Stop
                        </button>
                    )}
                    <button className="btn btn-sm" onClick={() => act('reset')} disabled={busy}>
                        <IconRefresh style={{ width: 12, height: 12 }} /> Reset
                    </button>

                    <div style={{ flex: 1 }} />

                    {/* Capital Breakdown Pills */}
                    <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
                        <div style={{
                            display: 'flex', gap: 2, alignItems: 'center',
                            background: 'rgba(255,255,255,0.04)', borderRadius: 8, padding: '4px 10px',
                            fontSize: 11, color: 'var(--text-2)'
                        }}>
                            <span>Available</span>
                            <span className="mono" style={{ color: 'var(--text-1)', fontWeight: 600, marginLeft: 4 }}>
                                {fmtMoney(available)}
                            </span>
                        </div>
                        <div style={{
                            display: 'flex', gap: 2, alignItems: 'center',
                            background: marginInUse > 0 ? 'rgba(59,130,246,0.08)' : 'rgba(255,255,255,0.04)',
                            borderRadius: 8, padding: '4px 10px', fontSize: 11, color: 'var(--text-2)'
                        }}>
                            <span>In Trade</span>
                            <span className="mono" style={{
                                color: marginInUse > 0 ? 'var(--blue)' : 'var(--text-2)',
                                fontWeight: 600, marginLeft: 4
                            }}>
                                {fmtMoney(marginInUse)}
                            </span>
                        </div>
                        <div className="balance-pill">{fmtMoney(balance)}</div>
                    </div>

                    <span className={`badge ${botRunning ? 'badge-long' : 'badge-neutral'}`}>
                        {botRunning ? 'LIVE' : 'IDLE'}
                    </span>
                </div>

                {/* Row 2: Config bar */}
                <div style={{
                    display: 'flex', gap: 16, padding: '6px 12px 4px',
                    fontSize: 11, color: 'var(--text-3)', borderTop: '1px solid rgba(255,255,255,0.04)'
                }}>
                    <span>Leverage: <span className="mono" style={{ color: 'var(--text-2)' }}>{leverage}x</span></span>
                    <span>Kelly: <span className="mono" style={{ color: 'var(--text-2)' }}>{(kelly * 100).toFixed(1)}%</span></span>
                    <span>Slots: <span className="mono" style={{ color: 'var(--text-2)' }}>{posCount} / {maxConcurrent}</span></span>
                    <span>Fees Paid: <span className="mono" style={{ color: 'var(--text-2)' }}>{fmtMoney(totalFees)}</span></span>
                </div>
            </div>

            {/* ─── Stats Grid (2x4) ──────────────── */}
            <div className="grid grid-4 stagger">
                <div className="card card-compact">
                    <div className="stat-label">Realized P&L</div>
                    <div className={`stat-value sm mono ${totalPnl >= 0 ? 'text-positive' : 'text-negative'}`}>
                        {fmtMoney(totalPnl)}
                    </div>
                    <div className="stat-sub">{totalTrades} trades</div>
                </div>
                <div className="card card-compact">
                    <div className="stat-label">Unrealized P&L</div>
                    <div className={`stat-value sm mono ${unrealizedPnl >= 0 ? 'text-positive' : 'text-negative'}`}>
                        {fmtMoney(unrealizedPnl)}
                    </div>
                    <div className="stat-sub">{posCount} open</div>
                </div>
                <div className="card card-compact">
                    <div className="stat-label">Win Rate</div>
                    <div className={`stat-value sm mono ${winRate >= 50 ? 'text-positive' : winRate > 0 ? 'text-negative' : ''}`}>
                        {winRate.toFixed(1)}%
                    </div>
                </div>
                <div className="card card-compact">
                    <div className="stat-label">Sharpe Ratio</div>
                    <div className={`stat-value sm mono ${sharpe >= 0 ? 'text-positive' : 'text-negative'}`}>
                        {sharpe.toFixed(2)}
                    </div>
                </div>
                <div className="card card-compact">
                    <div className="stat-label">Max Drawdown</div>
                    <div className="stat-value sm mono text-negative">{fmtPct(-Math.abs(maxDD))}</div>
                </div>
                <div className="card card-compact">
                    <div className="stat-label">Profit Factor</div>
                    <div className={`stat-value sm mono ${profitFactor >= 1 ? 'text-positive' : profitFactor > 0 ? 'text-negative' : ''}`}>
                        {profitFactor === Infinity ? '∞' : profitFactor.toFixed(2)}
                    </div>
                </div>
                <div className="card card-compact">
                    <div className="stat-label">Total Fees</div>
                    <div className="stat-value sm mono" style={{ color: 'var(--text-2)' }}>
                        {fmtMoney(totalFees)}
                    </div>
                </div>
                <div className="card card-compact">
                    <div className="stat-label">Positions</div>
                    <div className="stat-value sm mono">{posCount} / {maxConcurrent}</div>
                </div>
            </div>

            {/* ─── Open Positions ───────────────── */}
            <div className="card animate-in">
                <div className="card-header">
                    <span className="card-title">Open Positions</span>
                    <span style={{ fontSize: 11, color: 'var(--text-2)' }}>
                        {posCount} / {maxConcurrent} slots
                    </span>
                </div>
                {positions.length === 0 ? (
                    <div className="empty-state" style={{ padding: 24 }}>
                        <span style={{ fontSize: 12, color: 'var(--text-3)' }}>
                            {botRunning ? 'Waiting for signal...' : 'Bot is idle — press Start to begin trading'}
                        </span>
                    </div>
                ) : (
                    <div className="table-wrap">
                        <table>
                            <thead>
                                <tr>
                                    <th>Dir</th>
                                    <th>Entry</th>
                                    <th>Size</th>
                                    <th>Margin</th>
                                    <th>Lev</th>
                                    <th>P&L</th>
                                    <th>P&L %</th>
                                    <th>TP</th>
                                    <th>SL</th>
                                    <th>Liq</th>
                                    <th>Hold</th>
                                </tr>
                            </thead>
                            <tbody>
                                {positions.map((p, i) => {
                                    const isLong = (p.direction ?? p.side ?? '').toLowerCase().includes('long');
                                    const pnl = p.unrealized_pnl ?? p.pnl_usd ?? p.pnl ?? 0;
                                    const pnlPct = p.unrealized_pnl_pct ?? p.pnl_pct ?? 0;
                                    const margin = p.margin ?? (p.size_usd ? p.size_usd / (p.leverage ?? leverage) : 0);
                                    const posLev = p.leverage ?? leverage;
                                    const liqPrice = p.liquidation_price ?? p.liq_price ?? null;

                                    // Hold time formatting
                                    let holdStr = '—';
                                    if (p.entry_time || p.hold_minutes) {
                                        const mins = p.hold_minutes ?? Math.floor((Date.now() - new Date(p.entry_time).getTime()) / 60000);
                                        if (mins >= 60) holdStr = `${Math.floor(mins / 60)}h ${mins % 60}m`;
                                        else holdStr = `${mins}m`;
                                    }

                                    return (
                                        <tr key={i}>
                                            <td>
                                                <span className={`badge ${isLong ? 'badge-long' : 'badge-short'}`}>
                                                    {isLong ? <IconArrowUp style={{ width: 10, height: 10 }} /> : <IconArrowDown style={{ width: 10, height: 10 }} />}
                                                    {isLong ? 'LONG' : 'SHORT'}
                                                </span>
                                            </td>
                                            <td className="mono">{fmtMoney(p.entry_price ?? 0)}</td>
                                            <td className="mono">{p.size_usd ? fmtMoney(p.size_usd) : '—'}</td>
                                            <td className="mono" style={{ color: 'var(--blue)' }}>{fmtMoney(margin)}</td>
                                            <td className="mono">{posLev}x</td>
                                            <td className={`mono ${pnl >= 0 ? 'text-positive' : 'text-negative'}`}>{fmtMoney(pnl)}</td>
                                            <td className={`mono ${pnlPct >= 0 ? 'text-positive' : 'text-negative'}`}>{fmtPct(pnlPct)}</td>
                                            <td className="mono text-positive">{p.tp_price ? fmtMoney(p.tp_price) : '—'}</td>
                                            <td className="mono text-negative">{p.sl_price ? fmtMoney(p.sl_price) : '—'}</td>
                                            <td className="mono" style={{ color: 'var(--orange, #f59e0b)' }}>
                                                {liqPrice ? fmtMoney(liqPrice) : '—'}
                                            </td>
                                            <td style={{ fontSize: 11, color: 'var(--text-2)' }}>{holdStr}</td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            {/* ─── Trade History ────────────────── */}
            <div className="card animate-in">
                <div className="card-header">
                    <span className="card-title">Trade History</span>
                    <span style={{ fontSize: 11, color: 'var(--text-2)' }}>{trades.length} trades</span>
                </div>
                {trades.length === 0 ? (
                    <div className="empty-state" style={{ padding: 24 }}>
                        <span style={{ fontSize: 12, color: 'var(--text-3)' }}>No trades yet</span>
                    </div>
                ) : (
                    <div className="table-wrap" style={{ maxHeight: 350, overflowY: 'auto' }}>
                        <table>
                            <thead>
                                <tr>
                                    <th>Dir</th>
                                    <th>Entry</th>
                                    <th>Exit</th>
                                    <th>P&L</th>
                                    <th>%</th>
                                    <th>Reason</th>
                                    <th>Time</th>
                                </tr>
                            </thead>
                            <tbody>
                                {trades.slice().reverse().map((t, i) => {
                                    const isLong = (t.direction ?? '').toLowerCase().includes('long');
                                    const pnl = t.pnl_usd ?? t.pnl ?? 0;
                                    const reason = t.close_reason ?? '—';

                                    // Color-code close reasons
                                    const reasonColor = reason.includes('PROFIT') ? 'var(--green)'
                                        : reason.includes('STOP') || reason.includes('LIQUIDAT') ? 'var(--red)'
                                            : reason.includes('TRAIL') ? 'var(--yellow, #eab308)'
                                                : 'var(--text-2)';

                                    return (
                                        <tr key={i}>
                                            <td>
                                                <span className={`badge ${isLong ? 'badge-long' : 'badge-short'}`} style={{ fontSize: 9 }}>
                                                    {isLong ? 'L' : 'S'}
                                                </span>
                                            </td>
                                            <td className="mono">{fmtMoney(t.entry_price)}</td>
                                            <td className="mono">{t.exit_price ? fmtMoney(t.exit_price) : '—'}</td>
                                            <td className={`mono ${pnl >= 0 ? 'text-positive' : 'text-negative'}`}>{fmtMoney(pnl)}</td>
                                            <td className={`mono ${pnl >= 0 ? 'text-positive' : 'text-negative'}`}>{t.pnl_pct ? fmtPct(t.pnl_pct) : '—'}</td>
                                            <td style={{ fontSize: 10, color: reasonColor, fontWeight: 500 }}>{reason}</td>
                                            <td style={{ fontSize: 11, color: 'var(--text-2)' }}>{t.exit_time ?? t.timestamp ?? '—'}</td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    );
}
