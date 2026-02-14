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

    // Stats values
    const balance = stats?.balance ?? stats?.current_balance ?? 10000;
    const totalPnl = stats?.total_pnl ?? 0;
    const winRate = stats?.win_rate ?? 0;
    const sharpe = stats?.sharpe_ratio ?? stats?.sharpe ?? 0;
    const maxDD = stats?.max_drawdown_pct ?? stats?.max_drawdown ?? 0;
    const profitFactor = stats?.profit_factor ?? 0;
    const totalTrades = stats?.total_trades ?? 0;

    return (
        <div className="flex-col gap-10">
            {/* ─── Controls ────────────────────── */}
            <div className="card card-compact animate-in">
                <div className="trading-controls">
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
                    <div className="balance-pill">
                        {fmtMoney(balance)}
                    </div>
                    <span className={`badge ${botRunning ? 'badge-long' : 'badge-neutral'}`}>
                        {botRunning ? 'LIVE' : 'IDLE'}
                    </span>
                </div>
            </div>

            {/* ─── Stats Grid ──────────────────── */}
            <div className="grid grid-3 stagger">
                <div className="card card-compact">
                    <div className="stat-label">Total P&L</div>
                    <div className={`stat-value sm mono ${totalPnl >= 0 ? 'text-positive' : 'text-negative'}`}>
                        {fmtMoney(totalPnl)}
                    </div>
                </div>
                <div className="card card-compact">
                    <div className="stat-label">Win Rate</div>
                    <div className="stat-value sm mono">{winRate.toFixed(1)}%</div>
                    <div className="stat-sub">{totalTrades} trades</div>
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
                    <div className="stat-value sm mono">{profitFactor.toFixed(2)}</div>
                </div>
                <div className="card card-compact">
                    <div className="stat-label">Balance</div>
                    <div className="stat-value sm mono">{fmtMoney(balance)}</div>
                </div>
            </div>

            {/* ─── Open Positions ───────────────── */}
            <div className="card animate-in">
                <div className="card-header">
                    <span className="card-title">Open Positions</span>
                    <span style={{ fontSize: 11, color: 'var(--text-2)' }}>{positions.length}</span>
                </div>
                {positions.length === 0 ? (
                    <div className="empty-state" style={{ padding: 20 }}>
                        <span style={{ fontSize: 12 }}>No open positions</span>
                    </div>
                ) : (
                    <div className="table-wrap">
                        <table>
                            <thead>
                                <tr>
                                    <th>Dir</th><th>Entry</th><th>Size</th><th>P&L</th><th>TP</th><th>SL</th>
                                </tr>
                            </thead>
                            <tbody>
                                {positions.map((p, i) => {
                                    const isLong = (p.direction ?? p.side ?? '').toLowerCase().includes('long');
                                    const pnl = p.unrealized_pnl ?? p.pnl_usd ?? p.pnl ?? 0;
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
                                            <td className={`mono ${pnl >= 0 ? 'text-positive' : 'text-negative'}`}>{fmtMoney(pnl)}</td>
                                            <td className="mono text-positive">{p.tp_price ? fmtMoney(p.tp_price) : '—'}</td>
                                            <td className="mono text-negative">{p.sl_price ? fmtMoney(p.sl_price) : '—'}</td>
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
                    <div className="empty-state" style={{ padding: 20 }}>
                        <span style={{ fontSize: 12 }}>No trades yet</span>
                    </div>
                ) : (
                    <div className="table-wrap" style={{ maxHeight: 300, overflowY: 'auto' }}>
                        <table>
                            <thead>
                                <tr>
                                    <th>Dir</th><th>Entry</th><th>Exit</th><th>P&L</th><th>%</th><th>Time</th>
                                </tr>
                            </thead>
                            <tbody>
                                {trades.slice().reverse().map((t, i) => {
                                    const isLong = (t.direction ?? '').toLowerCase().includes('long');
                                    const pnl = t.pnl_usd ?? t.pnl ?? 0;
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
