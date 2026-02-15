import { useApi } from '../hooks/useApi';

type Stats = {
    balance: number;
    starting_balance: number;
    available_balance: number;
    margin_in_use: number;
    total_pnl: number;
    total_gross_pnl: number;
    total_fees: number;
    total_pnl_pct: number;
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    win_rate: number;
    sharpe_ratio: number;
    net_sharpe_ratio: number;
    max_drawdown_pct: number;
    profit_factor: number | null;
    kelly_fraction: number;
    unrealized_pnl: number;
    circuit_breaker: boolean;
    position_open: boolean;
    positions_count: number;
    max_concurrent: number;
    leverage: number;
    current_streak: number;
    best_trade_pnl: number;
    worst_trade_pnl: number;
};

type Position = {
    index: number;
    direction: 'LONG' | 'SHORT';
    entry_price: number;
    size_usd: number;
    unrealized_pnl: number;
    unrealized_pnl_pct: number;
    elapsed_seconds: number;
    tp_price?: number;
    sl_price?: number;
    tp1_hit?: boolean;
};

type PositionsData = {
    positions: Position[];
    current_price: number;
};

/* ── helpers ──────────────────────────────────────────── */
function fmtUsd(n: number) {
    if (Math.abs(n) >= 1000) return `$${(n / 1000).toFixed(1)}K`;
    return `$${n.toFixed(2)}`;
}
function fmtPct(n: number) {
    const sign = n > 0 ? '+' : '';
    return `${sign}${n.toFixed(2)}%`;
}
function pnlColor(n: number) {
    if (n > 0) return '#00E676';
    if (n < 0) return '#FF5252';
    return 'var(--text-2)';
}
function elapsed(secs: number) {
    if (secs < 60) return `${Math.round(secs)}s`;
    if (secs < 3600) return `${Math.floor(secs / 60)}m`;
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    return `${h}h ${m}m`;
}

/* ── Thin bar ─────────────────────────────────────────── */
function Bar({ pct, color }: { pct: number; color: string }) {
    return (
        <div style={{
            height: 3, borderRadius: 2, background: 'rgba(255,255,255,0.06)',
            overflow: 'hidden', marginTop: 3,
        }}>
            <div style={{
                height: '100%', borderRadius: 2,
                width: `${Math.max(1, Math.min(100, pct))}%`,
                background: `linear-gradient(90deg, ${color}66, ${color})`,
                transition: 'width 0.6s ease',
            }} />
        </div>
    );
}

/* ═══════════════════════════════════════════════════════════
   MAIN
   ═══════════════════════════════════════════════════════════ */

export default function PaperStats() {
    const { data: stats } = useApi<Stats>('/api/stats', 5_000);
    const { data: posData } = useApi<PositionsData>('/api/positions', 3_000);

    if (!stats) {
        return (
            <div className="card card-compact animate-in" style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                height: '100%', fontSize: 11, color: 'var(--text-2)',
            }}>
                Loading stats…
            </div>
        );
    }

    const positions = posData?.positions ?? [];
    const curPrice = posData?.current_price ?? 0;
    const equity = stats.balance + stats.unrealized_pnl;
    const equityPnl = equity - stats.starting_balance;
    const equityPct = stats.starting_balance > 0 ? (equityPnl / stats.starting_balance) * 100 : 0;
    const marginPct = stats.balance > 0 ? (stats.margin_in_use / stats.balance) * 100 : 0;

    const sec = (children: React.ReactNode, highlight?: boolean) => (
        <div style={{
            padding: '10px 12px', borderRadius: 8,
            background: highlight ? 'rgba(0,230,118,0.04)' : 'rgba(255,255,255,0.02)',
            border: highlight ? '1px solid rgba(0,230,118,0.10)' : '1px solid var(--border)',
        }}>
            {children}
        </div>
    );

    return (
        <div className="card card-compact animate-in">
            {/* ── Header ── */}
            <div className="card-header">
                <span className="card-title">Paper Trading</span>
                <span className="mono" style={{ fontSize: 10, color: 'var(--text-2)' }}>
                    {stats.leverage}× leverage
                </span>
            </div>

            <div className="flex-col gap-6" style={{ fontSize: 12 }}>

                {/* ═══ Equity summary — 3 columns ═══ */}
                <div style={{
                    display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)',
                    gap: 6,
                }}>
                    {/* Balance */}
                    <div style={{
                        padding: '6px 8px', borderRadius: 6, textAlign: 'center',
                        background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border)',
                    }}>
                        <div style={{ fontSize: 9, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 2 }}>
                            Equity
                        </div>
                        <div className="mono" style={{
                            fontSize: 15, fontWeight: 700,
                            color: pnlColor(equityPnl),
                        }}>
                            {fmtUsd(equity)}
                        </div>
                    </div>

                    {/* P&L */}
                    <div style={{
                        padding: '6px 8px', borderRadius: 6, textAlign: 'center',
                        background: equityPnl >= 0 ? 'rgba(0,230,118,0.05)' : 'rgba(255,82,82,0.05)',
                        border: equityPnl >= 0 ? '1px solid rgba(0,230,118,0.12)' : '1px solid rgba(255,82,82,0.12)',
                    }}>
                        <div style={{ fontSize: 9, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 2 }}>
                            Total P&L
                        </div>
                        <div className="mono" style={{
                            fontSize: 15, fontWeight: 700,
                            color: pnlColor(equityPnl),
                        }}>
                            {fmtPct(equityPct)}
                        </div>
                    </div>

                    {/* Unrealized */}
                    <div style={{
                        padding: '6px 8px', borderRadius: 6, textAlign: 'center',
                        background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border)',
                    }}>
                        <div style={{ fontSize: 9, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 2 }}>
                            Unrealized
                        </div>
                        <div className="mono" style={{
                            fontSize: 15, fontWeight: 700,
                            color: pnlColor(stats.unrealized_pnl),
                        }}>
                            {fmtUsd(stats.unrealized_pnl)}
                        </div>
                    </div>
                </div>

                {/* ═══ Performance stats ═══ */}
                {sec(
                    <div style={{
                        display: 'grid', gridTemplateColumns: '1fr 1fr',
                        gap: '6px 16px', fontSize: 11,
                    }}>
                        {/* Win Rate */}
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span style={{ color: 'var(--text-2)' }}>Win Rate</span>
                            <span className="mono" style={{
                                fontWeight: 600,
                                color: stats.total_trades > 0
                                    ? (stats.win_rate >= 50 ? '#00E676' : '#FF5252')
                                    : 'var(--text-2)',
                            }}>
                                {stats.total_trades > 0 ? `${stats.win_rate.toFixed(0)}%` : '—'}
                            </span>
                        </div>

                        {/* Trades */}
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span style={{ color: 'var(--text-2)' }}>Trades</span>
                            <span className="mono" style={{ fontWeight: 600, color: 'var(--text-1)' }}>
                                <span style={{ color: '#00E676' }}>{stats.winning_trades}W</span>
                                {' / '}
                                <span style={{ color: '#FF5252' }}>{stats.losing_trades}L</span>
                            </span>
                        </div>

                        {/* Sharpe */}
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span style={{ color: 'var(--text-2)' }}>Sharpe</span>
                            <span className="mono" style={{
                                fontWeight: 600,
                                color: stats.sharpe_ratio > 1 ? '#00E676' : stats.sharpe_ratio > 0 ? '#FFD740' : '#FF5252',
                            }}>
                                {stats.sharpe_ratio.toFixed(2)}
                            </span>
                        </div>

                        {/* Max DD */}
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span style={{ color: 'var(--text-2)' }}>Max DD</span>
                            <span className="mono" style={{
                                fontWeight: 600,
                                color: stats.max_drawdown_pct > 10 ? '#FF5252' : stats.max_drawdown_pct > 5 ? '#FFD740' : '#00E676',
                            }}>
                                {stats.max_drawdown_pct.toFixed(1)}%
                            </span>
                        </div>

                        {/* Profit Factor */}
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span style={{ color: 'var(--text-2)' }}>Profit Factor</span>
                            <span className="mono" style={{
                                fontWeight: 600,
                                color: (stats.profit_factor ?? 0) > 1.5 ? '#00E676' : (stats.profit_factor ?? 0) > 1 ? '#FFD740' : 'var(--text-2)',
                            }}>
                                {stats.profit_factor != null ? stats.profit_factor.toFixed(2) : '—'}
                            </span>
                        </div>

                        {/* Streak */}
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span style={{ color: 'var(--text-2)' }}>Streak</span>
                            <span className="mono" style={{
                                fontWeight: 600,
                                color: stats.current_streak > 0 ? '#00E676' : stats.current_streak < 0 ? '#FF5252' : 'var(--text-2)',
                            }}>
                                {stats.current_streak > 0 ? `+${stats.current_streak}` : stats.current_streak}
                            </span>
                        </div>
                    </div>
                )}

                {/* ═══ Margin usage ═══ */}
                {sec(
                    <>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
                            <span style={{ color: 'var(--text-2)' }}>Margin Usage</span>
                            <span className="mono" style={{ fontSize: 10, color: 'var(--text-2)' }}>
                                {fmtUsd(stats.margin_in_use)} / {fmtUsd(stats.balance)}
                            </span>
                        </div>
                        <Bar pct={marginPct} color={marginPct > 80 ? '#FF5252' : marginPct > 50 ? '#FFD740' : '#6366f1'} />
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: 10, color: 'var(--text-2)' }}>
                            <span className="mono">{positions.length} / {stats.max_concurrent} slots</span>
                            {stats.circuit_breaker && (
                                <span style={{ color: '#FF5252', fontWeight: 700, fontSize: 9, letterSpacing: 0.5 }}>
                                    ⚠ CIRCUIT BREAKER
                                </span>
                            )}
                        </div>
                    </>
                )}

                {/* ═══ Open positions ═══ */}
                {positions.length > 0 && (
                    <div className="flex-col gap-4" style={{
                        maxHeight: 260, overflowY: 'auto',
                    }}>
                        {positions.map((pos) => {
                            const isLong = pos.direction === 'LONG';
                            const dirColor = isLong ? '#00E676' : '#FF5252';
                            const dirBg = isLong ? 'rgba(0,230,118,0.10)' : 'rgba(255,82,82,0.10)';
                            const pnlPct = pos.unrealized_pnl_pct;
                            const distToTp = pos.tp_price && curPrice
                                ? ((pos.tp_price - curPrice) / curPrice * 100) * (isLong ? 1 : -1)
                                : null;
                            const distToSl = pos.sl_price && curPrice
                                ? ((curPrice - pos.sl_price) / curPrice * 100) * (isLong ? 1 : -1)
                                : null;

                            return (
                                <div key={pos.index} style={{
                                    padding: '8px 10px', borderRadius: 8,
                                    background: 'rgba(255,255,255,0.02)',
                                    border: '1px solid var(--border)',
                                }}>
                                    {/* Row 1: Direction + P&L + Duration */}
                                    <div style={{
                                        display: 'flex', alignItems: 'center', gap: 8,
                                        marginBottom: 6,
                                    }}>
                                        <span style={{
                                            fontSize: 9, fontWeight: 700, color: dirColor,
                                            background: dirBg, padding: '2px 6px',
                                            borderRadius: 3, letterSpacing: 0.3,
                                            fontFamily: 'var(--font-mono)',
                                        }}>
                                            {pos.direction}
                                        </span>
                                        <span className="mono" style={{
                                            fontSize: 13, fontWeight: 700,
                                            color: pnlColor(pos.unrealized_pnl),
                                        }}>
                                            {fmtUsd(pos.unrealized_pnl)}
                                        </span>
                                        <span className="mono" style={{
                                            fontSize: 10,
                                            color: pnlColor(pnlPct),
                                        }}>
                                            ({fmtPct(pnlPct)})
                                        </span>
                                        <span className="mono" style={{
                                            marginLeft: 'auto', fontSize: 10,
                                            color: 'var(--text-2)',
                                        }}>
                                            {elapsed(pos.elapsed_seconds)}
                                        </span>
                                    </div>

                                    {/* Row 2: Entry + Size + TP/SL distances */}
                                    <div style={{
                                        display: 'flex', gap: 8, fontSize: 10,
                                        color: 'var(--text-2)', flexWrap: 'wrap',
                                    }}>
                                        <span className="mono">
                                            Entry {pos.entry_price.toLocaleString('en-US', { minimumFractionDigits: 0 })}
                                        </span>
                                        <span className="mono">{fmtUsd(pos.size_usd)}</span>
                                        {distToTp != null && (
                                            <span className="mono" style={{
                                                padding: '0 4px', borderRadius: 3,
                                                background: 'rgba(0,230,118,0.08)',
                                            }}>
                                                TP {distToTp.toFixed(1)}%{pos.tp1_hit ? ' ✓' : ''}
                                            </span>
                                        )}
                                        {distToSl != null && (
                                            <span className="mono" style={{
                                                padding: '0 4px', borderRadius: 3,
                                                background: 'rgba(255,82,82,0.08)',
                                            }}>
                                                SL {distToSl.toFixed(1)}%
                                            </span>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                )}

                {/* No positions message */}
                {positions.length === 0 && !stats.position_open && (
                    <div style={{
                        textAlign: 'center', padding: 12,
                        fontSize: 11, color: 'var(--text-2)',
                    }}>
                        No open positions
                    </div>
                )}

            </div>
        </div>
    );
}
