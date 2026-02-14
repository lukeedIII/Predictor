import { useApi } from '../hooks/useApi';

type HistoryEntry = {
    timestamp: string;
    accuracy: number | null;
    prev_accuracy: number | null;
    delta: number | null;
    trend: string;
    streak: number;
    label: string;
    data_source: string;
    feature_count: number;
};

type HistorySummary = {
    total_retrains: number;
    best_accuracy: number;
    worst_accuracy: number;
    latest_accuracy: number | null;
    avg_delta: number | null;
    positive_retrains: number;
    negative_retrains: number;
    neutral_retrains: number;
    current_streak: number;
};

type HistoryData = {
    entries: HistoryEntry[];
    summary: HistorySummary | null;
};

function timeAgo(iso: string): string {
    const diff = Date.now() - new Date(iso).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ${mins % 60}m ago`;
    return `${Math.floor(hrs / 24)}d ago`;
}

function deltaColor(d: number | null): string {
    if (d == null) return 'var(--text-2)';
    if (d > 0.5) return '#00E676';
    if (d > 0) return '#69F0AE';
    if (d < -0.5) return '#FF5252';
    if (d < 0) return '#FF8A80';
    return 'var(--text-2)';
}

function accColor(acc: number): string {
    if (acc >= 55) return '#00E676';
    if (acc >= 52) return '#69F0AE';
    if (acc >= 50) return '#FFD740';
    return '#FF8A80';
}

function phaseTag(label: string): { text: string; color: string } {
    if (label.toLowerCase().includes('phase 1')) return { text: 'P1', color: '#7C4DFF' };
    if (label.toLowerCase().includes('phase 2')) return { text: 'P2', color: '#448AFF' };
    if (label.toLowerCase().includes('phase 3')) return { text: 'P3', color: '#00BCD4' };
    if (label.toLowerCase().includes('phase 4')) return { text: 'P4', color: '#00E676' };
    if (label.toLowerCase().includes('scheduled')) return { text: 'SCH', color: '#FFD740' };
    return { text: 'TR', color: 'var(--text-2)' };
}

export default function TrainingLog() {
    const { data } = useApi<HistoryData>('/api/retrain-history', 15_000);

    const entries = data?.entries ?? [];
    const summary = data?.summary;

    const bestAcc = summary?.best_accuracy ?? 50;
    const worstAcc = summary?.worst_accuracy ?? 50;
    const range = Math.max(bestAcc - worstAcc, 1);

    return (
        <div className="card card-compact animate-in">
            <div className="card-header">
                <span className="card-title">Training Log</span>
                {summary && (
                    <span className="mono" style={{ fontSize: 10, color: 'var(--text-2)' }}>
                        {summary.total_retrains} sessions
                    </span>
                )}
            </div>
            <div className="flex-col gap-6" style={{ fontSize: 12 }}>
                {/* ── Summary Stats ── */}
                {summary && (
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(3, 1fr)',
                        gap: 6,
                        padding: '6px 0',
                    }}>
                        {/* Best */}
                        <div style={{
                            padding: '8px 10px',
                            background: 'rgba(0,230,118,0.06)',
                            borderRadius: 6,
                            border: '1px solid rgba(0,230,118,0.12)',
                            textAlign: 'center',
                        }}>
                            <div style={{ fontSize: 9, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 2 }}>Best</div>
                            <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: '#00E676' }}>
                                {summary.best_accuracy.toFixed(1)}%
                            </div>
                        </div>
                        {/* Latest */}
                        <div style={{
                            padding: '8px 10px',
                            background: 'rgba(255,255,255,0.03)',
                            borderRadius: 6,
                            border: '1px solid var(--border)',
                            textAlign: 'center',
                        }}>
                            <div style={{ fontSize: 9, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 2 }}>Latest</div>
                            <div className="mono" style={{
                                fontSize: 16,
                                fontWeight: 700,
                                color: summary.latest_accuracy != null ? accColor(summary.latest_accuracy) : 'var(--text-2)',
                            }}>
                                {summary.latest_accuracy != null ? `${summary.latest_accuracy.toFixed(1)}%` : '—'}
                            </div>
                        </div>
                        {/* Trend */}
                        <div style={{
                            padding: '8px 10px',
                            background: 'rgba(255,255,255,0.03)',
                            borderRadius: 6,
                            border: '1px solid var(--border)',
                            textAlign: 'center',
                        }}>
                            <div style={{ fontSize: 9, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 2 }}>Trend</div>
                            <div style={{ fontSize: 13, fontWeight: 600 }}>
                                <span style={{ color: '#00E676' }}>{summary.positive_retrains}↑</span>
                                {' '}
                                <span style={{ color: 'var(--text-2)', fontSize: 10 }}>·</span>
                                {' '}
                                <span style={{ color: '#FFD740' }}>{summary.neutral_retrains}→</span>
                                {' '}
                                <span style={{ color: 'var(--text-2)', fontSize: 10 }}>·</span>
                                {' '}
                                <span style={{ color: '#FF5252' }}>{summary.negative_retrains}↓</span>
                            </div>
                        </div>
                    </div>
                )}

                {/* ── Streak alert ── */}
                {summary && Math.abs(summary.current_streak) >= 3 && (
                    <div style={{
                        padding: '4px 8px',
                        borderRadius: 4,
                        fontSize: 11,
                        background: summary.current_streak > 0 ? 'rgba(0,230,118,0.08)' : 'rgba(255,82,82,0.08)',
                        color: summary.current_streak > 0 ? '#69F0AE' : '#FF8A80',
                        textAlign: 'center',
                    }}>
                        {summary.current_streak > 0
                            ? `🔥 ${summary.current_streak}× improving streak`
                            : `⚠️ ${Math.abs(summary.current_streak)}× declining streak`}
                    </div>
                )}

                {/* ── Training entries ── */}
                {entries.length === 0 ? (
                    <div style={{ color: 'var(--text-2)', textAlign: 'center', padding: 16 }}>
                        <div style={{ fontSize: 20, marginBottom: 4 }}>⏳</div>
                        <div style={{ fontSize: 11 }}>Waiting for first training session...</div>
                    </div>
                ) : (
                    <div className="flex-col gap-4" style={{ maxHeight: 320, overflowY: 'auto' }}>
                        {entries.map((e, i) => {
                            const tag = phaseTag(e.label);
                            const acc = e.accuracy ?? 0;
                            const barWidth = range > 0.1
                                ? ((acc - worstAcc) / range) * 100
                                : 100;
                            const isLatest = i === 0;

                            return (
                                <div key={i} style={{
                                    padding: '8px 10px',
                                    borderRadius: 8,
                                    background: isLatest
                                        ? 'rgba(59,130,246,0.06)'
                                        : 'rgba(255,255,255,0.02)',
                                    border: isLatest
                                        ? '1px solid rgba(59,130,246,0.15)'
                                        : '1px solid transparent',
                                    transition: 'background 0.2s',
                                }}>
                                    {/* Row 1: Phase tag + Accuracy + Delta + Time */}
                                    <div style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 8,
                                        marginBottom: 6,
                                    }}>
                                        {/* Phase tag */}
                                        <span style={{
                                            fontSize: 9,
                                            fontWeight: 700,
                                            color: tag.color,
                                            background: `${tag.color}15`,
                                            padding: '2px 5px',
                                            borderRadius: 3,
                                            letterSpacing: 0.3,
                                            fontFamily: 'var(--font-mono)',
                                        }}>
                                            {tag.text}
                                        </span>

                                        {/* Accuracy */}
                                        <span className="mono" style={{
                                            fontSize: isLatest ? 15 : 13,
                                            fontWeight: isLatest ? 700 : 500,
                                            color: accColor(acc),
                                        }}>
                                            {acc > 0 ? `${acc.toFixed(1)}%` : 'N/A'}
                                        </span>

                                        {/* Delta */}
                                        {e.delta != null && e.delta !== 0 && (
                                            <span className="mono" style={{
                                                fontSize: 10,
                                                color: deltaColor(e.delta),
                                                fontWeight: 500,
                                            }}>
                                                {e.delta > 0 ? '+' : ''}{e.delta.toFixed(2)}%
                                            </span>
                                        )}

                                        {isLatest && (
                                            <span style={{
                                                fontSize: 8,
                                                color: 'var(--accent)',
                                                background: 'var(--accent-dim)',
                                                padding: '1px 5px',
                                                borderRadius: 3,
                                                fontWeight: 600,
                                                letterSpacing: 0.4,
                                                textTransform: 'uppercase',
                                            }}>
                                                latest
                                            </span>
                                        )}

                                        {/* Time */}
                                        <span className="mono" style={{
                                            fontSize: 10,
                                            color: 'var(--text-2)',
                                            marginLeft: 'auto',
                                            whiteSpace: 'nowrap',
                                        }}>
                                            {timeAgo(e.timestamp)}
                                        </span>
                                    </div>

                                    {/* Row 2: Progress bar */}
                                    <div style={{
                                        height: 3,
                                        borderRadius: 2,
                                        background: 'rgba(255,255,255,0.06)',
                                        marginBottom: 5,
                                        overflow: 'hidden',
                                    }}>
                                        <div style={{
                                            height: '100%',
                                            width: `${Math.max(5, barWidth)}%`,
                                            borderRadius: 2,
                                            background: `linear-gradient(90deg, ${accColor(acc)}66, ${accColor(acc)})`,
                                            transition: 'width 0.4s ease',
                                        }} />
                                    </div>

                                    {/* Row 3: Label + meta */}
                                    <div style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 6,
                                        fontSize: 10,
                                        color: 'var(--text-2)',
                                    }}>
                                        <span style={{ flex: 1 }}>{e.label}</span>
                                        <span className="mono" style={{
                                            fontSize: 9,
                                            padding: '1px 4px',
                                            background: 'rgba(255,255,255,0.04)',
                                            borderRadius: 3,
                                        }}>
                                            {e.data_source}
                                        </span>
                                        <span className="mono" style={{
                                            fontSize: 9,
                                            padding: '1px 4px',
                                            background: 'rgba(255,255,255,0.04)',
                                            borderRadius: 3,
                                        }}>
                                            {e.feature_count}f
                                        </span>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>
        </div>
    );
}
