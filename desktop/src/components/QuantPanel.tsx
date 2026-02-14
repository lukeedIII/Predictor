import { useState, useCallback } from 'react';
import { useLiveQuant } from '../stores/liveStore';
import { IconActivity } from './Icons';

/* ═══════════════════════════════════════════════════════════════
   Quant Intelligence Panel — Premium Multi-Model Dashboard
   Surfaces all 16 institutional-grade quant models to the UI.
   ═══════════════════════════════════════════════════════════════ */

// ── Helpers ─────────────────────────────────────────────────────

type GaugeProps = {
    label: string; value: number; min?: number; max?: number;
    fmt?: (v: number) => string; color?: string; compact?: boolean;
};

function MiniGauge({ label, value, min = 0, max = 1, fmt, color, compact }: GaugeProps) {
    const pct = Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100));
    const display = fmt ? fmt(value) : value.toFixed(2);
    return (
        <div style={{ marginBottom: compact ? 5 : 8 }}>
            <div className="flex justify-between items-center" style={{ marginBottom: 2 }}>
                <span style={{ fontSize: 10.5, color: 'var(--text-1)' }}>{label}</span>
                <span className="mono" style={{ fontSize: 10.5, color: color ?? 'var(--text-0)' }}>{display}</span>
            </div>
            <div className="progress-track" style={{ height: compact ? 3 : 4 }}>
                <div className="progress-fill" style={{ width: `${pct}%`, background: color ?? 'var(--accent)' }} />
            </div>
        </div>
    );
}

function SectionHeader({ title, open, toggle, badge, badgeColor }: {
    title: string; open: boolean; toggle: () => void; badge?: string; badgeColor?: string;
}) {
    return (
        <button
            onClick={toggle}
            className="flex justify-between items-center"
            style={{
                width: '100%', background: 'none', border: 'none', cursor: 'pointer',
                padding: '6px 0', color: 'var(--text-0)', fontSize: 11, fontWeight: 600,
                letterSpacing: '0.02em',
            }}
        >
            <span className="flex items-center gap-6">
                <span style={{
                    display: 'inline-block', width: 14, textAlign: 'center',
                    color: 'var(--text-2)', transition: 'transform 150ms var(--ease)',
                    transform: open ? 'rotate(90deg)' : 'rotate(0deg)', fontSize: 10,
                }}>▶</span>
                {title}
            </span>
            {badge && (
                <span className="badge" style={{
                    background: `${badgeColor ?? 'var(--accent)'}18`,
                    color: badgeColor ?? 'var(--accent)',
                    fontSize: 9, padding: '1px 6px',
                }}>{badge}</span>
            )}
        </button>
    );
}

function Section({ title, open, toggle, badge, badgeColor, children }: {
    title: string; open: boolean; toggle: () => void;
    badge?: string; badgeColor?: string; children: React.ReactNode;
}) {
    return (
        <div style={{ borderBottom: '1px solid var(--border)', paddingBottom: open ? 6 : 0 }}>
            <SectionHeader title={title} open={open} toggle={toggle} badge={badge} badgeColor={badgeColor} />
            <div style={{
                maxHeight: open ? 500 : 0, overflow: 'hidden',
                transition: 'max-height 250ms var(--ease), opacity 200ms var(--ease)',
                opacity: open ? 1 : 0, paddingTop: open ? 4 : 0,
            }}>
                {children}
            </div>
        </div>
    );
}

/** 3-segment probability bar for HMM regime states */
function RegimeBar({ probs }: { probs: number[] }) {
    const labels = ['BULL', 'SIDE', 'BEAR'];
    const colors = ['var(--positive)', 'var(--accent)', 'var(--negative)'];
    const total = probs.reduce((a, b) => a + b, 0) || 1;
    return (
        <div style={{ marginTop: 4, marginBottom: 6 }}>
            <div className="flex" style={{ height: 6, borderRadius: 3, overflow: 'hidden', gap: 1 }}>
                {probs.map((p, i) => (
                    <div key={i} style={{
                        width: `${(p / total) * 100}%`, background: colors[i],
                        minWidth: 2, transition: 'width 300ms var(--ease)',
                    }} />
                ))}
            </div>
            <div className="flex justify-between" style={{ marginTop: 3 }}>
                {probs.map((p, i) => (
                    <span key={i} className="mono" style={{ fontSize: 9, color: colors[i] }}>
                        {labels[i]} {(p * 100).toFixed(0)}%
                    </span>
                ))}
            </div>
        </div>
    );
}

function SignalBadge({ signal }: { signal: { type: string; action: string; reason: string } }) {
    const colorMap: Record<string, string> = {
        REGIME: 'var(--accent)', OFI: '#A78BFA', VOL: 'var(--warning)',
        HESTON: 'var(--negative)', JUMP: '#F43F5E', MFDFA: '#06B6D4',
        RQA: '#8B5CF6',
    };
    const c = colorMap[signal.type] ?? 'var(--text-1)';
    return (
        <div className="flex items-center gap-6" style={{ marginBottom: 4 }}>
            <span style={{
                fontSize: 8.5, fontWeight: 700, borderRadius: 3,
                padding: '1px 5px', background: `${c}20`, color: c,
                fontFamily: 'var(--font-mono)', letterSpacing: '0.05em',
            }}>{signal.type}</span>
            <span style={{ fontSize: 10, color: 'var(--text-1)' }}>{signal.reason}</span>
        </div>
    );
}

// ── Main Component ──────────────────────────────────────────────

export default function QuantPanel() {
    const quant = useLiveQuant() as Record<string, any> | null;

    // Section open/close state — default: regime + basics open
    const [sections, setSections] = useState<Record<string, boolean>>({
        regime: true, volatility: false, orderflow: false, jumps: false,
        cycles: false, wavelets: false, fractals: false, rqa: false,
        signals: true,
    });
    const toggle = useCallback((key: string) =>
        setSections(s => ({ ...s, [key]: !s[key] }))
        , []);

    // ── Awaiting data state ──
    if (!quant) {
        return (
            <div className="card animate-in">
                <div className="card-header">
                    <span className="card-title">Quant Intelligence</span>
                    <IconActivity style={{ width: 14, height: 14, color: 'var(--text-2)' }} />
                </div>
                <div className="empty-state" style={{ padding: 20 }}>
                    <span style={{ fontSize: 12 }}>Awaiting data…</span>
                </div>
            </div>
        );
    }

    // ── Extract sections (supports both old raw format and new ui_summary) ──
    const regime = quant.regime ?? {};
    const regimeLabel = regime.label ?? regime.regime ?? 'UNKNOWN';
    const regimeConf = regime.confidence ?? 0;
    const regimeProbs = regime.probabilities ?? [0.33, 0.34, 0.33];

    const vol = quant.volatility ?? {};
    const garchForecast = vol.garch_forecast ?? vol.forecast ?? 0;
    const garchAsymmetry = vol.garch_asymmetry ?? vol.asymmetry ?? 0;
    const garchCurrent = vol.garch_current ?? vol.current ?? 0;
    const hestonLeverage = vol.heston_leverage ?? -0.7;
    const roughH = vol.rough_H ?? 0.5;
    const roughInterpretation = vol.rough_interpretation ?? 'UNKNOWN';
    const roughScore = vol.rough_score ?? 50;

    const of = quant.order_flow ?? {};
    const ofiSignal = of.signal ?? 'NEUTRAL';
    const ofiStrength = of.strength ?? 'WEAK';
    const ofiNormalized = of.normalized ?? 0;

    const jumps = quant.jumps ?? {};
    const jumpDetected = jumps.detected ?? false;
    const jumpProb = jumps.probability ?? 0;
    const jumpDirection = jumps.direction ?? 'NONE';
    const jumpRisk = jumps.risk_level ?? 'UNKNOWN';
    const batesStatus = jumps.bates_status ?? 'N/A';

    const cycles = quant.cycles ?? {};
    const cycleStrengths = cycles.strengths ?? [0, 0, 0];
    const hhtPeriod = cycles.hht_period_minutes ?? 0;

    const wav = quant.wavelets ?? {};
    const trendStrength = wav.trend_strength ?? 0.5;

    const fractal = quant.fractals ?? {};
    const mfDeltaH = fractal.multifractal_delta_h ?? 0;
    const mfInterp = fractal.multifractal_interpretation ?? 'UNKNOWN';
    const tdaComplexity = fractal.tda_complexity ?? 'UNKNOWN';
    const tdaPersistence = fractal.tda_persistence ?? 0;

    const rqa = quant.rqa ?? {};
    const rqaDet = rqa.determinism ?? 0;
    const rqaRR = rqa.recurrence_rate ?? 0;
    const rqaInterp = rqa.interpretation ?? 'UNKNOWN';

    const signals: Array<{ type: string; action: string; reason: string }> = quant.signals ?? [];

    const basics = quant.basics ?? {};
    const momentum = basics.momentum ?? quant.momentum ?? 0;
    const rsi = basics.rsi ?? quant.rsi ?? 50;
    const vwapDist = basics.vwap_dist ?? quant.vwap_dist ?? 0;
    const sharpe = basics.sharpe ?? quant.sharpe ?? 0;

    // ── Colors ──
    const regimeColor = regimeLabel === 'BULL' ? 'var(--positive)'
        : regimeLabel === 'BEAR' ? 'var(--negative)' : 'var(--accent)';
    const ofiColor = ofiSignal === 'BUY_PRESSURE' ? 'var(--positive)'
        : ofiSignal === 'SELL_PRESSURE' ? 'var(--negative)' : 'var(--text-2)';

    return (
        <div className="card animate-in">
            {/* ── Header ── */}
            <div className="card-header">
                <span className="card-title">Quant Intelligence</span>
                <div className="flex items-center gap-6">
                    {signals.length > 0 && (
                        <span className="badge" style={{
                            background: 'var(--warning-dim)', color: 'var(--warning)',
                            fontSize: 9, padding: '1px 6px',
                        }}>{signals.length} signal{signals.length > 1 ? 's' : ''}</span>
                    )}
                    <span className="badge" style={{ background: `${regimeColor}20`, color: regimeColor }}>
                        {regimeLabel}
                    </span>
                </div>
            </div>

            {/* ── Quick Metrics Strip ── */}
            <div className="flex gap-8" style={{ marginBottom: 8 }}>
                {[
                    { label: 'RSI', value: rsi.toFixed(0), color: rsi > 70 ? 'var(--negative)' : rsi < 30 ? 'var(--positive)' : 'var(--text-1)' },
                    { label: 'Mom', value: momentum.toFixed(3), color: momentum >= 0 ? 'var(--positive)' : 'var(--negative)' },
                    { label: 'Sharpe', value: sharpe.toFixed(2), color: sharpe > 0 ? 'var(--positive)' : 'var(--negative)' },
                    { label: 'VWAP', value: `${(vwapDist * 100).toFixed(2)}%`, color: 'var(--accent)' },
                ].map((m, i) => (
                    <div key={i} style={{
                        flex: 1, textAlign: 'center', padding: '4px 0',
                        borderRadius: 'var(--radius-sm)', background: 'var(--bg-3)',
                    }}>
                        <div className="mono" style={{ fontSize: 11, fontWeight: 600, color: m.color }}>{m.value}</div>
                        <div style={{ fontSize: 8.5, color: 'var(--text-2)', marginTop: 1 }}>{m.label}</div>
                    </div>
                ))}
            </div>

            {/* ── Market Regime ── */}
            <Section title="Market Regime (HMM)" open={sections.regime} toggle={() => toggle('regime')}
                badge={`${regimeConf.toFixed(0)}%`} badgeColor={regimeColor}>
                <RegimeBar probs={regimeProbs} />
            </Section>

            {/* ── Volatility Suite ── */}
            <Section title="Volatility Suite" open={sections.volatility} toggle={() => toggle('volatility')}
                badge={`${(garchCurrent * 100).toFixed(1)}%`} badgeColor="var(--warning)">
                <MiniGauge label="GARCH Forecast" value={garchForecast} max={0.1}
                    fmt={v => `${(v * 100).toFixed(2)}%`} color="var(--warning)" compact />
                <MiniGauge label="Leverage (γ)" value={garchAsymmetry} max={0.5}
                    fmt={v => v.toFixed(4)} color="var(--warning)" compact />
                <MiniGauge label="Heston Leverage" value={Math.abs(hestonLeverage)} max={1}
                    fmt={() => `ρ = ${hestonLeverage.toFixed(2)}`} color="var(--negative)" compact />
                <MiniGauge label="Roughness (H)" value={roughH} max={1}
                    fmt={v => `${v.toFixed(3)} (${roughInterpretation})`}
                    color={roughH < 0.3 ? '#06B6D4' : roughH > 0.5 ? 'var(--accent)' : 'var(--text-1)'} compact />
                <div className="flex justify-between" style={{ fontSize: 9.5, color: 'var(--text-2)', marginTop: 2 }}>
                    <span>Roughness Score</span>
                    <span className="mono" style={{ color: 'var(--text-1)' }}>{roughScore.toFixed(0)}/100</span>
                </div>
            </Section>

            {/* ── Order Flow ── */}
            <Section title="Order Flow (OFI)" open={sections.orderflow} toggle={() => toggle('orderflow')}
                badge={ofiSignal.replace('_', ' ')} badgeColor={ofiColor}>
                <div className="flex justify-between items-center" style={{ marginBottom: 6 }}>
                    <span style={{ fontSize: 10.5, color: 'var(--text-1)' }}>Pressure</span>
                    <span className="mono" style={{ fontSize: 11, color: ofiColor, fontWeight: 600 }}>
                        {ofiSignal.replace('_', ' ')} · {ofiStrength}
                    </span>
                </div>
                <MiniGauge label="OFI Normalized" value={(ofiNormalized + 1) / 2} max={1}
                    fmt={() => ofiNormalized.toFixed(3)}
                    color={ofiNormalized > 0.2 ? 'var(--positive)' : ofiNormalized < -0.2 ? 'var(--negative)' : 'var(--text-2)'}
                    compact />
            </Section>

            {/* ── Jump Detection ── */}
            <Section title="Jump Detection" open={sections.jumps} toggle={() => toggle('jumps')}
                badge={jumpDetected ? `⚡ ${jumpDirection}` : jumpRisk}
                badgeColor={jumpDetected ? 'var(--negative)' : 'var(--text-2)'}>
                {jumpDetected && (
                    <div style={{
                        padding: '6px 10px', borderRadius: 'var(--radius-sm)', marginBottom: 6,
                        background: 'var(--negative-dim)', border: '1px solid rgba(244,63,94,0.25)',
                        animation: 'pulse-glow 2s ease-in-out infinite',
                    }}>
                        <span style={{ fontSize: 11, color: 'var(--negative)', fontWeight: 600 }}>
                            ⚡ Black Swan Detected — {jumpDirection} ({jumpProb.toFixed(0)}% probability)
                        </span>
                    </div>
                )}
                <div className="flex justify-between" style={{ fontSize: 10, color: 'var(--text-1)', marginBottom: 3 }}>
                    <span>Merton Jump Risk</span>
                    <span className="mono" style={{ color: jumpRisk === 'HIGH' ? 'var(--negative)' : 'var(--text-1)' }}>{jumpRisk}</span>
                </div>
                <div className="flex justify-between" style={{ fontSize: 10, color: 'var(--text-1)' }}>
                    <span>Bates SVJ Status</span>
                    <span className="mono">{batesStatus}</span>
                </div>
            </Section>

            {/* ── Cycle Analysis ── */}
            <Section title="Cycle Analysis (EMD + HHT)" open={sections.cycles} toggle={() => toggle('cycles')}
                badge={hhtPeriod > 0 ? `${hhtPeriod.toFixed(0)}m` : undefined} badgeColor="#06B6D4">
                {cycleStrengths.map((s: number, i: number) => (
                    <MiniGauge key={i} label={`Cycle ${i + 1}`} value={s} max={1}
                        fmt={v => v.toFixed(3)} color="#06B6D4" compact />
                ))}
                {hhtPeriod > 0 && (
                    <div className="flex justify-between" style={{ fontSize: 10, color: 'var(--text-1)', marginTop: 2 }}>
                        <span>HHT Dominant Period</span>
                        <span className="mono" style={{ color: '#06B6D4' }}>{hhtPeriod.toFixed(1)} min</span>
                    </div>
                )}
            </Section>

            {/* ── Wavelet Analysis ── */}
            <Section title="Wavelet Analysis" open={sections.wavelets} toggle={() => toggle('wavelets')}
                badge={`${(trendStrength * 100).toFixed(0)}%`} badgeColor="#A78BFA">
                <MiniGauge label="Trend Strength" value={trendStrength} max={1}
                    fmt={v => `${(v * 100).toFixed(1)}%`} color="#A78BFA" compact />
                <div className="flex justify-between" style={{ fontSize: 10, color: 'var(--text-2)', marginTop: 2 }}>
                    <span>Signal vs Noise</span>
                    <span className="mono" style={{ color: 'var(--text-1)' }}>
                        {trendStrength > 0.6 ? 'Strong Trend' : trendStrength < 0.3 ? 'Noisy' : 'Moderate'}
                    </span>
                </div>
            </Section>

            {/* ── Fractal Intelligence ── */}
            <Section title="Fractal Intelligence" open={sections.fractals} toggle={() => toggle('fractals')}
                badge={mfInterp} badgeColor="#EC4899">
                <MiniGauge label="Multifractal ΔH" value={mfDeltaH} max={1}
                    fmt={v => v.toFixed(4)} color="#EC4899" compact />
                <div className="flex justify-between" style={{ fontSize: 10, color: 'var(--text-1)', marginBottom: 3 }}>
                    <span>Market Structure</span>
                    <span className="mono" style={{ color: '#EC4899' }}>{mfInterp}</span>
                </div>
                <div className="flex justify-between" style={{ fontSize: 10, color: 'var(--text-1)', marginBottom: 3 }}>
                    <span>TDA Complexity</span>
                    <span className="mono">{tdaComplexity}</span>
                </div>
                <MiniGauge label="TDA Persistence" value={tdaPersistence} max={1}
                    fmt={v => v.toFixed(4)} color="#8B5CF6" compact />
            </Section>

            {/* ── RQA Pattern Detection ── */}
            <Section title="Pattern Detection (RQA)" open={sections.rqa} toggle={() => toggle('rqa')}
                badge={rqaInterp} badgeColor="#8B5CF6">
                <MiniGauge label="Determinism" value={rqaDet} max={1}
                    fmt={v => v.toFixed(3)} color="#8B5CF6" compact />
                <MiniGauge label="Recurrence Rate" value={rqaRR} max={1}
                    fmt={v => v.toFixed(3)} color="#8B5CF6" compact />
            </Section>

            {/* ── Active Signals ── */}
            {signals.length > 0 && (
                <Section title="Active Signals" open={sections.signals} toggle={() => toggle('signals')}
                    badge={`${signals.length}`} badgeColor="var(--warning)">
                    {signals.map((s, i) => <SignalBadge key={i} signal={s} />)}
                </Section>
            )}
        </div>
    );
}
