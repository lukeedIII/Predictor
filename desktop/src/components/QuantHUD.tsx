import { useApi } from '../hooks/useApi';
import { IconCrosshair, IconChart, IconBolt } from './Icons';

type CycleData = { cycles: Array<{ label: string; strength: number }> };

type Props = {
    quant: Record<string, unknown>;
    prediction: Record<string, unknown>;
};

function RegimeCard({ quant }: { quant: Record<string, unknown> }) {
    const regime = quant?.regime as Record<string, unknown> | undefined;
    const name = (regime?.regime as string) || 'UNKNOWN';
    const conf = (regime?.confidence as number) || 0;

    const styles: Record<string, { color: string; bg: string }> = {
        BULL: { color: 'var(--positive)', bg: 'var(--positive-bg)' },
        BEAR: { color: 'var(--negative)', bg: 'var(--negative-bg)' },
        SIDEWAYS: { color: 'var(--accent)', bg: 'var(--accent-bg)' },
        UNKNOWN: { color: 'var(--text-3)', bg: 'var(--surface-0)' },
    };
    const s = styles[name] || styles.UNKNOWN;

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title"><IconCrosshair size={14} style={{ marginRight: 5, verticalAlign: -2 }} /> Regime</span>
            </div>
            <div className="regime-display" style={{ background: s.bg, borderColor: `${s.color}30` }}>
                <div className="stat-value sm" style={{ color: s.color }}>{name}</div>
                <div className="stat-sub">{conf.toFixed(0)}% Confidence</div>
            </div>
        </div>
    );
}

function CyclesCard() {
    const { data } = useApi<CycleData>('/api/cycles', 15000);
    const cycles = data?.cycles || [];

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title"><IconChart size={14} style={{ marginRight: 5, verticalAlign: -2 }} /> FFT Cycles</span>
            </div>
            {cycles.length > 0 ? (
                <div className="flex-col gap-10">
                    {cycles.map(c => (
                        <div key={c.label}>
                            <div className="flex-between mb-3">
                                <span className="stat-label mb-0">{c.label}</span>
                                <span className="confidence-value text-info">
                                    {(c.strength * 100).toFixed(1)}%
                                </span>
                            </div>
                            <div className="progress-track">
                                <div
                                    className="progress-fill"
                                    style={{
                                        width: `${Math.min(c.strength * 200, 100)}%`,
                                        background: 'linear-gradient(90deg, var(--info), var(--accent))',
                                        boxShadow: '0 0 6px rgba(96,165,250,0.3)',
                                    }}
                                />
                            </div>
                        </div>
                    ))}
                </div>
            ) : (
                <div className="skeleton skeleton-h80" />
            )}
        </div>
    );
}

function HurstCard({ prediction }: { prediction: Record<string, unknown> }) {
    const hurst = (prediction?.hurst as number) || 0.5;

    let color = 'var(--info)';
    let label = 'Chaotic';
    if (hurst > 0.55) { color = 'var(--positive)'; label = 'Trending'; }
    else if (hurst < 0.45) { color = 'var(--negative)'; label = 'Mean Revert'; }

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">🌀 Hurst</span>
            </div>
            <div className="gauge-container">
                <div className="stat-value sm mono" style={{ color }}>{hurst.toFixed(3)}</div>
                <div className="stat-sub">{label}</div>
                <div className="gauge-bar">
                    <div
                        className="gauge-fill"
                        style={{
                            width: `${hurst * 100}%`,
                            background: `linear-gradient(90deg, var(--negative), var(--info), var(--positive))`,
                        }}
                    />
                </div>
            </div>
        </div>
    );
}

function OrderFlowCard({ quant }: { quant: Record<string, unknown> }) {
    const ofi = quant?.order_flow as Record<string, unknown> | undefined;
    const norm = (ofi?.normalized as number) || 0;

    const color = norm > 0 ? 'var(--positive)' : norm < 0 ? 'var(--negative)' : 'var(--text-3)';
    const label = norm > 0 ? 'Buy Pressure' : norm < 0 ? 'Sell Pressure' : 'Neutral';

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title"><IconChart size={14} style={{ marginRight: 5, verticalAlign: -2 }} /> Order Flow</span>
            </div>
            <div className="text-center">
                <div className="stat-value xs" style={{ color }}>{label}</div>
                <div className="flow-track">
                    <div className="flow-center" />
                    <div
                        className="flow-bar"
                        style={{
                            [norm >= 0 ? 'left' : 'right']: '50%',
                            width: `${Math.abs(norm) * 50}%`,
                            background: color,
                        }}
                    />
                </div>
                <div className="stat-sub mt-4">{norm > 0 ? '+' : ''}{norm.toFixed(2)}</div>
            </div>
        </div>
    );
}

function JumpRiskCard({ quant }: { quant: Record<string, unknown> }) {
    const jump = quant?.jump_risk as Record<string, unknown> | undefined;
    const level = (jump?.risk_level as string) || 'UNKNOWN';
    const intensity = (jump?.jump_intensity as number) || 0;

    const styles: Record<string, { color: string; bg: string }> = {
        HIGH: { color: 'var(--negative)', bg: 'var(--negative-bg)' },
        MEDIUM: { color: 'var(--warning)', bg: 'var(--warning-bg)' },
        LOW: { color: 'var(--positive)', bg: 'var(--positive-bg)' },
        UNKNOWN: { color: 'var(--text-3)', bg: 'var(--surface-0)' },
    };
    const s = styles[level] || styles.UNKNOWN;

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title"><IconBolt size={14} style={{ marginRight: 5, verticalAlign: -2 }} /> Jump Risk</span>
            </div>
            <div className="regime-display" style={{ background: s.bg, borderColor: `${s.color}30` }}>
                <div className="stat-value xs" style={{ color: s.color }}>{level}</div>
                <div className="stat-sub">λ = {intensity.toFixed(3)}</div>
            </div>
        </div>
    );
}

export default function QuantHUD({ quant, prediction }: Props) {
    return (
        <>
            <RegimeCard quant={quant} />
            <CyclesCard />
            <HurstCard prediction={prediction} />
            <OrderFlowCard quant={quant} />
            <JumpRiskCard quant={quant} />
        </>
    );
}
