import { useLivePrediction, useLiveAccuracy, useLiveAccuracySource, useLiveAccuracySamples } from '../stores/liveStore';
import { IconTrend, IconTrendDown } from './Icons';

export function SignalBadge() {
    const prediction = useLivePrediction();
    if (!prediction) {
        return (
            <div className="card card-compact animate-in">
                <div className="stat-label">Prediction</div>
                <div className="stat-value xs text-muted">Waiting…</div>
            </div>
        );
    }

    const direction = (prediction as any).direction?.toUpperCase?.() ?? 'NEUTRAL';
    const rawConf = (prediction as any).confidence ?? 0;
    // Backend sends 0-100 range; normalize to 0-1 for display logic
    const confidence = rawConf > 1 ? rawConf / 100 : rawConf;
    const isLong = direction === 'LONG' || direction === 'UP';
    const isShort = direction === 'SHORT' || direction === 'DOWN';

    const badgeCls = isLong ? 'badge-long' : isShort ? 'badge-short' : 'badge-neutral';
    const Icon = isLong ? IconTrend : IconTrendDown;
    const pct = (confidence * 100).toFixed(2);

    return (
        <div className="card card-compact animate-in">
            <div className="stat-label">Prediction</div>
            <div className="flex items-center gap-8" style={{ marginTop: 4 }}>
                <span className={`badge ${badgeCls}`}>
                    <Icon style={{ width: 12, height: 12 }} />
                    {direction}
                </span>
                <span className="mono stat-value xs">{pct}%</span>
            </div>
            {/* Confidence bar */}
            <div className="progress-track" style={{ marginTop: 8 }}>
                <div
                    className="progress-fill"
                    style={{
                        width: `${confidence * 100}%`,
                        background: isLong ? 'var(--positive)' : isShort ? 'var(--negative)' : 'var(--accent)',
                    }}
                />
            </div>
        </div>
    );
}

export function AccuracyCard() {
    const rawAcc = useLiveAccuracy();
    const source = useLiveAccuracySource();
    const samples = useLiveAccuracySamples();
    // Backend sends 0-100 range; normalize to 0-1 for display
    const accuracy = rawAcc != null ? (rawAcc > 1 ? rawAcc / 100 : rawAcc) : null;
    const pct = accuracy != null ? (accuracy * 100).toFixed(1) : '—';
    const isGood = (accuracy ?? 0) >= 0.5;
    const isLive = source === 'live';

    return (
        <div className="card card-compact animate-in">
            <div className="flex items-center justify-between">
                <span className="stat-label">Accuracy</span>
                <span className="mono" style={{
                    fontSize: 9,
                    padding: '1px 5px',
                    borderRadius: 3,
                    background: isLive ? 'rgba(0,230,118,0.12)' : 'rgba(255,171,64,0.12)',
                    color: isLive ? '#69F0AE' : '#FFAB40',
                    fontWeight: 600,
                    letterSpacing: 0.5,
                }}>
                    {isLive ? `LIVE · ${samples}` : 'TRAIN'}
                </span>
            </div>
            <div className={`stat-value ${isGood ? 'text-positive' : 'text-warning'}`}>
                <span className="mono">{pct}{accuracy != null ? '%' : ''}</span>
            </div>
        </div>
    );
}
