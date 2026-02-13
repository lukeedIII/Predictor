type Props = {
    prediction: Record<string, unknown>;
    currentPrice: number;
};

export default function PredictionCard({ prediction, currentPrice }: Props) {
    const direction = (prediction?.direction as string) || '—';
    const confidence = (prediction?.confidence as number) || 0;
    const target1h = prediction?.target_price_1h as number | undefined;

    const color = direction === 'UP' ? 'var(--positive)' : direction === 'DOWN' ? 'var(--negative)' : 'var(--text-3)';
    const bgColor = direction === 'UP' ? 'var(--positive-bg)' : direction === 'DOWN' ? 'var(--negative-bg)' : 'var(--accent-bg)';
    const arrow = direction === 'UP' ? '↑' : direction === 'DOWN' ? '↓' : '–';

    // Confidence arc percentage
    const arcPct = Math.min(confidence, 100);

    // Expected move
    const move = target1h && currentPrice ? ((target1h - currentPrice) / currentPrice * 100) : null;

    return (
        <div className="card relative overflow-hide">
            {/* Subtle glow behind card when confident */}
            {confidence > 60 && (
                <div
                    className="confidence-glow"
                    style={{ background: `radial-gradient(circle, ${color}15 0%, transparent 70%)` }}
                />
            )}

            <div className="stat-label">AI PREDICTION</div>

            <div className="flex items-center gap-12 mt-8">
                {/* Direction badge */}
                <div
                    className="direction-badge"
                    style={{ background: bgColor, border: `1px solid ${color}30`, color }}
                >
                    {arrow}
                </div>

                <div>
                    <div className="stat-value sm" style={{ color }}>{direction || '—'}</div>
                    <div className="stat-sub">
                        {move !== null
                            ? <span style={{ color }}>{move > 0 ? '+' : ''}{move.toFixed(2)}% expected</span>
                            : 'Waiting for signal'
                        }
                    </div>
                </div>
            </div>

            {/* Confidence bar */}
            <div className="confidence-section">
                <div className="confidence-header">
                    <span className="stat-label mb-0">CONFIDENCE</span>
                    <span className="confidence-value" style={{ color }}>
                        {confidence.toFixed(1)}%
                    </span>
                </div>
                <div className="progress-track">
                    <div
                        className="progress-fill"
                        style={{
                            width: `${arcPct}%`,
                            background: `linear-gradient(90deg, ${color}90, ${color})`,
                            boxShadow: `0 0 10px ${color}40`,
                        }}
                    />
                </div>
            </div>
        </div>
    );
}
