import { useLivePrice, useLiveChangePct, useLiveHigh24h, useLiveLow24h, useLiveVolumeBtc } from '../stores/liveStore';

type Props = { label: string; children: React.ReactNode; sub?: string };

export function MetricCard({ label, children, sub }: Props) {
    return (
        <div className="card card-compact animate-in">
            <div className="stat-label">{label}</div>
            <div className="stat-value">{children}</div>
            {sub && <div className="stat-sub">{sub}</div>}
        </div>
    );
}

/* ─── Pre-built Metric Cards ─────────────────────── */

export function PriceCard() {
    const price = useLivePrice();
    const changePct = useLiveChangePct();
    const high = useLiveHigh24h();
    const low = useLiveLow24h();

    const pctClass = (changePct ?? 0) >= 0 ? 'text-positive' : 'text-negative';
    const arrow = (changePct ?? 0) >= 0 ? '▲' : '▼';

    return (
        <MetricCard
            label="BTC / USDT"
            sub={high != null && low != null ? `H ${high.toLocaleString()} / L ${low.toLocaleString()}` : undefined}
        >
            <span className="mono">
                {price != null ? `$${price.toLocaleString('en-US', { minimumFractionDigits: 2 })}` : '—'}
            </span>
            {changePct != null && (
                <span className={pctClass} style={{ fontSize: 13, fontWeight: 600, marginLeft: 8 }}>
                    {arrow} {Math.abs(changePct).toFixed(2)}%
                </span>
            )}
        </MetricCard>
    );
}

export function VolumeCard() {
    const vol = useLiveVolumeBtc();
    const fmt = (v: number) => {
        if (v >= 1000) return `${(v / 1000).toFixed(1)}K`;
        return v.toFixed(1);
    };
    return (
        <MetricCard label="24h Volume">
            <span className="mono">{vol != null ? `${fmt(vol)} BTC` : '—'}</span>
        </MetricCard>
    );
}
