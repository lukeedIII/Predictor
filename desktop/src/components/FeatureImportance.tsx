import { useEffect, useState } from 'react';

interface FeatureData {
    name: string;
    importance: number;
    category: string;
}

const CATEGORY_COLORS: Record<string, string> = {
    ohlcv: '#4fc3f7',
    technical: '#81c784',
    microstructure: '#ff8a65',
    quant: '#ba68c8',
    trend: '#ffd54f',
    volume: '#4dd0e1',
    cross_asset: '#e57373',
    other: '#90a4ae',
};

const CATEGORY_LABELS: Record<string, string> = {
    ohlcv: 'OHLCV',
    technical: 'Technical',
    microstructure: 'Microstructure',
    quant: 'Quant',
    trend: 'Trend',
    volume: 'Volume',
    cross_asset: 'Cross-Asset',
    other: 'Other',
};

export default function FeatureImportance() {
    const [features, setFeatures] = useState<FeatureData[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await fetch('http://localhost:8420/api/feature-importance');
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();
                setFeatures(data.features.slice(0, 20)); // Top 20
                setError('');
            } catch (e: any) {
                setError(e.message || 'Failed to load');
            } finally {
                setLoading(false);
            }
        };
        fetchData();
        const interval = setInterval(fetchData, 60000); // refresh every minute
        return () => clearInterval(interval);
    }, []);

    if (loading) return <div className="feature-importance loading">Loading features...</div>;
    if (error) return <div className="feature-importance error">{error}</div>;

    const maxImportance = features.length > 0 ? features[0].importance : 1;

    return (
        <div className="feature-importance">
            <div className="fi-header">
                <h3>Feature Importance</h3>
                <span className="fi-count">{features.length} features</span>
            </div>

            <div className="fi-legend">
                {Object.entries(CATEGORY_COLORS).map(([cat, color]) => (
                    <span key={cat} className="fi-legend-item">
                        <span className="fi-dot" style={{ background: color }} />
                        {CATEGORY_LABELS[cat] || cat}
                    </span>
                ))}
            </div>

            <div className="fi-bars">
                {features.map((f, i) => {
                    const widthPct = (f.importance / maxImportance) * 100;
                    const color = CATEGORY_COLORS[f.category] || CATEGORY_COLORS.other;
                    return (
                        <div key={f.name} className="fi-row" style={{ animationDelay: `${i * 30}ms` }}>
                            <span className="fi-label">{f.name.replace(/_/g, ' ')}</span>
                            <div className="fi-bar-track">
                                <div
                                    className="fi-bar-fill"
                                    style={{ width: `${widthPct}%`, background: color }}
                                />
                            </div>
                            <span className="fi-value">{(f.importance * 100).toFixed(1)}%</span>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
