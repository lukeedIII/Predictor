import { useApi } from '../hooks/useApi';

type NewsItem = { headline: string; source: string; sentiment?: string; time?: string; url?: string };

export default function NewsFeed() {
    const { data } = useApi<{ items: NewsItem[] }>('/api/news', 30_000);
    const items = data?.items ?? [];

    const sentimentDot = (s?: string) => {
        if (!s) return null;
        const lower = s.toLowerCase();
        const color = lower === 'positive' || lower === 'bullish' ? 'var(--positive)'
            : lower === 'negative' || lower === 'bearish' ? 'var(--negative)'
                : 'var(--text-2)';
        return <span className="status-dot" style={{ background: color, width: 5, height: 5 }} />;
    };

    return (
        <div className="card animate-in">
            <div className="card-header">
                <span className="card-title">Market News</span>
                <span style={{ fontSize: 10, color: 'var(--text-2)' }}>{items.length} items</span>
            </div>
            <div style={{ maxHeight: 380, overflowY: 'auto' }}>
                {items.length === 0 ? (
                    <div className="empty-state" style={{ padding: 20 }}>
                        <span style={{ fontSize: 12 }}>Fetching news feeds…</span>
                    </div>
                ) : (
                    items.map((item, i) => (
                        <div
                            key={i}
                            style={{
                                padding: '8px 12px',
                                borderBottom: i < items.length - 1 ? '1px solid var(--border)' : 'none',
                            }}
                        >
                            <div className="flex items-center gap-4" style={{ marginBottom: 2 }}>
                                {sentimentDot(item.sentiment)}
                                <span style={{ fontSize: 12, color: 'var(--text-0)', flex: 1, lineHeight: 1.4 }}>
                                    {item.url ? (
                                        <a href={item.url} target="_blank" rel="noopener" style={{ color: 'inherit', textDecoration: 'none' }}>
                                            {item.headline}
                                        </a>
                                    ) : item.headline}
                                </span>
                            </div>
                            <div className="flex items-center gap-8" style={{ fontSize: 10, color: 'var(--text-2)', paddingLeft: 9 }}>
                                <span style={{ fontWeight: 500 }}>{item.source}</span>
                                {item.time && <span>· {item.time}</span>}
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}
