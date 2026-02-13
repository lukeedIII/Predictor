import { useApi } from '../hooks/useApi';
import { IconNews, IconActivity } from './Icons';

type NewsItem = {
    source?: string;
    title?: string;
    sentiment?: string;
    sentiment_score?: number;
};

type NewsData = {
    items: NewsItem[];
    error?: string;
};

export default function NewsFeed() {
    const { data } = useApi<NewsData>('/api/news', 60000);
    const items = data?.items || [];

    if (items.length === 0) {
        return (
            <div className="card">
                <div className="card-header">
                    <span className="card-title"><IconNews size={14} style={{ marginRight: 5, verticalAlign: -2 }} /> News Intelligence</span>
                </div>
                <div className="empty-state p-20">
                    <div className="text-24 mb-8"><IconActivity size={24} style={{ color: 'var(--text-3)' }} /></div>
                    <div className="text-3">Fetching news feed...</div>
                </div>
            </div>
        );
    }

    const isBull = (i: NewsItem) => i.sentiment === 'BULLISH' || (i.sentiment_score || 0) > 0.1;
    const isBear = (i: NewsItem) => i.sentiment === 'BEARISH' || (i.sentiment_score || 0) < -0.1;
    const bullish = items.filter(isBull).length;
    const bearish = items.filter(isBear).length;

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title"><IconNews size={14} style={{ marginRight: 5, verticalAlign: -2 }} /> News Intelligence</span>
                <div className="news-stats">
                    <span>{items.length} items</span>
                    <span className="text-positive">{bullish} Bullish</span>
                    <span className="text-negative">{bearish} Bearish</span>
                </div>
            </div>
            <div className="news-scroll">
                {items.slice(0, 12).map((item, i) => {
                    let sentColor = 'var(--text-3)';
                    let sentLabel = 'Neutral';
                    let borderColor = 'var(--text-4)';

                    if (isBull(item)) {
                        sentColor = 'var(--positive)';
                        sentLabel = 'Bullish';
                        borderColor = 'var(--positive)';
                    } else if (isBear(item)) {
                        sentColor = 'var(--negative)';
                        sentLabel = 'Bearish';
                        borderColor = 'var(--negative)';
                    }

                    return (
                        <div key={i} className="news-item" style={{ borderLeftColor: borderColor }}>
                            <div className="news-source"><IconActivity size={12} style={{ marginRight: 4, verticalAlign: -2 }} /> {item.source || 'Unknown'}</div>
                            <div className="news-title">{(item.title || '').slice(0, 150)}</div>
                            <div className="news-sentiment" style={{ color: sentColor }}>{sentLabel}</div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
