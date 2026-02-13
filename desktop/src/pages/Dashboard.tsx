import { useWebSocket } from '../hooks/useWebSocket';
import TradingViewChart from '../components/TradingViewChart';
import PredictionCard from '../components/PredictionCard';
import QuantHUD from '../components/QuantHUD';
import NewsFeed from '../components/NewsFeed';
import FeatureImportance from '../components/FeatureImportance';
import SystemHealth from '../components/SystemHealth';

export default function Dashboard() {
    // High-frequency data from WebSocket (1s push)
    const ws = useWebSocket();

    const prediction = ws.prediction || {};
    const accuracy = ws.accuracy;
    const currentPrice = ws.price || 0;
    const changePctColor = (ws.change_pct ?? 0) >= 0 ? 'var(--positive)' : 'var(--negative)';

    return (
        <div className="animate-in">
            {/* Top metrics row */}
            <div className="grid-3 animate-in animate-in-1" style={{ marginBottom: 12 }}>
                <PredictionCard prediction={prediction} currentPrice={currentPrice} />

                <div className="card">
                    <div className="stat-label">CURRENT PRICE</div>
                    <div className="stat-value mono" style={{ color: 'var(--text-1)' }}>
                        {currentPrice > 0
                            ? `$${currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
                            : '—'}
                    </div>
                    <div className="stat-sub" style={{ color: changePctColor }}>
                        {ws.change_pct !== null
                            ? `${ws.change_pct >= 0 ? '+' : ''}${ws.change_pct.toFixed(2)}% (24h)`
                            : 'BTC/USDT'}
                    </div>
                </div>

                <div className="card">
                    <div className="stat-label">ACCURACY</div>
                    <div className="stat-value" style={{ color: 'var(--accent)' }}>
                        {accuracy != null && accuracy > 0
                            ? `${accuracy > 1 ? accuracy.toFixed(1) : (accuracy * 100).toFixed(1)}%`
                            : '—'}
                    </div>
                    <div className="stat-sub">Validated predictions</div>
                </div>
            </div>

            {/* Main layout: chart + HUD */}
            <div className="dash-layout animate-in animate-in-2">
                {/* Left: TradingView Chart */}
                <TradingViewChart />

                {/* Right: Quant HUD — scrollable sidebar */}
                <div style={{
                    display: 'flex', flexDirection: 'column', gap: 10,
                    overflowY: 'auto', maxHeight: 'calc(100vh - 280px)',
                    paddingRight: 4,
                }}>
                    <QuantHUD quant={ws.quant || {}} prediction={prediction} />
                </div>
            </div>

            {/* Phase 3+4: Feature Importance + System Health */}
            <div className="grid-2 animate-in animate-in-3" style={{ marginTop: 12 }}>
                <FeatureImportance />
                <SystemHealth />
            </div>

            {/* News feed */}
            <div className="animate-in animate-in-4" style={{ marginTop: 12 }}>
                <NewsFeed />
            </div>
        </div>
    );
}
