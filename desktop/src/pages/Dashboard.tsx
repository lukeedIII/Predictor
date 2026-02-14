import { PriceCard, VolumeCard } from '../components/MetricCard';
import { SignalBadge, AccuracyCard } from '../components/SignalBadge';
import QuantPanel from '../components/QuantPanel';
import NewsFeed from '../components/NewsFeed';
import SystemHealth from '../components/SystemHealth';
import TrainingLog from '../components/TrainingLog';
import TradingViewChart from '../components/TradingViewChart';
import SwissWeather from '../components/SwissWeather';

export default function Dashboard() {
    return (
        <div className="flex-col gap-10">
            {/* ─── Metrics Strip ────────────────── */}
            <div className="grid grid-5 stagger">
                <PriceCard />
                <SignalBadge />
                <AccuracyCard />
                <VolumeCard />
                <SwissWeather />
            </div>

            {/* ─── Chart + Intel Panel ──────────── */}
            <div className="dash-grid">
                {/* Chart */}
                <div className="card card-flush animate-in" style={{ animationDelay: '100ms' }}>
                    <TradingViewChart />
                </div>

                {/* Intel Panel */}
                <div className="flex-col gap-10 stagger">
                    <QuantPanel />
                    <NewsFeed />
                    <SystemHealth />
                    <TrainingLog />
                </div>
            </div>
        </div>
    );
}
