import { useRef, useState, useEffect, useCallback } from 'react';
import ReactGridLayout, { type LayoutItem } from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

import { PriceCard, VolumeCard } from '../components/MetricCard';
import { SignalBadge, AccuracyCard } from '../components/SignalBadge';
import QuantPanel from '../components/QuantPanel';
import NewsFeed from '../components/NewsFeed';
import SystemHealth from '../components/SystemHealth';
import TrainingLog from '../components/TrainingLog';
import TradingViewChart from '../components/TradingViewChart';
import SwissWeather from '../components/SwissWeather';
import { IconRefresh } from '../components/Icons';

// ── Layout Definitions ──────────────────────────────────────
const STORAGE_KEY = 'nexus-dashboard-layout';
const COLS = 12;
const ROW_HEIGHT = 110;

const DEFAULT_LAYOUT: LayoutItem[] = [
    // Top metrics strip
    { i: 'price', x: 0, y: 0, w: 2, h: 2, minW: 2, minH: 2 },
    { i: 'signal', x: 2, y: 0, w: 2, h: 2, minW: 2, minH: 2 },
    { i: 'accuracy', x: 4, y: 0, w: 3, h: 2, minW: 2, minH: 2 },
    { i: 'volume', x: 7, y: 0, w: 2, h: 2, minW: 2, minH: 2 },
    { i: 'weather', x: 9, y: 0, w: 3, h: 2, minW: 2, minH: 2 },
    // Main content
    { i: 'chart', x: 0, y: 2, w: 8, h: 5, minW: 4, minH: 3 },
    { i: 'quant', x: 8, y: 2, w: 4, h: 5, minW: 3, minH: 3 },
    // Bottom panels
    { i: 'news', x: 0, y: 7, w: 4, h: 4, minW: 3, minH: 2 },
    { i: 'health', x: 4, y: 7, w: 4, h: 4, minW: 3, minH: 2 },
    { i: 'training', x: 8, y: 7, w: 4, h: 4, minW: 3, minH: 2 },
];

function loadLayout(): LayoutItem[] {
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
            const parsed = JSON.parse(saved);
            if (Array.isArray(parsed) && parsed.length === DEFAULT_LAYOUT.length) {
                // Re-apply min constraints from defaults
                return parsed.map((item: any) => {
                    const def = DEFAULT_LAYOUT.find(d => d.i === item.i);
                    return { ...item, minW: def?.minW, minH: def?.minH };
                });
            }
        }
    } catch { /* ignore */ }
    return DEFAULT_LAYOUT;
}

function saveLayout(layout: readonly LayoutItem[]) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(layout));
}

// ── Dashboard Component ─────────────────────────────────────
export default function Dashboard() {
    const containerRef = useRef<HTMLDivElement>(null);
    const [width, setWidth] = useState(1200);
    const [layout, setLayout] = useState<LayoutItem[]>(loadLayout);

    // Measure container width
    useEffect(() => {
        if (!containerRef.current) return;
        const ro = new ResizeObserver(entries => {
            for (const entry of entries) {
                setWidth(entry.contentRect.width);
            }
        });
        ro.observe(containerRef.current);
        return () => ro.disconnect();
    }, []);

    const handleLayoutChange = useCallback((newLayout: readonly LayoutItem[]) => {
        setLayout([...newLayout]);
        saveLayout(newLayout);
    }, []);

    const resetLayout = useCallback(() => {
        setLayout([...DEFAULT_LAYOUT]);
        saveLayout(DEFAULT_LAYOUT);
    }, []);

    return (
        <div ref={containerRef} className="dashboard-grid-container">
            {/* Reset layout button */}
            <div style={{
                display: 'flex', justifyContent: 'flex-end', padding: '0 0 4px',
                position: 'relative', zIndex: 10
            }}>
                <button
                    className="btn btn-sm"
                    onClick={resetLayout}
                    title="Reset dashboard layout to default"
                    style={{ fontSize: 11, opacity: 0.6, gap: 4 }}
                >
                    <IconRefresh style={{ width: 10, height: 10 }} /> Reset Layout
                </button>
            </div>

            <ReactGridLayout
                layout={layout}
                cols={COLS}
                rowHeight={ROW_HEIGHT}
                width={width}
                onLayoutChange={handleLayoutChange}
                draggableHandle=".grid-drag-handle"
                compactType="vertical"
                margin={[10, 10] as [number, number]}
                containerPadding={[0, 0] as [number, number]}
                useCSSTransforms={true}
                isResizable={true}
                isDraggable={true}
            >
                <div key="price">
                    <GridCard title="Price">
                        <PriceCard />
                    </GridCard>
                </div>
                <div key="signal">
                    <GridCard title="Signal">
                        <SignalBadge />
                    </GridCard>
                </div>
                <div key="accuracy">
                    <GridCard title="Accuracy">
                        <AccuracyCard />
                    </GridCard>
                </div>
                <div key="volume">
                    <GridCard title="Volume">
                        <VolumeCard />
                    </GridCard>
                </div>
                <div key="weather">
                    <GridCard title="Weather">
                        <SwissWeather />
                    </GridCard>
                </div>
                <div key="chart">
                    <GridCard title="Chart" flush>
                        <TradingViewChart />
                    </GridCard>
                </div>
                <div key="quant">
                    <GridCard title="Quant Intelligence">
                        <QuantPanel />
                    </GridCard>
                </div>
                <div key="news">
                    <GridCard title="News Feed">
                        <NewsFeed />
                    </GridCard>
                </div>
                <div key="health">
                    <GridCard title="System Health">
                        <SystemHealth />
                    </GridCard>
                </div>
                <div key="training">
                    <GridCard title="Training Log">
                        <TrainingLog />
                    </GridCard>
                </div>
            </ReactGridLayout>
        </div>
    );
}

// ── Grid Card Wrapper ───────────────────────────────────────
function GridCard({ children, title, flush }: {
    children: React.ReactNode;
    title: string;
    flush?: boolean;
}) {
    return (
        <div className={`grid-card-wrapper ${flush ? 'grid-card-flush' : ''}`}>
            <div className="grid-drag-handle" title={`Drag to move "${title}"`}>
                <span className="grid-drag-dots">⠿</span>
                <span className="grid-drag-label">{title}</span>
            </div>
            <div className="grid-card-content">
                {children}
            </div>
        </div>
    );
}
