import { useRef, useState, useEffect, useCallback } from 'react';
import ReactGridLayout, { type LayoutItem, verticalCompactor } from 'react-grid-layout';
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
import WorldClock from '../components/WorldClock';
import { IconRefresh } from '../components/Icons';

// ── Constants ────────────────────────────────────────────────
const STORAGE_KEY = 'nexus-dashboard-layout-v5';
const PRESETS_KEY = 'nexus-dashboard-presets-v2';
const ACTIVE_SLOT_KEY = 'nexus-dashboard-active-slot';
const COLS = 12;
const ROW_HEIGHT = 30;

// Min constraints applied to every layout
const MINS: Record<string, { minW: number; minH: number }> = {
    price: { minW: 2, minH: 4 }, signal: { minW: 2, minH: 4 },
    accuracy: { minW: 2, minH: 4 }, volume: { minW: 2, minH: 4 },
    weather: { minW: 2, minH: 4 }, clock: { minW: 2, minH: 4 },
    chart: { minW: 4, minH: 8 }, quant: { minW: 3, minH: 8 },
    news: { minW: 3, minH: 6 }, health: { minW: 3, minH: 6 },
    training: { minW: 3, minH: 6 },
};

function applyMins(layout: LayoutItem[]): LayoutItem[] {
    return layout.map(item => {
        const m = MINS[item.i] || {};
        return { ...item, minW: m.minW, minH: m.minH };
    });
}

// ── Default Layout ───────────────────────────────────────────
const DEFAULT_LAYOUT: LayoutItem[] = applyMins([
    { i: 'price', x: 0, y: 0, w: 2, h: 5 },
    { i: 'signal', x: 2, y: 0, w: 2, h: 5 },
    { i: 'accuracy', x: 4, y: 0, w: 2, h: 5 },
    { i: 'volume', x: 6, y: 0, w: 2, h: 5 },
    { i: 'weather', x: 8, y: 0, w: 2, h: 5 },
    { i: 'clock', x: 10, y: 0, w: 2, h: 5 },
    { i: 'chart', x: 0, y: 5, w: 8, h: 16 },
    { i: 'quant', x: 8, y: 5, w: 4, h: 16 },
    { i: 'news', x: 0, y: 21, w: 4, h: 12 },
    { i: 'health', x: 4, y: 21, w: 4, h: 12 },
    { i: 'training', x: 8, y: 21, w: 4, h: 12 },
]);

// ── Preset 1: Trading Focus (hardcoded from user's screenshot) ──
// Large chart dominates, compact metrics, quant panel, bottom panels smaller
const PRESET_1: LayoutItem[] = applyMins([
    { i: 'price', x: 0, y: 0, w: 2, h: 5 },
    { i: 'signal', x: 2, y: 0, w: 2, h: 5 },
    { i: 'accuracy', x: 4, y: 0, w: 2, h: 5 },
    { i: 'volume', x: 6, y: 0, w: 2, h: 5 },
    { i: 'weather', x: 8, y: 0, w: 2, h: 5 },
    { i: 'clock', x: 10, y: 0, w: 2, h: 5 },
    { i: 'chart', x: 0, y: 5, w: 8, h: 20 },
    { i: 'quant', x: 8, y: 5, w: 4, h: 20 },
    { i: 'news', x: 0, y: 25, w: 4, h: 10 },
    { i: 'health', x: 4, y: 25, w: 4, h: 10 },
    { i: 'training', x: 8, y: 25, w: 4, h: 10 },
]);

// ── Persistence ──────────────────────────────────────────────
type Presets = { [slot: number]: LayoutItem[] | null };

function loadLayout(): LayoutItem[] {
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
            const parsed = JSON.parse(saved);
            if (Array.isArray(parsed) && parsed.length === DEFAULT_LAYOUT.length) {
                return applyMins(parsed);
            }
        }
    } catch { /* ignore */ }
    return [...DEFAULT_LAYOUT];
}

function saveLayout(layout: readonly LayoutItem[]) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(layout));
}

function loadPresets(): Presets {
    try {
        const saved = localStorage.getItem(PRESETS_KEY);
        if (saved) {
            const parsed = JSON.parse(saved);
            return {
                1: parsed['1'] ? applyMins(parsed['1']) : PRESET_1,
                2: parsed['2'] ? applyMins(parsed['2']) : null,
                3: parsed['3'] ? applyMins(parsed['3']) : null,
            };
        }
    } catch { /* ignore */ }
    return { 1: PRESET_1, 2: null, 3: null };
}

function savePresets(presets: Presets) {
    localStorage.setItem(PRESETS_KEY, JSON.stringify(presets));
}

function loadActiveSlot(): number | null {
    try {
        const s = localStorage.getItem(ACTIVE_SLOT_KEY);
        return s ? parseInt(s) : null;
    } catch { return null; }
}

function saveActiveSlot(slot: number | null) {
    if (slot === null) localStorage.removeItem(ACTIVE_SLOT_KEY);
    else localStorage.setItem(ACTIVE_SLOT_KEY, String(slot));
}

// ── Dashboard Component ─────────────────────────────────────
const THEME_KEY = 'nexus-dashboard-theme';

export default function Dashboard() {
    const containerRef = useRef<HTMLDivElement>(null);
    const [width, setWidth] = useState(1200);
    const [layout, setLayout] = useState<LayoutItem[]>(loadLayout);
    const [presets, setPresets] = useState<Presets>(loadPresets);
    const [activeSlot, setActiveSlot] = useState<number | null>(loadActiveSlot);
    const [saving, setSaving] = useState(false);
    const [lightMode, setLightMode] = useState(() => {
        try { return localStorage.getItem(THEME_KEY) === 'light'; } catch { return false; }
    });

    // Persist theme
    useEffect(() => {
        localStorage.setItem(THEME_KEY, lightMode ? 'light' : 'dark');
    }, [lightMode]);

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
        const arr = [...newLayout];
        setLayout(arr);
        saveLayout(arr);
    }, []);

    const resetLayout = useCallback(() => {
        setLayout([...DEFAULT_LAYOUT]);
        saveLayout(DEFAULT_LAYOUT);
        setActiveSlot(null);
        saveActiveSlot(null);
    }, []);

    // Load a preset slot
    const loadPreset = useCallback((slot: number) => {
        const preset = presets[slot];
        if (!preset) return;
        setLayout([...preset]);
        saveLayout(preset);
        setActiveSlot(slot);
        saveActiveSlot(slot);
    }, [presets]);

    // Save current layout into a slot
    const saveToSlot = useCallback((slot: number) => {
        const updated = { ...presets, [slot]: [...layout] };
        setPresets(updated);
        savePresets(updated);
        setActiveSlot(slot);
        saveActiveSlot(slot);
        setSaving(false);
    }, [presets, layout]);

    const handleSlotClick = useCallback((slot: number) => {
        if (saving) {
            saveToSlot(slot);
        } else {
            loadPreset(slot);
        }
    }, [saving, saveToSlot, loadPreset]);

    return (
        <div ref={containerRef} className={`dashboard-grid-container ${lightMode ? 'dashboard-light' : ''}`}>
            {/* Toolbar: presets + theme + reset */}
            <div className="dashboard-toolbar">
                {/* Left: Preset slots */}
                <div className="preset-group">
                    {[1, 2, 3].map(slot => {
                        const hasPreset = presets[slot] !== null;
                        const isActive = activeSlot === slot && !saving;
                        return (
                            <button
                                key={slot}
                                className={`preset-btn ${isActive ? 'preset-active' : ''} ${saving ? 'preset-saving' : ''} ${!hasPreset && !saving ? 'preset-empty' : ''}`}
                                onClick={() => handleSlotClick(slot)}
                                title={saving
                                    ? `Save current layout to slot ${slot}`
                                    : hasPreset
                                        ? `Load layout ${slot}`
                                        : `Slot ${slot} — empty (use Save to store)`}
                            >
                                {slot}
                            </button>
                        );
                    })}
                    <button
                        className={`preset-save-btn ${saving ? 'preset-save-active' : ''}`}
                        onClick={() => setSaving(prev => !prev)}
                        title={saving ? 'Cancel save' : 'Save current layout to a slot'}
                    >
                        {saving ? '✕' : '💾'}
                    </button>
                    <button
                        className={`preset-save-btn ${lightMode ? 'preset-save-active' : ''}`}
                        onClick={() => setLightMode(prev => !prev)}
                        title={lightMode ? 'Switch to dark mode' : 'Switch to light mode'}
                        style={{ fontSize: 13 }}
                    >
                        {lightMode ? '🌙' : '☀️'}
                    </button>
                </div>

                <button
                    className="btn btn-sm"
                    onClick={resetLayout}
                    title="Reset dashboard layout to default"
                    style={{ fontSize: 11, opacity: 0.6, gap: 4 }}
                >
                    <IconRefresh style={{ width: 10, height: 10 }} /> Reset
                </button>
            </div>

            {/* Save mode hint */}
            {saving && (
                <div className="preset-hint">
                    Click a slot number to save the current layout there
                </div>
            )}

            <ReactGridLayout
                layout={layout}
                width={width}
                onLayoutChange={handleLayoutChange}
                gridConfig={{
                    cols: COLS,
                    rowHeight: ROW_HEIGHT,
                    margin: [10, 10] as [number, number],
                    containerPadding: [0, 0] as [number, number],
                }}
                dragConfig={{
                    enabled: true,
                    handle: '.grid-drag-handle',
                }}
                resizeConfig={{
                    enabled: true,
                }}
                compactor={verticalCompactor}
                autoSize={true}
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
                <div key="clock">
                    <GridCard title="World Clock">
                        <WorldClock />
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
