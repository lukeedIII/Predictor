import { useEffect, useRef, useState, useCallback } from 'react';
import {
    createChart, createSeriesMarkers,
    CandlestickSeries, LineSeries, HistogramSeries,
    type IChartApi, type Time,
} from 'lightweight-charts';

/* ═══════════════════════ Types ═══════════════════════ */
type Candle = {
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
};

type Trade = {
    direction: string;
    entry_price: number;
    entry_time: string;
    exit_price?: number;
    exit_time?: string;
    pnl_pct?: number;
};

type ChartMode = 'candle' | 'line';

const TIMEFRAMES = [
    { label: '1H', limit: 60, interval: '1m' },    // 60 x 1-min candles = 1 hour
    { label: '4H', limit: 240, interval: '1m' },    // 240 x 1-min  = 4 hours
    { label: '1D', limit: 400, interval: '5m' },    // 400 x 5-min  = ~33 hours
    { label: '3D', limit: 300, interval: '15m' },   // 300 x 15-min = ~3 days
    { label: '1W', limit: 350, interval: '1h' },    // 350 x 1-hour = ~2 weeks
    { label: '1M', limit: 300, interval: '4h' },    // 300 x 4-hour = ~50 days
    { label: '3M', limit: 500, interval: '4h' },    // 500 x 4-hour = ~83 days
];

type Props = {
    candles: Candle[];
    prediction: Record<string, unknown>;
    trades?: Trade[];
    onTimeframeChange?: (limit: number, interval: string) => void;
};

/* ═══════════════════════ Helpers ═══════════════════════ */
function toUnix(ts: string): Time {
    return Math.floor(new Date(ts).getTime() / 1000) as Time;
}

function dedup(candles: Candle[]) {
    const seen = new Set<number>();
    return candles
        .map(c => ({ ...c, time: toUnix(c.timestamp) }))
        .filter(c => {
            const t = c.time as number;
            if (seen.has(t)) return false;
            seen.add(t);
            return true;
        })
        .sort((a, b) => (a.time as number) - (b.time as number));
}

/** Simple Moving Average */
function computeSMA(closes: { time: Time; value: number }[], period: number) {
    const result: { time: Time; value: number }[] = [];
    for (let i = period - 1; i < closes.length; i++) {
        let sum = 0;
        for (let j = i - period + 1; j <= i; j++) sum += closes[j].value;
        result.push({ time: closes[i].time, value: sum / period });
    }
    return result;
}

/** Exponential Moving Average */
function computeEMA(closes: { time: Time; value: number }[], period: number) {
    const result: { time: Time; value: number }[] = [];
    const k = 2 / (period + 1);
    let ema = closes[0]?.value ?? 0;
    for (let i = 0; i < closes.length; i++) {
        ema = i === 0 ? closes[i].value : closes[i].value * k + ema * (1 - k);
        if (i >= period - 1) {
            result.push({ time: closes[i].time, value: ema });
        }
    }
    return result;
}

/** VWAP (Volume Weighted Average Price) */
function computeVWAP(sorted: { time: Time; high: number; low: number; close: number; volume: number }[]) {
    const result: { time: Time; value: number }[] = [];
    let cumVol = 0;
    let cumTP = 0;
    for (const c of sorted) {
        const tp = (c.high + c.low + c.close) / 3;
        cumVol += c.volume;
        cumTP += tp * c.volume;
        if (cumVol > 0) {
            result.push({ time: c.time, value: cumTP / cumVol });
        }
    }
    return result;
}

/** Bollinger Bands */
function computeBollinger(closes: { time: Time; value: number }[], period: number, stdDev: number) {
    const upper: { time: Time; value: number }[] = [];
    const lower: { time: Time; value: number }[] = [];
    for (let i = period - 1; i < closes.length; i++) {
        let sum = 0;
        for (let j = i - period + 1; j <= i; j++) sum += closes[j].value;
        const mean = sum / period;
        let variance = 0;
        for (let j = i - period + 1; j <= i; j++) variance += (closes[j].value - mean) ** 2;
        const std = Math.sqrt(variance / period);
        upper.push({ time: closes[i].time, value: mean + stdDev * std });
        lower.push({ time: closes[i].time, value: mean - stdDev * std });
    }
    return { upper, lower };
}

/* ═══════════════════════ Component ═══════════════════════ */
export default function PriceChart({ candles, prediction, trades = [], onTimeframeChange }: Props) {
    const containerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);

    const [mode, setMode] = useState<ChartMode>('candle');
    const [activeTF, setActiveTF] = useState('4H');
    const [showSMA, setShowSMA] = useState(true);
    const [showBB, setShowBB] = useState(true);
    const [showVWAP, setShowVWAP] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Single effect: create chart + series + overlays
    useEffect(() => {
        const el = containerRef.current;
        if (!el || candles.length < 2) return;

        if (chartRef.current) {
            try { chartRef.current.remove(); } catch { /* */ }
            chartRef.current = null;
        }

        try {
            const chart = createChart(el, {
                width: el.clientWidth,
                height: 700,
                layout: {
                    background: { color: '#0d1117' },
                    textColor: '#566A84',
                    fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                    fontSize: 11,
                },
                grid: {
                    vertLines: { color: 'rgba(255,255,255,0.03)' },
                    horzLines: { color: 'rgba(255,255,255,0.03)' },
                },
                crosshair: {
                    mode: 0,
                    vertLine: { color: 'rgba(129,140,248,0.3)', width: 1, style: 2, labelBackgroundColor: '#1a1f35' },
                    horzLine: { color: 'rgba(129,140,248,0.3)', width: 1, style: 2, labelBackgroundColor: '#1a1f35' },
                },
                rightPriceScale: {
                    borderVisible: false,
                    scaleMargins: { top: 0.05, bottom: 0.15 },
                },
                timeScale: {
                    borderVisible: false,
                    timeVisible: ['1H', '4H', '12H', '1D'].includes(activeTF),
                    secondsVisible: false,
                    rightOffset: 8,
                    barSpacing: ['1M', '3M'].includes(activeTF) ? 3 : ['1W', '3D'].includes(activeTF) ? 4 : 6,
                },
            });
            chartRef.current = chart;

            const sorted = dedup(candles);
            const closes = sorted.map(c => ({ time: c.time, value: c.close }));

            /* ─── 1. Main Price Series ─── */
            let mainSeries: ReturnType<typeof chart.addSeries>;

            if (mode === 'candle') {
                const series = chart.addSeries(CandlestickSeries, {
                    upColor: '#34D399',
                    downColor: '#F87171',
                    borderUpColor: '#34D399',
                    borderDownColor: '#F87171',
                    wickUpColor: 'rgba(52,211,153,0.5)',
                    wickDownColor: 'rgba(248,113,113,0.5)',
                });
                series.setData(sorted.map(c => ({
                    time: c.time, open: c.open, high: c.high, low: c.low, close: c.close,
                })));
                mainSeries = series;

                // Trade markers
                addTradeMarkers(mainSeries, trades);
            } else {
                const series = chart.addSeries(LineSeries, {
                    color: '#818CF8',
                    lineWidth: 2,
                    lastValueVisible: true,
                    priceLineVisible: false,
                });
                series.setData(closes);
                mainSeries = series;
                addTradeMarkers(mainSeries, trades);
            }

            /* ─── 2. SMA 20 + SMA 50 ─── */
            if (showSMA && closes.length >= 50) {
                const sma20 = computeSMA(closes, 20);
                const sma50 = computeSMA(closes, 50);

                const sma20Series = chart.addSeries(LineSeries, {
                    color: 'rgba(251, 191, 36, 0.7)',  // amber
                    lineWidth: 1,
                    lastValueVisible: false,
                    priceLineVisible: false,
                    title: 'SMA 20',
                });
                sma20Series.setData(sma20);

                const sma50Series = chart.addSeries(LineSeries, {
                    color: 'rgba(168, 85, 247, 0.7)',  // purple
                    lineWidth: 1,
                    lastValueVisible: false,
                    priceLineVisible: false,
                    title: 'SMA 50',
                });
                sma50Series.setData(sma50);
            }

            /* ─── 3. Bollinger Bands ─── */
            if (showBB && closes.length >= 20) {
                const bb = computeBollinger(closes, 20, 2);

                const bbUpper = chart.addSeries(LineSeries, {
                    color: 'rgba(99, 102, 241, 0.25)',
                    lineWidth: 1,
                    lineStyle: 2,
                    lastValueVisible: false,
                    priceLineVisible: false,
                });
                bbUpper.setData(bb.upper);

                const bbLower = chart.addSeries(LineSeries, {
                    color: 'rgba(99, 102, 241, 0.25)',
                    lineWidth: 1,
                    lineStyle: 2,
                    lastValueVisible: false,
                    priceLineVisible: false,
                });
                bbLower.setData(bb.lower);
            }

            /* ─── 4. VWAP ─── */
            if (showVWAP && sorted.length > 10) {
                const vwap = computeVWAP(sorted);
                const vwapSeries = chart.addSeries(LineSeries, {
                    color: 'rgba(56, 189, 248, 0.6)',  // sky blue
                    lineWidth: 1,
                    lineStyle: 2,
                    lastValueVisible: false,
                    priceLineVisible: false,
                    title: 'VWAP',
                });
                vwapSeries.setData(vwap);
            }

            /* ─── 5. Fibonacci Retracement Lines (on main series) ─── */
            const prices = sorted.map(c => c.close);
            const pMin = Math.min(...prices);
            const pMax = Math.max(...prices);
            const range = pMax - pMin;

            if (range > 0) {
                const fibLevels = [
                    { ratio: 0, label: '0%', color: 'rgba(107,114,128,0.3)' },
                    { ratio: 0.236, label: '23.6%', color: 'rgba(251,191,36,0.2)' },
                    { ratio: 0.382, label: '38.2%', color: 'rgba(52,211,153,0.2)' },
                    { ratio: 0.5, label: '50%', color: 'rgba(129,140,248,0.25)' },
                    { ratio: 0.618, label: '61.8%', color: 'rgba(52,211,153,0.2)' },
                    { ratio: 0.786, label: '78.6%', color: 'rgba(251,191,36,0.2)' },
                    { ratio: 1, label: '100%', color: 'rgba(107,114,128,0.3)' },
                ];

                for (const fib of fibLevels) {
                    const price = pMin + range * fib.ratio;
                    mainSeries.createPriceLine({
                        price,
                        color: fib.color,
                        lineWidth: 1,
                        lineStyle: 2,
                        axisLabelVisible: false,
                        title: `Fib ${fib.label}`,
                    });
                }
            }

            /* ─── 6. AI Prediction Lines (on main series) ─── */
            const target1h = prediction?.target_price_1h as number | undefined;
            const target2h = prediction?.target_price_2h as number | undefined;
            const currentPrice = prediction?.current_price as number | undefined;
            const direction = prediction?.direction as string | undefined;
            const confidence = prediction?.confidence as number | undefined;

            if (target1h) {
                const isUp = direction?.toUpperCase() !== 'DOWN';
                mainSeries.createPriceLine({
                    price: target1h,
                    color: isUp ? '#34D399' : '#F87171',
                    lineWidth: 2,
                    lineStyle: 2,
                    axisLabelVisible: true,
                    title: `🎯 1H Target`,
                });
            }
            if (target2h) {
                const isUp = direction?.toUpperCase() !== 'DOWN';
                mainSeries.createPriceLine({
                    price: target2h,
                    color: isUp ? 'rgba(52,211,153,0.4)' : 'rgba(248,113,113,0.4)',
                    lineWidth: 1,
                    lineStyle: 3,
                    axisLabelVisible: true,
                    title: `2H Target`,
                });
            }
            if (currentPrice) {
                mainSeries.createPriceLine({
                    price: currentPrice,
                    color: 'rgba(129,140,248,0.5)',
                    lineWidth: 1,
                    lineStyle: 0,
                    axisLabelVisible: true,
                    title: `Live`,
                });
            }

            /* ─── 7. Regime Zone Label (CSS watermark) ─── */
            const regime = prediction?.regime_label as string | undefined;
            const watermarkEl = el.querySelector('.chart-watermark') as HTMLElement | null;
            if (watermarkEl) watermarkEl.remove();
            if (regime) {
                const regimeColor = regime === 'TRENDING_UP' ? '#34D399'
                    : regime === 'TRENDING_DOWN' ? '#F87171'
                        : '#818CF8';
                const wm = document.createElement('div');
                wm.className = 'chart-watermark';
                wm.textContent = `${regime}${confidence ? ` · ${confidence.toFixed(0)}%` : ''}${direction ? ` · AI: ${direction}` : ''}`;
                Object.assign(wm.style, {
                    position: 'absolute',
                    top: '50%', left: '50%',
                    transform: 'translate(-50%, -50%)',
                    fontSize: '42px', fontWeight: '800',
                    fontFamily: "'JetBrains Mono', monospace",
                    color: regimeColor,
                    opacity: '0.06',
                    pointerEvents: 'none',
                    whiteSpace: 'nowrap',
                    letterSpacing: '2px',
                    zIndex: '1',
                });
                el.style.position = 'relative';
                el.appendChild(wm);
            }

            /* ─── 8. Volume Histogram ─── */
            const volSeries = chart.addSeries(HistogramSeries, {
                priceFormat: { type: 'volume' },
                priceScaleId: 'vol',
            });
            chart.priceScale('vol').applyOptions({
                scaleMargins: { top: 0.88, bottom: 0 },
            });
            volSeries.setData(sorted.map(c => ({
                time: c.time,
                value: c.volume,
                color: c.close >= c.open ? 'rgba(52,211,153,0.12)' : 'rgba(248,113,113,0.12)',
            })));

            /* ─── 9. EMA 9 (fast signal line) ─── */
            if (closes.length >= 9) {
                const ema9 = computeEMA(closes, 9);
                const ema9Series = chart.addSeries(LineSeries, {
                    color: 'rgba(236, 72, 153, 0.5)',  // pink
                    lineWidth: 1,
                    lastValueVisible: false,
                    priceLineVisible: false,
                    title: 'EMA 9',
                });
                ema9Series.setData(ema9);
            }

            chart.timeScale().fitContent();
            setError(null);

            const onResize = () => chart.applyOptions({ width: el.clientWidth });
            window.addEventListener('resize', onResize);

            return () => {
                window.removeEventListener('resize', onResize);
                try { chart.remove(); } catch { /* */ }
                chartRef.current = null;
            };
        } catch (err) {
            console.error('[PriceChart]', err);
            setError(String(err));
        }
    }, [candles, mode, trades, prediction, showSMA, showBB, showVWAP, activeTF]);

    const handleTF = useCallback((label: string, limit: number, interval: string) => {
        setActiveTF(label);
        onTimeframeChange?.(limit, interval);
    }, [onTimeframeChange]);

    if (candles.length < 2) {
        return (
            <div className="empty-state">
                <div className="empty-state-icon">📊</div>
                <div>Awaiting market data...</div>
            </div>
        );
    }

    return (
        <div>
            {/* ── Top Toolbar ── */}
            <div style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                marginBottom: 8, flexWrap: 'wrap', gap: 6,
            }}>
                {/* Timeframes */}
                <div style={{ display: 'flex', gap: 3 }}>
                    {TIMEFRAMES.map(tf => (
                        <button key={tf.label} onClick={() => handleTF(tf.label, tf.limit, tf.interval)}
                            style={btnStyle(activeTF === tf.label)}>
                            {tf.label}
                        </button>
                    ))}
                </div>

                {/* Indicators + chart mode */}
                <div style={{ display: 'flex', gap: 3, alignItems: 'center' }}>
                    <button onClick={() => setShowSMA(v => !v)} style={indicatorBtn(showSMA, '#FBBF24')}>
                        SMA
                    </button>
                    <button onClick={() => setShowBB(v => !v)} style={indicatorBtn(showBB, '#6366F1')}>
                        BB
                    </button>
                    <button onClick={() => setShowVWAP(v => !v)} style={indicatorBtn(showVWAP, '#38BDF8')}>
                        VWAP
                    </button>
                    <div style={{ width: 1, height: 16, background: 'rgba(255,255,255,0.08)', margin: '0 4px' }} />
                    <button onClick={() => setMode('candle')} style={btnStyle(mode === 'candle')}>🕯 Candle</button>
                    <button onClick={() => setMode('line')} style={btnStyle(mode === 'line')}>📈 Line</button>
                    {trades.length > 0 && (
                        <span style={{ fontSize: 10, color: '#4A5568', marginLeft: 4, fontFamily: 'monospace' }}>
                            {trades.length} trades
                        </span>
                    )}
                </div>
            </div>

            {/* ── Legend ── */}
            <div style={{
                display: 'flex', gap: 12, marginBottom: 6, fontSize: 10,
                fontFamily: "'JetBrains Mono', monospace", color: '#566A84', flexWrap: 'wrap',
            }}>
                {showSMA && <>
                    <span><span style={{ color: '#FBBF24' }}>━</span> SMA 20</span>
                    <span><span style={{ color: '#A855F7' }}>━</span> SMA 50</span>
                </>}
                <span><span style={{ color: '#EC4899' }}>━</span> EMA 9</span>
                {showBB && <span><span style={{ color: '#6366F1' }}>┈</span> Bollinger</span>}
                {showVWAP && <span><span style={{ color: '#38BDF8' }}>┈</span> VWAP</span>}
                <span><span style={{ color: 'rgba(129,140,248,0.3)' }}>┈</span> Fibonacci</span>
            </div>

            {error && (
                <div style={{ color: '#F87171', fontSize: 12, padding: 8, fontFamily: 'monospace' }}>
                    Chart error: {error}
                </div>
            )}

            {/* ── Chart Canvas ── */}
            <div ref={containerRef} style={{
                width: '100%', height: 700, borderRadius: 8,
                overflow: 'hidden', background: '#0d1117',
            }} />
        </div>
    );
}

/* ═══════════════════════ Utilities ═══════════════════════ */
function addTradeMarkers(series: ReturnType<IChartApi['addSeries']>, trades: Trade[]) {
    if (trades.length === 0) return;
    const markers = trades
        .map(t => {
            const isLong = t.direction?.toUpperCase() === 'LONG' || t.direction?.toUpperCase() === 'UP';
            // Color based on PnL: green = profit, red = loss. Fallback to direction for open trades.
            const hasPnl = t.pnl_pct !== undefined && t.pnl_pct !== null;
            const isProfit = hasPnl ? t.pnl_pct! > 0 : isLong;
            const markerColor = isProfit ? '#34D399' : '#F87171';
            return {
                time: toUnix(t.entry_time),
                position: isLong ? 'belowBar' as const : 'aboveBar' as const,
                color: markerColor,
                shape: isLong ? 'arrowUp' as const : 'arrowDown' as const,
                text: `${isLong ? 'LONG' : 'SHORT'}${hasPnl ? ` ${t.pnl_pct! > 0 ? '+' : ''}${t.pnl_pct!.toFixed(2)}%` : ''}`,
            };
        })
        .sort((a, b) => (a.time as number) - (b.time as number));
    try { createSeriesMarkers(series, markers); } catch { /* */ }
}

function btnStyle(active: boolean): React.CSSProperties {
    return {
        background: active ? 'rgba(129,140,248,0.15)' : 'transparent',
        border: active ? '1px solid rgba(129,140,248,0.35)' : '1px solid rgba(255,255,255,0.05)',
        color: active ? '#818CF8' : '#566A84',
        borderRadius: 5, padding: '4px 10px', fontSize: 11, fontWeight: 600,
        fontFamily: "'JetBrains Mono', monospace",
        cursor: 'pointer', transition: 'all 0.15s ease', letterSpacing: '0.3px',
    };
}

function indicatorBtn(active: boolean, color: string): React.CSSProperties {
    return {
        background: active ? `${color}15` : 'transparent',
        border: active ? `1px solid ${color}55` : '1px solid rgba(255,255,255,0.05)',
        color: active ? color : '#566A84',
        borderRadius: 5, padding: '4px 8px', fontSize: 10, fontWeight: 700,
        fontFamily: "'JetBrains Mono', monospace",
        cursor: 'pointer', transition: 'all 0.15s ease', letterSpacing: '0.5px',
    };
}
