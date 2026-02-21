/**
 * TradingViewChart — Professional candlestick chart using TradingView Lightweight Charts v5.
 * 
 * Features:
 *  - Candlestick series with volume histogram overlay
 *  - MA(7), MA(25), MA(99) line overlays
 *  - Real-time candle updates via WebSocket
 *  - Multi-timeframe selector (1m, 5m, 15m, 1H, 4H, 1D)
 *  - Crosshair with OHLCV tooltip
 *  - Dark theme matching our design system
 *  - Auto-resize with ResizeObserver
 * 
 * Based on: lightweight-charts v5 API
 *   - chart.addSeries(CandlestickSeries) — NOT addCandlestickSeries()
 *   - series.update() for real-time — NOT setData()
 *   - Time format: UNIX seconds (NOT milliseconds)
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import {
    createChart,
    CandlestickSeries,
    HistogramSeries,
    LineSeries,
    type IChartApi,
    type ISeriesApi,
    type CandlestickData,
    type UTCTimestamp,
} from 'lightweight-charts';
import { useLivePrice, useLiveChangePct, useLiveHigh24h, useLiveLow24h, useLiveVolumeBtc } from '../stores/liveStore';
import { IconWarning } from './Icons';

// ─── Types ──────────────────────────────────────────
type Timeframe = '1m' | '5m' | '15m' | '1h' | '4h' | '1d';

type OHLCVRow = {
    time: UTCTimestamp;
    open: number;
    high: number;
    low: number;
    close: number;
    volume?: number;
};

type ChartRefs = {
    chart: IChartApi | null;
    candleSeries: ISeriesApi<'Candlestick'> | null;
    volumeSeries: ISeriesApi<'Histogram'> | null;
    ma7Series: ISeriesApi<'Line'> | null;
    ma25Series: ISeriesApi<'Line'> | null;
    ma99Series: ISeriesApi<'Line'> | null;
    trajectorySeries: ISeriesApi<'Line'> | null;
};

type TrajectoryData = {
    trajectory: { time: number; value: number }[];
    anchor: { time: number; value: number };
    direction: 'UP' | 'FLAT' | 'DOWN';
    confidence: number;
};

const TIMEFRAMES: { label: string; value: Timeframe }[] = [
    { label: '1m', value: '1m' },
    { label: '5m', value: '5m' },
    { label: '15m', value: '15m' },
    { label: '1H', value: '1h' },
    { label: '4H', value: '4h' },
    { label: '1D', value: '1d' },
];

const API_BASE = 'http://127.0.0.1:8420';

// ─── Helpers ────────────────────────────────────────
function calculateMA(data: OHLCVRow[], period: number): { time: UTCTimestamp; value: number }[] {
    const result: { time: UTCTimestamp; value: number }[] = [];
    for (let i = period - 1; i < data.length; i++) {
        let sum = 0;
        for (let j = 0; j < period; j++) {
            sum += data[i - j].close;
        }
        result.push({ time: data[i].time, value: sum / period });
    }
    return result;
}

// ─── Component ──────────────────────────────────────
export default function TradingViewChart() {
    const containerRef = useRef<HTMLDivElement>(null);
    const chartRefs = useRef<ChartRefs>({
        chart: null,
        candleSeries: null,
        volumeSeries: null,
        ma7Series: null,
        ma25Series: null,
        ma99Series: null,
        trajectorySeries: null,
    });

    const [timeframe, setTimeframe] = useState<Timeframe>('1m');
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [crosshairData, setCrosshairData] = useState<CandlestickData | null>(null);
    const [trajectoryInfo, setTrajectoryInfo] = useState<TrajectoryData | null>(null);
    // Granular selectors — chart only re-renders on price/stats changes
    const livePrice = useLivePrice();
    const changePct = useLiveChangePct();
    const high24h = useLiveHigh24h();
    const low24h = useLiveLow24h();
    const volumeBtc = useLiveVolumeBtc();

    // Track current candle for real-time updates
    const currentCandleRef = useRef<OHLCVRow | null>(null);

    // ─── Chart Initialization ──────────────────────────
    useEffect(() => {
        if (!containerRef.current) return;

        const chart = createChart(containerRef.current, {
            width: containerRef.current.clientWidth,
            height: containerRef.current.clientHeight || 400,
            layout: {
                background: { color: '#0A0E17' },
                textColor: '#9CA3AF',
                fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
                fontSize: 11,
            },
            grid: {
                vertLines: { color: 'rgba(31, 41, 55, 0.5)' },
                horzLines: { color: 'rgba(31, 41, 55, 0.5)' },
            },
            crosshair: {
                mode: 0, // CrosshairMode.Normal
                vertLine: { color: 'rgba(255, 255, 255, 0.15)', width: 1, style: 2 },
                horzLine: { color: 'rgba(255, 255, 255, 0.15)', width: 1, style: 2 },
            },
            rightPriceScale: {
                borderColor: 'rgba(31, 41, 55, 0.8)',
                scaleMargins: { top: 0.05, bottom: 0.25 },
            },
            timeScale: {
                borderColor: 'rgba(31, 41, 55, 0.8)',
                timeVisible: true,
                secondsVisible: timeframe === '1m',
                fixLeftEdge: true,
                fixRightEdge: true,
            },
        });

        // Candlestick series — Binance-style colors
        const candleSeries = chart.addSeries(CandlestickSeries, {
            upColor: '#0ECB81',
            downColor: '#F6465D',
            borderUpColor: '#0ECB81',
            borderDownColor: '#F6465D',
            wickUpColor: '#0ECB81',
            wickDownColor: '#F6465D',
        });

        // Volume histogram — bottom 20%
        const volumeSeries = chart.addSeries(HistogramSeries, {
            priceFormat: { type: 'volume' },
            priceScaleId: 'volume',
        });
        chart.priceScale('volume').applyOptions({
            scaleMargins: { top: 0.8, bottom: 0 },
        });

        // MA(7) — yellow
        const ma7Series = chart.addSeries(LineSeries, {
            color: '#F0B90B',
            lineWidth: 1,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
        });

        // MA(25) — pink
        const ma25Series = chart.addSeries(LineSeries, {
            color: '#E040FB',
            lineWidth: 1,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
        });

        // MA(99) — cyan
        const ma99Series = chart.addSeries(LineSeries, {
            color: '#00BCD4',
            lineWidth: 1,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
        });

        // AI Trajectory — dashed blue-grey ghost line
        const trajectorySeries = chart.addSeries(LineSeries, {
            color: 'rgba(120, 180, 255, 0.6)',
            lineWidth: 2,
            lineStyle: 2, // Dashed
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
            pointMarkersVisible: true,
        });

        // Crosshair tooltip
        chart.subscribeCrosshairMove((param) => {
            if (param.time && param.seriesData) {
                const candleData = param.seriesData.get(candleSeries);
                if (candleData && 'open' in candleData) {
                    setCrosshairData(candleData as CandlestickData);
                }
            } else {
                setCrosshairData(null);
            }
        });

        chartRefs.current = { chart, candleSeries, volumeSeries, ma7Series, ma25Series, ma99Series, trajectorySeries };

        // ResizeObserver for responsive
        const observer = new ResizeObserver(entries => {
            const { width, height } = entries[0].contentRect;
            chart.applyOptions({ width, height });
        });
        observer.observe(containerRef.current);

        return () => {
            observer.disconnect();
            chart.remove();
            chartRefs.current = {
                chart: null, candleSeries: null, volumeSeries: null,
                ma7Series: null, ma25Series: null, ma99Series: null,
                trajectorySeries: null,
            };
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []); // Chart created once


    // ─── Data Loading ─────────────────────────────────
    const loadData = useCallback(async (tf: Timeframe) => {
        setIsLoading(true);
        setError(null);

        try {
            const res = await fetch(`${API_BASE}/api/market-data?interval=${tf}&limit=500`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);

            const result = await res.json();
            const candles: OHLCVRow[] = (result.candles || []).map((c: Record<string, unknown>) => ({
                time: (Math.floor(Number(c.timestamp || c.time) / 1000)) as UTCTimestamp,
                open: Number(c.open),
                high: Number(c.high),
                low: Number(c.low),
                close: Number(c.close),
                volume: Number(c.volume || 0),
            }));

            if (candles.length === 0) {
                setError('No data available');
                setIsLoading(false);
                return;
            }

            // Sort by time ascending
            candles.sort((a, b) => (a.time as number) - (b.time as number));

            // Deduplicate by time
            const seen = new Set<number>();
            const uniqueCandles = candles.filter(c => {
                if (seen.has(c.time as number)) return false;
                seen.add(c.time as number);
                return true;
            });

            const refs = chartRefs.current;
            if (!refs.candleSeries) return;

            // Set data
            refs.candleSeries.setData(uniqueCandles);

            // Volume with color
            refs.volumeSeries?.setData(
                uniqueCandles.map(c => ({
                    time: c.time,
                    value: c.volume || 0,
                    color: c.close >= c.open
                        ? 'rgba(14, 203, 129, 0.3)'
                        : 'rgba(246, 70, 93, 0.3)',
                }))
            );

            // MAs
            refs.ma7Series?.setData(calculateMA(uniqueCandles, 7));
            refs.ma25Series?.setData(calculateMA(uniqueCandles, 25));
            if (uniqueCandles.length >= 99) {
                refs.ma99Series?.setData(calculateMA(uniqueCandles, 99));
            }

            // Track the latest candle for real-time updates
            currentCandleRef.current = uniqueCandles[uniqueCandles.length - 1];

            // Scroll to last candle
            refs.chart?.timeScale().scrollToPosition(5, false);

        } catch (e) {
            console.error('Chart data load failed:', e);
            setError(e instanceof Error ? e.message : 'Failed to load data');
        } finally {
            setIsLoading(false);
        }
    }, []);

    // Load data when timeframe changes
    useEffect(() => {
        loadData(timeframe);
    }, [timeframe, loadData]);

    // Update timeScale seconds visibility
    useEffect(() => {
        chartRefs.current.chart?.timeScale().applyOptions({
            secondsVisible: timeframe === '1m',
        });
    }, [timeframe]);


    // ─── Real-Time Candle Update ──────────────────────
    useEffect(() => {
        if (!livePrice || livePrice <= 0) return;
        const refs = chartRefs.current;
        if (!refs.candleSeries || !currentCandleRef.current) return;

        const now = Math.floor(Date.now() / 1000);
        const tfSeconds = getTimeframeSeconds(timeframe);
        const candleStart = Math.floor(now / tfSeconds) * tfSeconds;

        const current = currentCandleRef.current;

        if ((current.time as number) === candleStart) {
            // Update existing candle
            current.high = Math.max(current.high, livePrice);
            current.low = Math.min(current.low, livePrice);
            current.close = livePrice;
        } else if (candleStart > (current.time as number)) {
            // New candle
            const newCandle: OHLCVRow = {
                time: candleStart as UTCTimestamp,
                open: livePrice,
                high: livePrice,
                low: livePrice,
                close: livePrice,
            };
            currentCandleRef.current = newCandle;
        }

        // Push update to chart
        refs.candleSeries.update({
            time: currentCandleRef.current.time,
            open: currentCandleRef.current.open,
            high: currentCandleRef.current.high,
            low: currentCandleRef.current.low,
            close: currentCandleRef.current.close,
        });

        // Update volume bar color
        refs.volumeSeries?.update({
            time: currentCandleRef.current.time,
            value: currentCandleRef.current.volume || 0,
            color: currentCandleRef.current.close >= currentCandleRef.current.open
                ? 'rgba(14, 203, 129, 0.3)'
                : 'rgba(246, 70, 93, 0.3)',
        });
    }, [livePrice, timeframe]);


    // ─── AI Trajectory Polling ────────────────────────
    useEffect(() => {
        let cancelled = false;

        const fetchTrajectory = async () => {
            try {
                const res = await fetch(`${API_BASE}/api/prediction/trajectory?interval=${timeframe}&steps=5`);
                if (!res.ok) return;
                const data: TrajectoryData = await res.json();
                if (cancelled) return;

                setTrajectoryInfo(data);

                const refs = chartRefs.current;
                if (!refs.trajectorySeries) return;

                // Build line: perfectly sew the anchor point to the live chart's current candle close
                // to eliminate 'gaps' caused by slight async delays between the HTTP fetch and the WebSocket.
                const anchorValue = currentCandleRef.current ? currentCandleRef.current.close : data.anchor.value;

                const points = [
                    { time: data.anchor.time as UTCTimestamp, value: anchorValue },
                    ...data.trajectory.map(p => ({
                        time: p.time as UTCTimestamp,
                        value: p.value,
                    })),
                ];

                // Color based on direction
                const color = data.direction === 'UP'
                    ? 'rgba(14, 203, 129, 0.5)'
                    : data.direction === 'DOWN'
                        ? 'rgba(246, 70, 93, 0.5)'
                        : 'rgba(156, 163, 175, 0.5)';

                refs.trajectorySeries.applyOptions({ color });
                refs.trajectorySeries.setData(points);
            } catch {
                // Silently fail — trajectory is non-critical
            }
        };

        fetchTrajectory();
        const interval = setInterval(fetchTrajectory, 30_000); // Every 30s

        return () => {
            cancelled = true;
            clearInterval(interval);
        };
    }, [timeframe]);


    // ─── Render ───────────────────────────────────────
    return (
        <div className="tv-chart-wrapper">
            {/* Header bar */}
            <div className="tv-chart-header">
                <div className="tv-chart-symbol">
                    <span className="tv-chart-pair">BTC/USDT</span>
                    {livePrice !== null && livePrice > 0 && (
                        <span className="tv-chart-price mono" style={{
                            color: (changePct ?? 0) >= 0 ? '#0ECB81' : '#F6465D'
                        }}>
                            ${livePrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </span>
                    )}
                </div>

                {/* Timeframe selector */}
                <div className="tv-chart-timeframes">
                    {TIMEFRAMES.map(tf => (
                        <button
                            key={tf.value}
                            className={`tv-tf-btn ${timeframe === tf.value ? 'active' : ''}`}
                            onClick={() => setTimeframe(tf.value)}
                        >
                            {tf.label}
                        </button>
                    ))}
                </div>

                {/* 24h stats */}
                <div className="tv-chart-stats">
                    {high24h !== null && (
                        <span className="tv-stat">
                            <span className="tv-stat-label">24h H</span>
                            <span className="tv-stat-value mono">${high24h.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
                        </span>
                    )}
                    {low24h !== null && (
                        <span className="tv-stat">
                            <span className="tv-stat-label">24h L</span>
                            <span className="tv-stat-value mono">${low24h.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
                        </span>
                    )}
                    {volumeBtc !== null && (
                        <span className="tv-stat">
                            <span className="tv-stat-label">Vol</span>
                            <span className="tv-stat-value mono">{volumeBtc.toLocaleString(undefined, { maximumFractionDigits: 0 })} BTC</span>
                        </span>
                    )}
                </div>
            </div>

            {/* Crosshair tooltip */}
            {crosshairData && (
                <div className="tv-chart-tooltip">
                    <span>O<span className="mono">{Number(crosshairData.open).toFixed(2)}</span></span>
                    <span>H<span className="mono">{Number(crosshairData.high).toFixed(2)}</span></span>
                    <span>L<span className="mono">{Number(crosshairData.low).toFixed(2)}</span></span>
                    <span>C<span className="mono">{Number(crosshairData.close).toFixed(2)}</span></span>
                </div>
            )}

            {/* MA Legend */}
            <div className="tv-chart-ma-legend">
                <span style={{ color: '#F0B90B' }}>MA(7)</span>
                <span style={{ color: '#E040FB' }}>MA(25)</span>
                <span style={{ color: '#00BCD4' }}>MA(99)</span>
                {trajectoryInfo && (
                    <span style={{
                        color: trajectoryInfo.direction === 'UP' ? 'rgba(14, 203, 129, 0.8)'
                            : trajectoryInfo.direction === 'DOWN' ? 'rgba(246, 70, 93, 0.8)'
                                : 'rgba(156, 163, 175, 0.8)',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 4,
                    }}>
                        <span style={{
                            width: 14, height: 2,
                            borderTop: '2px dashed currentColor',
                            display: 'inline-block',
                        }} />
                        AI Forecast
                        <span style={{
                            fontSize: 9,
                            padding: '1px 4px',
                            borderRadius: 3,
                            background: 'rgba(255,255,255,0.08)',
                            fontWeight: 600,
                        }}>
                            {trajectoryInfo.direction} {(trajectoryInfo.confidence * 100).toFixed(0)}%
                        </span>
                    </span>
                )}
            </div>

            {/* Chart container */}
            <div
                ref={containerRef}
                className="tv-chart-container"
                style={{ position: 'relative', width: '100%', flex: 1 }}
            />

            {/* Loading overlay */}
            {isLoading && (
                <div className="tv-chart-loading">
                    <div className="tv-loading-spinner" />
                    Loading chart data...
                </div>
            )}

            {/* Error overlay */}
            {error && !isLoading && (
                <div className="tv-chart-error">
                    <IconWarning size={14} style={{ marginRight: 5, verticalAlign: -2 }} /> {error}
                    <button className="tv-retry-btn" onClick={() => loadData(timeframe)}>
                        Retry
                    </button>
                </div>
            )}
        </div>
    );
}


// ─── Utility ────────────────────────────────────────
function getTimeframeSeconds(tf: Timeframe): number {
    switch (tf) {
        case '1m': return 60;
        case '5m': return 300;
        case '15m': return 900;
        case '1h': return 3600;
        case '4h': return 14400;
        case '1d': return 86400;
        default: return 60;
    }
}
