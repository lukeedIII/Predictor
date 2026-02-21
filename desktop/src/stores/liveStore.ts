/**
 * Live WebSocket Store — External Store with useSyncExternalStore
 * ================================================================
 * Replaces the old WebSocketProvider/Context pattern that caused
 * full-tree re-renders on every WS tick.
 *
 * Components subscribe to only the slices of state they need via
 * selector hooks (useLivePrice, useLivePositions, etc.), so only
 * the affected component re-renders.
 */
import { useSyncExternalStore, useRef, useLayoutEffect } from 'react';

// ─── Types ──────────────────────────────────────
export type WSState = {
    price: number | null;
    prediction: Record<string, unknown> | null;
    quant: Record<string, unknown> | null;
    alt_signals: Record<string, unknown> | null;
    accuracy: number | null;
    accuracy_source: 'live' | 'training' | null;
    live_accuracy_samples: number;
    positions: Array<Record<string, unknown>>;
    stats: Record<string, unknown> | null;
    bot_running: boolean;
    connected: boolean;
    timestamp: string | null;
    change_24h: number | null;
    change_pct: number | null;
    high_24h: number | null;
    low_24h: number | null;
    volume_btc: number | null;
    volume_usdt: number | null;
    ws_connected: boolean;
};

const DEFAULT_STATE: WSState = {
    price: null,
    prediction: null,
    quant: null,
    alt_signals: null,
    accuracy: null,
    accuracy_source: null,
    live_accuracy_samples: 0,
    positions: [],
    stats: null,
    bot_running: false,
    connected: false,
    timestamp: null,
    change_24h: null,
    change_pct: null,
    high_24h: null,
    low_24h: null,
    volume_btc: null,
    volume_usdt: null,
    ws_connected: false,
};

// ─── External Store (module-level singleton) ────
let state: WSState = { ...DEFAULT_STATE };
const listeners = new Set<() => void>();

function getSnapshot(): WSState {
    return state;
}

function subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
}

function setState(partial: Partial<WSState>) {
    state = { ...state, ...partial };
    listeners.forEach((l) => l());
}

// ─── WebSocket Connection ───────────────────────
const WS_URL = 'ws://127.0.0.1:8420/ws/live';
const MAX_BACKOFF_MS = 30_000;
const BASE_DELAY_MS = 1_000;

let ws: WebSocket | null = null;
let backoff = BASE_DELAY_MS;
let mounted = false;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

function scheduleReconnect() {
    if (!mounted) return;
    const jitter = Math.random() * 500;
    const delay = backoff + jitter;
    backoff = Math.min(backoff * 2, MAX_BACKOFF_MS);
    reconnectTimer = setTimeout(connect, delay);
}

function connect() {
    if (!mounted) return;

    try {
        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            if (!mounted) return;
            backoff = BASE_DELAY_MS;
            setState({ connected: true });
        };

        ws.onmessage = (event) => {
            if (!mounted) return;
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'state_update') {
                    setState({
                        price: data.price ?? state.price,
                        prediction: data.prediction ?? state.prediction,
                        quant: data.quant ?? state.quant,
                        alt_signals: data.alt_signals ?? state.alt_signals,
                        accuracy: data.accuracy ?? state.accuracy,
                        accuracy_source: data.accuracy_source ?? state.accuracy_source,
                        live_accuracy_samples: data.live_accuracy_samples ?? state.live_accuracy_samples,
                        positions: data.positions ?? state.positions,
                        stats: data.stats ?? state.stats,
                        bot_running: data.bot_running ?? state.bot_running,
                        timestamp: data.timestamp ?? state.timestamp,
                        change_24h: data.change_24h ?? state.change_24h,
                        change_pct: data.change_pct ?? state.change_pct,
                        high_24h: data.high_24h ?? state.high_24h,
                        low_24h: data.low_24h ?? state.low_24h,
                        volume_btc: data.volume_btc ?? state.volume_btc,
                        volume_usdt: data.volume_usdt ?? state.volume_usdt,
                        ws_connected: data.ws_connected ?? state.ws_connected,
                    });
                }
            } catch {
                // Malformed message, skip
            }
        };

        ws.onclose = () => {
            if (!mounted) return;
            setState({ connected: false });
            scheduleReconnect();
        };

        ws.onerror = () => {
            ws?.close();
        };
    } catch {
        scheduleReconnect();
    }
}

/** Start the WS connection. Call once at app mount. */
export function connectLiveWS(): () => void {
    mounted = true;
    connect();

    return () => {
        mounted = false;
        if (reconnectTimer) clearTimeout(reconnectTimer);
        if (ws) {
            ws.onclose = null; // prevent reconnect on cleanup
            ws.close();
        }
        // Reset to default so next mount starts clean
        state = { ...DEFAULT_STATE };
    };
}

// ─── Selector Hooks ─────────────────────────────
// Each hook re-renders only when its selected slice changes.

/**
 * Generic selector: subscribe to a computed slice of WS state.
 * Uses referential equality — careful with object selectors
 * (prefer primitive selectors or stable sub-objects).
 */
export function useLiveSelector<T>(selector: (s: WSState) => T): T {
    const selectorRef = useRef(selector);
    useLayoutEffect(() => {
        selectorRef.current = selector;
    });

    const snap = useSyncExternalStore(
        subscribe,
        () => selectorRef.current(getSnapshot()),
    );
    return snap;
}

/** Full WS state — use sparingly (re-renders on every tick). */
export function useLiveAll(): WSState {
    return useSyncExternalStore(subscribe, getSnapshot);
}

/** Just the live BTC price. */
export function useLivePrice() {
    return useSyncExternalStore(subscribe, () => getSnapshot().price);
}

/** 24h change percent. */
export function useLiveChangePct() {
    return useSyncExternalStore(subscribe, () => getSnapshot().change_pct);
}

/** Connection status (our WS + Binance WS). */
export function useLiveConnected() {
    const connected = useSyncExternalStore(subscribe, () => getSnapshot().connected);
    const wsConnected = useSyncExternalStore(subscribe, () => getSnapshot().ws_connected);
    return { connected, wsConnected };
}

/** Bot running status. */
export function useLiveBotStatus() {
    return useSyncExternalStore(subscribe, () => getSnapshot().bot_running);
}

/** Positions array. */
export function useLivePositions() {
    return useSyncExternalStore(subscribe, () => getSnapshot().positions);
}

/** Stats object. */
export function useLiveStats() {
    return useSyncExternalStore(subscribe, () => getSnapshot().stats);
}

/** Prediction data. */
export function useLivePrediction() {
    return useSyncExternalStore(subscribe, () => getSnapshot().prediction);
}

/** Quant analysis data. */
export function useLiveQuant() {
    return useSyncExternalStore(subscribe, () => getSnapshot().quant);
}

/** Accuracy value. */
export function useLiveAccuracy() {
    return useSyncExternalStore(subscribe, () => getSnapshot().accuracy);
}

/** Accuracy source: 'live' or 'training'. */
export function useLiveAccuracySource() {
    return useSyncExternalStore(subscribe, () => getSnapshot().accuracy_source);
}

/** Number of live predictions validated. */
export function useLiveAccuracySamples() {
    return useSyncExternalStore(subscribe, () => getSnapshot().live_accuracy_samples);
}

/** Alt signals. */
export function useLiveAltSignals() {
    return useSyncExternalStore(subscribe, () => getSnapshot().alt_signals);
}

/** 24h high price. */
export function useLiveHigh24h() {
    return useSyncExternalStore(subscribe, () => getSnapshot().high_24h);
}

/** 24h low price. */
export function useLiveLow24h() {
    return useSyncExternalStore(subscribe, () => getSnapshot().low_24h);
}

/** 24h BTC volume. */
export function useLiveVolumeBtc() {
    return useSyncExternalStore(subscribe, () => getSnapshot().volume_btc);
}
