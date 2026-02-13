import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from 'react';

/* ─── Types ──────────────────────────────────────── */
export type WSState = {
    price: number | null;
    prediction: Record<string, unknown> | null;
    quant: Record<string, unknown> | null;
    alt_signals: Record<string, unknown> | null;
    accuracy: number | null;
    positions: Array<Record<string, unknown>>;
    stats: Record<string, unknown> | null;
    bot_running: boolean;
    connected: boolean;
    timestamp: string | null;
    // New: 24h market data from Binance WebSocket
    change_24h: number | null;
    change_pct: number | null;
    high_24h: number | null;
    low_24h: number | null;
    volume_btc: number | null;
    volume_usdt: number | null;
    ws_connected: boolean; // Binance WS health (separate from our WS)
};

const DEFAULT_STATE: WSState = {
    price: null,
    prediction: null,
    quant: null,
    alt_signals: null,
    accuracy: null,
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

/* ─── Context ────────────────────────────────────── */
const WebSocketContext = createContext<WSState>(DEFAULT_STATE);

export function useWebSocket(): WSState {
    return useContext(WebSocketContext);
}

/* ─── Provider ───────────────────────────────────── */
const WS_URL = 'ws://127.0.0.1:8420/ws/live';
const MAX_BACKOFF_MS = 30_000;
const BASE_DELAY_MS = 1_000;

export function WebSocketProvider({ children }: { children: ReactNode }) {
    const [state, setState] = useState<WSState>(DEFAULT_STATE);
    const wsRef = useRef<WebSocket | null>(null);
    const backoffRef = useRef(BASE_DELAY_MS);
    const mountedRef = useRef(true);
    const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    useEffect(() => {
        mountedRef.current = true;

        function connect() {
            if (!mountedRef.current) return;

            try {
                const ws = new WebSocket(WS_URL);
                wsRef.current = ws;

                ws.onopen = () => {
                    if (!mountedRef.current) return;
                    backoffRef.current = BASE_DELAY_MS; // reset
                    setState(prev => ({ ...prev, connected: true }));
                };

                ws.onmessage = (event) => {
                    if (!mountedRef.current) return;
                    try {
                        const data = JSON.parse(event.data);
                        if (data.type === 'state_update') {
                            setState(prev => ({
                                ...prev,
                                price: data.price ?? prev.price,
                                prediction: data.prediction ?? prev.prediction,
                                quant: data.quant ?? prev.quant,
                                alt_signals: data.alt_signals ?? prev.alt_signals,
                                accuracy: data.accuracy ?? prev.accuracy,
                                positions: data.positions ?? prev.positions,
                                stats: data.stats ?? prev.stats,
                                bot_running: data.bot_running ?? prev.bot_running,
                                timestamp: data.timestamp ?? prev.timestamp,
                                // New 24h market data
                                change_24h: data.change_24h ?? prev.change_24h,
                                change_pct: data.change_pct ?? prev.change_pct,
                                high_24h: data.high_24h ?? prev.high_24h,
                                low_24h: data.low_24h ?? prev.low_24h,
                                volume_btc: data.volume_btc ?? prev.volume_btc,
                                volume_usdt: data.volume_usdt ?? prev.volume_usdt,
                                ws_connected: data.ws_connected ?? prev.ws_connected,
                            }));
                        }
                    } catch {
                        // Malformed message, skip
                    }
                };

                ws.onclose = () => {
                    if (!mountedRef.current) return;
                    setState(prev => ({ ...prev, connected: false }));
                    scheduleReconnect();
                };

                ws.onerror = () => {
                    // onclose will fire after onerror
                    ws.close();
                };
            } catch {
                scheduleReconnect();
            }
        }

        function scheduleReconnect() {
            if (!mountedRef.current) return;
            const jitter = Math.random() * 500;
            const delay = backoffRef.current + jitter;
            backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS);
            reconnectTimerRef.current = setTimeout(connect, delay);
        }

        connect();

        return () => {
            mountedRef.current = false;
            if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
            if (wsRef.current) {
                wsRef.current.onclose = null; // prevent reconnect on cleanup
                wsRef.current.close();
            }
        };
    }, []);

    return (
        <WebSocketContext.Provider value={state}>
            {children}
        </WebSocketContext.Provider>
    );
}
