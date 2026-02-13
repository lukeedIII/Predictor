import { useState, useEffect, useCallback, useRef } from 'react';

const API_BASE = 'http://127.0.0.1:8420';

const MAX_BACKOFF_MS = 30_000;

type FetchState<T> = {
    data: T | null;
    loading: boolean;
    error: string | null;
};

/** 
 * Poll an API endpoint at a given interval (ms).
 * Uses exponential backoff with jitter on failure (max 30s).
 * Returns { data, loading, error, refresh }.
 */
export function useApi<T>(path: string, intervalMs = 5000): FetchState<T> & { refresh: () => void } {
    const [state, setState] = useState<FetchState<T>>({ data: null, loading: true, error: null });
    const mountedRef = useRef(true);
    const backoffRef = useRef(intervalMs);
    const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    const scheduleNext = useCallback((delay: number) => {
        if (timerRef.current) clearTimeout(timerRef.current);
        timerRef.current = setTimeout(() => {
            if (mountedRef.current) fetchLoop();
        }, delay);
    }, []);

    const fetchLoop = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}${path}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const json = await res.json();
            if (mountedRef.current) {
                setState({ data: json as T, loading: false, error: null });
                backoffRef.current = intervalMs; // reset on success
                scheduleNext(intervalMs);
            }
        } catch (err: unknown) {
            if (mountedRef.current) {
                setState(prev => ({ ...prev, loading: false, error: (err as Error).message }));
                // Exponential backoff with jitter
                const jitter = Math.random() * 500;
                backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS);
                scheduleNext(backoffRef.current + jitter);
            }
        }
    }, [path, intervalMs, scheduleNext]);

    useEffect(() => {
        mountedRef.current = true;
        backoffRef.current = intervalMs;
        fetchLoop();
        return () => {
            mountedRef.current = false;
            if (timerRef.current) clearTimeout(timerRef.current);
        };
    }, [fetchLoop, intervalMs]);

    return { ...state, refresh: fetchLoop };
}

/**
 * Fire-and-forget POST to an API endpoint.
 */
export async function apiPost<T = unknown>(path: string, body?: object): Promise<T> {
    const res = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
    }
    return res.json();
}

export { API_BASE };
