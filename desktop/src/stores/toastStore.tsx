/* eslint-disable react-refresh/only-export-components */
/**
 * Toast — Lightweight notification system
 * =========================================
 * Usage:
 *   import { toast } from '../stores/toastStore';
 *   toast.success('Trade opened');
 *   toast.error('Connection lost');
 *
 * Render <Toaster /> once at app root.
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import { IconCheck, IconWarning, IconInfo, IconX } from '../components/Icons';

// ─── Store ──────────────────────────────────────
export type ToastType = 'success' | 'error' | 'warning' | 'info';

type ToastItem = {
    id: number;
    type: ToastType;
    message: string;
    duration: number;
};

let nextId = 0;
const listeners = new Set<() => void>();
let items: ToastItem[] = [];

function notify() { listeners.forEach(l => l()); }

function add(type: ToastType, message: string, duration = 4000) {
    const id = nextId++;
    items = [...items, { id, type, message, duration }];
    notify();
}

function remove(id: number) {
    items = items.filter(t => t.id !== id);
    notify();
}

export const toast = {
    success: (msg: string, ms?: number) => add('success', msg, ms),
    error: (msg: string, ms?: number) => add('error', msg, ms ?? 6000),
    warning: (msg: string, ms?: number) => add('warning', msg, ms),
    info: (msg: string, ms?: number) => add('info', msg, ms),
};

function useToasts() {
    const [, setTick] = useState(0);
    useEffect(() => {
        const listener = () => setTick(t => t + 1);
        listeners.add(listener);
        return () => { listeners.delete(listener); };
    }, []);
    return items;
}

// ─── Components ─────────────────────────────────
const ICON_MAP = {
    success: <IconCheck size={16} />,
    error: <IconWarning size={16} />,
    warning: <IconWarning size={16} />,
    info: <IconInfo size={16} />,
};

function ToastCard({ item }: { item: ToastItem }) {
    const [exiting, setExiting] = useState(false);
    const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined);

    const dismiss = useCallback(() => {
        setExiting(true);
        setTimeout(() => remove(item.id), 300);
    }, [item.id]);

    useEffect(() => {
        timerRef.current = setTimeout(dismiss, item.duration);
        return () => clearTimeout(timerRef.current);
    }, [dismiss, item.duration]);

    return (
        <div
            className={`toast toast-${item.type} ${exiting ? 'toast-exit' : 'toast-enter'}`}
            role="alert"
            aria-live="assertive"
        >
            <span className="toast-icon">{ICON_MAP[item.type]}</span>
            <span className="toast-msg">{item.message}</span>
            <button className="toast-close" onClick={dismiss} aria-label="Dismiss notification">
                <IconX size={14} />
            </button>
        </div>
    );
}

export function Toaster() {
    const toasts = useToasts();
    if (toasts.length === 0) return null;

    return (
        <div className="toast-container" aria-label="Notifications">
            {toasts.map(t => <ToastCard key={t.id} item={t} />)}
        </div>
    );
}
