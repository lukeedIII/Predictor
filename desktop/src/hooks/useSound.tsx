import { useCallback, useRef } from 'react';

/**
 * Simple notification sound hook using Web Audio API.
 * No external files needed — generates tones programmatically.
 */
export function useSound() {
    const ctxRef = useRef<AudioContext | null>(null);

    const getCtx = () => {
        if (!ctxRef.current) {
            ctxRef.current = new AudioContext();
        }
        return ctxRef.current;
    };

    const playTone = useCallback((freq: number, duration: number, type: OscillatorType = 'sine') => {
        try {
            const ctx = getCtx();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();

            osc.type = type;
            osc.frequency.setValueAtTime(freq, ctx.currentTime);
            gain.gain.setValueAtTime(0.15, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);

            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start(ctx.currentTime);
            osc.stop(ctx.currentTime + duration);
        } catch {
            // Audio not available (e.g., no user interaction yet)
        }
    }, []);

    const playTradeOpen = useCallback(() => {
        // Ascending two-tone chime (bullish feel)
        playTone(523.25, 0.15, 'sine'); // C5
        setTimeout(() => playTone(659.25, 0.2, 'sine'), 120); // E5
    }, [playTone]);

    const playTradeClose = useCallback(() => {
        // Descending two-tone (neutral close)
        playTone(659.25, 0.15, 'sine'); // E5
        setTimeout(() => playTone(440.0, 0.2, 'sine'), 120); // A4
    }, [playTone]);

    const playProfit = useCallback(() => {
        // Happy ascending triad
        playTone(523.25, 0.12, 'sine'); // C5
        setTimeout(() => playTone(659.25, 0.12, 'sine'), 100); // E5
        setTimeout(() => playTone(783.99, 0.2, 'sine'), 200); // G5
    }, [playTone]);

    const playLoss = useCallback(() => {
        // Single low tone
        playTone(220.0, 0.3, 'triangle'); // A3
    }, [playTone]);

    return { playTradeOpen, playTradeClose, playProfit, playLoss, playTone };
}
