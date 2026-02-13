/**
 * useKeyboardShortcuts — Global keyboard shortcuts for Nexus Shadow-Quant
 * 
 * Navigation:
 *   Alt+1 → Dashboard
 *   Alt+2 → Paper Trading  
 *   Alt+3 → Nexus Agent
 *   Alt+4 → Settings
 * 
 * Actions:
 *   Alt+T → Toggle trading bot
 *   Alt+R → Trigger model retrain
 *   Alt+? → Show shortcuts help overlay
 */

import { useEffect, useCallback, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { apiPost } from './useApi';

export const SHORTCUTS = [
    { keys: 'Alt + 1', label: 'Go to Dashboard' },
    { keys: 'Alt + 2', label: 'Go to Paper Trading' },
    { keys: 'Alt + 3', label: 'Go to Nexus Agent' },
    { keys: 'Alt + 4', label: 'Go to Settings' },
    { keys: 'Alt + T', label: 'Toggle auto-trade bot' },
    { keys: 'Alt + R', label: 'Trigger model retrain' },
    { keys: 'Alt + /', label: 'Show/hide shortcuts help' },
] as const;

export function useKeyboardShortcuts() {
    const navigate = useNavigate();
    const [showHelp, setShowHelp] = useState(false);

    const handler = useCallback((e: KeyboardEvent) => {
        // Skip if user is typing in an input/textarea
        const tag = (e.target as HTMLElement)?.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

        if (!e.altKey) return;

        switch (e.key) {
            case '1':
                e.preventDefault();
                navigate('/');
                break;
            case '2':
                e.preventDefault();
                navigate('/trading');
                break;
            case '3':
                e.preventDefault();
                navigate('/agent');
                break;
            case '4':
                e.preventDefault();
                navigate('/settings');
                break;
            case 't':
            case 'T':
                e.preventDefault();
                apiPost('/api/bot/start').catch(() => { });
                break;
            case 'r':
            case 'R':
                e.preventDefault();
                apiPost('/api/train').catch(() => { });
                break;
            case '/':
                e.preventDefault();
                setShowHelp(prev => !prev);
                break;
        }
    }, [navigate]);

    useEffect(() => {
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [handler]);

    return { showHelp, setShowHelp };
}
