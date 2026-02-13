import { useState, useEffect, useCallback } from 'react';
import { IconSettings, IconWarning } from '../components/Icons';

import { API_BASE } from '../hooks/useApi';
const API = API_BASE;

type KeyStatus = Record<string, boolean>;
type KeyMasked = Record<string, string>;

type SettingsData = {
    keys: KeyMasked;
    has_keys: KeyStatus;
    data_root: string;
    version: string;
    is_installed: boolean;
};

type KeyConfig = {
    id: string;
    label: string;
    envKey: string;
    provider: string;
    icon: string;
    placeholder: string;
    description: string;
};

const KEY_CONFIGS: KeyConfig[] = [
    {
        id: 'gemini',
        label: 'Google Gemini',
        envKey: 'GEMINI_API_KEY',
        provider: 'gemini',
        icon: '🔮',
        placeholder: 'AIza...',
        description: 'Enhanced AI market commentary and analysis',
    },
    {
        id: 'openai',
        label: 'OpenAI (ChatGPT)',
        envKey: 'OPENAI_API_KEY',
        provider: 'openai',
        icon: '🤖',
        placeholder: 'sk-...',
        description: 'Alternative AI provider for analysis',
    },
    {
        id: 'binance',
        label: 'Binance API Key',
        envKey: 'BINANCE_API_KEY',
        provider: 'binance',
        icon: '📊',
        placeholder: 'Your Binance API key...',
        description: 'Live market data access (public endpoints work without)',
    },
    {
        id: 'binance_secret',
        label: 'Binance Secret Key',
        envKey: 'BINANCE_SECRET_KEY',
        provider: '',
        icon: '🔒',
        placeholder: 'Your Binance secret key...',
        description: 'Required only for future real trading features',
    },
];


export default function Settings() {
    const [settings, setSettings] = useState<SettingsData | null>(null);
    const [editValues, setEditValues] = useState<Record<string, string>>({});
    const [validating, setValidating] = useState<Record<string, boolean>>({});
    const [validResults, setValidResults] = useState<Record<string, { valid: boolean; message: string }>>({});
    const [saving, setSaving] = useState(false);
    const [saveResult, setSaveResult] = useState<string | null>(null);

    // Load settings
    useEffect(() => {
        fetch(`${API}/api/settings`)
            .then(r => r.json())
            .then(d => setSettings(d))
            .catch(() => { });
    }, []);

    // Validate a key
    const handleValidate = useCallback(async (cfg: KeyConfig) => {
        const key = editValues[cfg.envKey];
        if (!key || !cfg.provider) return;

        setValidating(v => ({ ...v, [cfg.envKey]: true }));
        try {
            const res = await fetch(`${API}/api/settings/validate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ provider: cfg.provider, key }),
            });
            const data = await res.json();
            setValidResults(v => ({ ...v, [cfg.envKey]: data }));
        } catch {
            setValidResults(v => ({ ...v, [cfg.envKey]: { valid: false, message: 'Connection failed' } }));
        }
        setValidating(v => ({ ...v, [cfg.envKey]: false }));
    }, [editValues]);

    // Save all keys
    const handleSave = useCallback(async () => {
        const body: Record<string, string> = {};
        for (const cfg of KEY_CONFIGS) {
            if (editValues[cfg.envKey] !== undefined && editValues[cfg.envKey] !== '') {
                const field = cfg.envKey.toLowerCase();
                body[field] = editValues[cfg.envKey];
            }
        }
        if (Object.keys(body).length === 0) return;

        setSaving(true);
        setSaveResult(null);
        try {
            const res = await fetch(`${API}/api/settings`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const data = await res.json();
            if (data.saved) {
                setSaveResult('Settings saved successfully!');
                setEditValues({});
                // Reload settings
                const r2 = await fetch(`${API}/api/settings`);
                setSettings(await r2.json());
            } else {
                setSaveResult('Failed to save settings');
            }
        } catch {
            setSaveResult('Connection error');
        }
        setSaving(false);
    }, [editValues]);

    const hasEdits = Object.values(editValues).some(v => v !== '');

    return (
        <div className="settings-page">
            {/* Header */}
            <div className="mb-32">
                <h1 className="settings-title"><IconSettings size={20} style={{ marginRight: 8, verticalAlign: -3 }} /> Settings</h1>
                <p className="settings-subtitle">
                    Configure API keys and preferences. Keys are stored locally on your device.
                </p>
            </div>

            {/* API Key Cards */}
            <div className="flex-col gap-16">
                {KEY_CONFIGS.map(cfg => {
                    const hasKey = settings?.has_keys?.[cfg.envKey];
                    const masked = settings?.keys?.[cfg.envKey] || '';
                    const editVal = editValues[cfg.envKey] ?? '';
                    const vResult = validResults[cfg.envKey];
                    const isValidating = validating[cfg.envKey];

                    return (
                        <div key={cfg.id} className="settings-card">
                            {/* Label Row */}
                            <div className="settings-card-head">
                                <div className="settings-card-label">
                                    <span className="text-18">{cfg.icon}</span>
                                    <span className="font-600 text-1 text-14">{cfg.label}</span>
                                    {hasKey && (
                                        <span className="settings-badge-ok">Configured</span>
                                    )}
                                </div>
                            </div>

                            {/* Description */}
                            <p className="settings-desc">{cfg.description}</p>

                            {/* Input Row */}
                            <div className="flex gap-8">
                                <input
                                    type="password"
                                    className="settings-input"
                                    placeholder={hasKey ? `Current: ${masked}` : cfg.placeholder}
                                    value={editVal}
                                    onChange={e => setEditValues(v => ({ ...v, [cfg.envKey]: e.target.value }))}
                                />
                                {cfg.provider && editVal && (
                                    <button
                                        className="settings-test-btn"
                                        onClick={() => handleValidate(cfg)}
                                        disabled={isValidating}
                                    >
                                        {isValidating ? '...' : 'Test'}
                                    </button>
                                )}
                            </div>

                            {/* Validation Result */}
                            {vResult && (
                                <div
                                    className="validation-result"
                                    style={{
                                        background: vResult.valid ? 'rgba(0,217,156,0.1)' : 'rgba(244,67,54,0.1)',
                                        color: vResult.valid ? 'var(--positive)' : 'var(--negative)',
                                    }}
                                >
                                    {vResult.valid ? '✓' : '✗'} {vResult.message}
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>

            {/* Save Button */}
            {hasEdits && (
                <div className="mt-24 flex items-center gap-16">
                    <button
                        className="settings-save-btn"
                        onClick={handleSave}
                        disabled={saving}
                    >
                        {saving ? 'Saving...' : 'Save All Keys'}
                    </button>
                    {saveResult && (
                        <span className="text-13 text-positive">{saveResult}</span>
                    )}
                </div>
            )}

            {/* System Info */}
            <div className="mt-40 settings-info-panel">
                <h3 className="text-14 font-600 text-2 mb-12">System Information</h3>
                <div className="settings-info-grid">
                    <span className="text-4">Version</span>
                    <span className="text-2" style={{ fontFamily: 'monospace' }}>
                        {settings?.version || '...'}
                    </span>
                    <span className="text-4">Data Location</span>
                    <span className="text-2" style={{ fontFamily: 'monospace', fontSize: 11, wordBreak: 'break-all' as const }}>
                        {settings?.data_root || '...'}
                    </span>
                    <span className="text-4">Runtime</span>
                    <span className="text-2" style={{ fontFamily: 'monospace' }}>
                        {settings?.is_installed ? 'Installed App' : 'Development Mode'}
                    </span>
                </div>
            </div>

            {/* Disclaimer */}
            <div className="mt-24 settings-disclaimer">
                <p>
                    <IconWarning size={14} style={{ marginRight: 6, verticalAlign: -2, color: 'var(--warning)' }} /> <strong>Research Tool Disclaimer</strong> — Nexus Shadow-Quant is an educational
                    and research tool. It is NOT financial advice. All predictions are statistical models
                    and do NOT guarantee profits. You are fully responsible for any trading decisions.
                </p>
            </div>
        </div>
    );
}
