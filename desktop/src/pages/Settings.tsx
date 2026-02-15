import { useState, useEffect } from 'react';
import { apiPost } from '../hooks/useApi';
import { toast } from '../stores/toastStore';
import { IconKey, IconEye, IconEyeOff, IconCheck } from '../components/Icons';

type KeyConfig = {
    name: string;
    envKey: string;
    placeholder: string;
    description: string;
};

const API_KEYS: KeyConfig[] = [
    { name: 'Binance API Key', envKey: 'BINANCE_API_KEY', placeholder: 'Enter Binance API key…', description: 'Required for live market data' },
    { name: 'Binance Secret', envKey: 'BINANCE_API_SECRET', placeholder: 'Enter Binance secret…', description: 'Required for live market data' },
    { name: 'OpenAI API Key', envKey: 'OPENAI_API_KEY', placeholder: 'sk-…', description: 'For Nexus Agent (GPT-4o)' },
    { name: 'Gemini API Key', envKey: 'GEMINI_API_KEY', placeholder: 'AIza…', description: 'For Nexus Agent (Gemini 2.0 Flash)' },
    { name: 'News API Key', envKey: 'NEWS_API_KEY', placeholder: 'Enter News API key…', description: 'Optional — for news feed' },
    { name: 'Telegram Bot Token', envKey: 'TELEGRAM_BOT_TOKEN', placeholder: '123456:ABC-DEF1234ghIkl-...…', description: 'From @BotFather — for trade alerts' },
    { name: 'Telegram Chat ID', envKey: 'TELEGRAM_CHAT_ID', placeholder: '123456789', description: 'Your Telegram user/group chat ID' },
];

const LLM_PROVIDERS = [
    { value: 'ollama', label: '🦙 Ollama (Local)', description: 'Free, private, runs on your GPU' },
    { value: 'openai', label: '⚡ OpenAI (GPT-4o)', description: 'Requires API key, best quality' },
    { value: 'gemini', label: '💎 Gemini (2.0 Flash)', description: 'Requires API key, fast & free tier' },
];

function KeyCard({ config }: { config: KeyConfig }) {
    const [value, setValue] = useState('');
    const [show, setShow] = useState(false);
    const [saved, setSaved] = useState(false);
    const [configured, setConfigured] = useState(false);

    useEffect(() => {
        // Check if key is already configured
        fetch(`http://127.0.0.1:8420/api/settings/key-status/${config.envKey}`)
            .then(r => r.json())
            .then(d => setConfigured(d.is_set === true))
            .catch(() => { });
    }, [config.envKey]);

    const save = async () => {
        try {
            await apiPost('/api/settings/keys', { key: config.envKey, value });
            setSaved(true);
            setConfigured(true);
            toast.success(`${config.name} saved`);
            setTimeout(() => setSaved(false), 2000);
        } catch (e: any) {
            toast.error(e.message);
        }
    };

    return (
        <div className="card key-card animate-in">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-8">
                    <IconKey style={{ width: 16, height: 16, color: 'var(--accent)' }} />
                    <div>
                        <div style={{ fontSize: 13, fontWeight: 600 }}>{config.name}</div>
                        <div style={{ fontSize: 11, color: 'var(--text-2)' }}>{config.description}</div>
                    </div>
                </div>
                <span className={`badge ${configured ? 'badge-long' : 'badge-warning'}`}>
                    {configured ? 'Configured' : 'Missing'}
                </span>
            </div>
            <div className="input-row">
                <input
                    className="input"
                    type={show ? 'text' : 'password'}
                    placeholder={config.placeholder}
                    value={value}
                    onChange={e => setValue(e.target.value)}
                    autoComplete="off"
                />
                <button className="btn btn-sm btn-ghost" onClick={() => setShow(s => !s)} aria-label="Toggle visibility">
                    {show ? <IconEyeOff style={{ width: 14, height: 14 }} /> : <IconEye style={{ width: 14, height: 14 }} />}
                </button>
                <button
                    className={`btn btn-sm ${saved ? 'btn-success' : 'btn-primary'}`}
                    onClick={save}
                    disabled={!value.trim()}
                >
                    {saved ? <IconCheck style={{ width: 14, height: 14 }} /> : 'Save'}
                </button>
            </div>
        </div>
    );
}

function LlmProviderSelector() {
    const [provider, setProvider] = useState('ollama');
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        fetch('http://127.0.0.1:8420/api/settings')
            .then(r => r.json())
            .then(d => {
                if (d.llm_provider) setProvider(d.llm_provider);
            })
            .catch(() => { });
    }, []);

    const save = async (newProvider: string) => {
        setProvider(newProvider);
        setSaving(true);
        try {
            await apiPost('/api/settings', { llm_provider: newProvider });
            const label = LLM_PROVIDERS.find(p => p.value === newProvider)?.label || newProvider;
            toast.success(`LLM priority set to ${label}`);
        } catch (e: any) {
            toast.error(e.message);
        }
        setSaving(false);
    };

    return (
        <div className="card animate-in">
            <div className="card-title" style={{ marginBottom: 8 }}>AI Provider Priority</div>
            <p style={{ fontSize: 11, color: 'var(--text-2)', marginBottom: 12 }}>
                Choose which AI provider Dr. Nexus uses first. If it fails, the system automatically falls back to the next available provider.
            </p>
            <div className="flex-col gap-6">
                {LLM_PROVIDERS.map(p => (
                    <label
                        key={p.value}
                        className="flex items-center gap-10"
                        style={{
                            padding: '8px 12px',
                            borderRadius: 8,
                            cursor: 'pointer',
                            background: provider === p.value ? 'rgba(99,102,241,0.12)' : 'transparent',
                            border: provider === p.value ? '1px solid var(--accent)' : '1px solid transparent',
                            transition: 'all 0.2s',
                        }}
                    >
                        <input
                            type="radio"
                            name="llm_provider"
                            value={p.value}
                            checked={provider === p.value}
                            onChange={() => save(p.value)}
                            disabled={saving}
                            style={{ accentColor: 'var(--accent)' }}
                        />
                        <div>
                            <div style={{ fontSize: 13, fontWeight: 600 }}>{p.label}</div>
                            <div style={{ fontSize: 11, color: 'var(--text-2)' }}>{p.description}</div>
                        </div>
                    </label>
                ))}
            </div>
            <p style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 10, fontStyle: 'italic' }}>
                Fallback order: if the primary provider fails, the system tries the remaining providers automatically.
            </p>
        </div>
    );
}

function TelegramSection() {
    const [testing, setTesting] = useState(false);
    const [result, setResult] = useState<{ ok: boolean; message?: string; error?: string } | null>(null);

    const testConnection = async () => {
        setTesting(true);
        setResult(null);
        try {
            const r = await fetch('http://127.0.0.1:8420/api/telegram/test', { method: 'POST' });
            const data = await r.json();
            setResult(data);
            if (data.ok) toast.success(data.message || 'Test message sent!');
            else toast.error(data.error || 'Connection failed');
        } catch (e: any) {
            setResult({ ok: false, error: e.message });
            toast.error('Failed to reach backend');
        }
        setTesting(false);
    };

    return (
        <div className="card animate-in">
            <div className="card-title" style={{ marginBottom: 8 }}>📱 Telegram Notifications</div>
            <p style={{ fontSize: 11, color: 'var(--text-2)', marginBottom: 12 }}>
                Get real-time trade alerts and hourly P&L summaries on your phone.
                Save your Bot Token and Chat ID above, then test the connection.
            </p>
            <div className="flex items-center gap-10">
                <button
                    className={`btn btn-sm ${result?.ok ? 'btn-success' : 'btn-primary'}`}
                    onClick={testConnection}
                    disabled={testing}
                    style={{ minWidth: 140 }}
                >
                    {testing ? '⏳ Sending...' : result?.ok ? '✅ Connected!' : '🔔 Test Connection'}
                </button>
                {result && !result.ok && (
                    <span style={{ fontSize: 11, color: 'var(--negative)' }}>{result.error}</span>
                )}
            </div>
            <div style={{ marginTop: 12, padding: '8px 12px', borderRadius: 8, background: 'rgba(99,102,241,0.08)', fontSize: 11, color: 'var(--text-2)' }}>
                <b>Setup:</b> Message <code>@BotFather</code> on Telegram → <code>/newbot</code> → copy token.
                Then send any message to your bot and visit
                <code style={{ wordBreak: 'break-all' }}> https://api.telegram.org/bot&lt;TOKEN&gt;/getUpdates</code> to find your Chat ID.
            </div>
        </div>
    );
}

export default function Settings() {
    return (
        <div className="flex-col gap-16">
            <div>
                <h1 style={{ fontSize: 18, fontWeight: 700, marginBottom: 4 }}>Settings</h1>
                <p style={{ fontSize: 12, color: 'var(--text-2)' }}>Manage API keys, AI provider priority, and system configuration</p>
            </div>

            <LlmProviderSelector />

            <div>
                <div className="card-title" style={{ marginBottom: 12 }}>API Keys</div>
                <div className="settings-grid stagger">
                    {API_KEYS.map(config => (
                        <KeyCard key={config.envKey} config={config} />
                    ))}
                </div>
            </div>

            <TelegramSection />

            <div className="card animate-in" style={{ maxWidth: 480 }}>
                <div className="card-title" style={{ marginBottom: 12 }}>System Info</div>
                <div className="flex-col gap-8" style={{ fontSize: 12 }}>
                    <div className="flex justify-between">
                        <span style={{ color: 'var(--text-1)' }}>Platform</span>
                        <span className="mono">Nexus Shadow-Quant</span>
                    </div>
                    <div className="flex justify-between">
                        <span style={{ color: 'var(--text-1)' }}>Frontend</span>
                        <span className="mono">React + Vite</span>
                    </div>
                    <div className="flex justify-between">
                        <span style={{ color: 'var(--text-1)' }}>Backend</span>
                        <span className="mono">FastAPI @ 127.0.0.1:8420</span>
                    </div>
                </div>
            </div>
        </div>
    );
}
