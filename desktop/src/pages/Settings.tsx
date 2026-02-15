import { useState, useEffect, useCallback } from 'react';
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
    { value: 'embedded', label: '🧠 Embedded (Qwen 0.5B)', description: 'Built-in local model (~1GB), no setup needed' },
];

function KeyCard({ config }: { config: KeyConfig }) {
    const [value, setValue] = useState('');
    const [show, setShow] = useState(false);
    const [saved, setSaved] = useState(false);
    const [configured, setConfigured] = useState(false);

    useEffect(() => {
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


/* ═════════════════════════════════════════════════════════════════
   BETA FEATURES TOGGLE
   ═════════════════════════════════════════════════════════════════ */

type BetaFeature = {
    label: string;
    description: string;
    enabled: boolean;
};

function BetaFeaturesPanel({ onBetaChange }: { onBetaChange?: (key: string, enabled: boolean) => void }) {
    const [features, setFeatures] = useState<Record<string, BetaFeature>>({});

    useEffect(() => {
        fetch('http://127.0.0.1:8420/api/beta-features')
            .then(r => r.json())
            .then(setFeatures)
            .catch(() => { });
    }, []);

    const toggle = async (key: string) => {
        const current = features[key]?.enabled || false;
        try {
            await apiPost('/api/beta-features', { [key]: !current });
            setFeatures(prev => ({
                ...prev,
                [key]: { ...prev[key], enabled: !current },
            }));
            toast.success(`${features[key]?.label} ${!current ? 'enabled' : 'disabled'}`);
            onBetaChange?.(key, !current);
        } catch (e: any) {
            toast.error(e.message);
        }
    };

    const keys = Object.keys(features);
    if (!keys.length) return null;

    return (
        <div className="card animate-in">
            <div className="flex items-center gap-8" style={{ marginBottom: 8 }}>
                <div className="card-title" style={{ margin: 0 }}>🧪 Beta Features</div>
                <span style={{
                    fontSize: 9, fontWeight: 700, padding: '2px 6px', borderRadius: 4,
                    background: 'linear-gradient(135deg, #f59e0b, #ef4444)',
                    color: '#fff', letterSpacing: '0.05em', textTransform: 'uppercase',
                }}>BETA</span>
            </div>
            <p style={{ fontSize: 11, color: 'var(--text-2)', marginBottom: 12 }}>
                Experimental features that may change or be removed. Enable at your own risk.
            </p>
            <div className="flex-col gap-6">
                {keys.map(key => (
                    <label
                        key={key}
                        className="flex items-center gap-10"
                        style={{
                            padding: '8px 12px',
                            borderRadius: 8,
                            cursor: 'pointer',
                            background: features[key].enabled ? 'rgba(245,158,11,0.10)' : 'transparent',
                            border: features[key].enabled ? '1px solid rgba(245,158,11,0.4)' : '1px solid rgba(255,255,255,0.06)',
                            transition: 'all 0.2s',
                        }}
                    >
                        <div style={{
                            width: 36, height: 20, borderRadius: 10,
                            background: features[key].enabled ? 'var(--accent)' : 'rgba(255,255,255,0.1)',
                            position: 'relative', transition: 'background 0.2s', cursor: 'pointer', flexShrink: 0,
                        }}
                            onClick={() => toggle(key)}
                        >
                            <div style={{
                                width: 16, height: 16, borderRadius: '50%', background: '#fff',
                                position: 'absolute', top: 2,
                                left: features[key].enabled ? 18 : 2,
                                transition: 'left 0.2s', boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
                            }} />
                        </div>
                        <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ fontSize: 13, fontWeight: 600 }}>{features[key].label}</div>
                            <div style={{ fontSize: 11, color: 'var(--text-2)' }}>{features[key].description}</div>
                        </div>
                    </label>
                ))}
            </div>
        </div>
    );
}


/* ═════════════════════════════════════════════════════════════════
   MODEL ARCHITECTURE SELECTOR
   ═════════════════════════════════════════════════════════════════ */

type ModelInfo = {
    key: string;
    label: string;
    params: string;
    vram_gb: number;
    description: string;
    has_weights: boolean;
    file_size_mb: number;
    vram_ok: boolean;
    is_active: boolean;
};

type ModelsResponse = {
    models: ModelInfo[];
    active_arch: string;
    running_arch: string | null;
    beta_enabled: boolean;
    gpu: { name: string; vram_total_gb: number; vram_free_gb: number };
};

type HFModel = {
    id: string;
    author: string;
    downloads: number;
    likes: number;
    description: string;
    tags: string[];
};

function VramBar({ used, total, required }: { used: number; total: number; required: number }) {
    if (total <= 0) return null;
    const usedPct = Math.min((used / total) * 100, 100);
    const reqPct = Math.min((required / total) * 100, 100);
    const ok = (total - used) >= required;

    return (
        <div style={{ width: '100%', marginTop: 4 }}>
            <div style={{
                height: 6, borderRadius: 3, background: 'rgba(255,255,255,0.06)',
                position: 'relative', overflow: 'hidden',
            }}>
                <div style={{
                    position: 'absolute', left: 0, top: 0, height: '100%',
                    width: `${usedPct}%`, background: 'rgba(255,255,255,0.15)',
                    borderRadius: 3,
                }} />
                <div style={{
                    position: 'absolute', left: `${usedPct}%`, top: 0, height: '100%',
                    width: `${reqPct}%`,
                    background: ok ? 'rgba(34,197,94,0.5)' : 'rgba(239,68,68,0.5)',
                    borderRadius: 3,
                }} />
            </div>
            <div style={{ fontSize: 9, color: 'var(--text-3)', marginTop: 2 }}>
                Needs {required} GB — {ok ? '✅ OK' : '⚠️ Insufficient VRAM'}
            </div>
        </div>
    );
}

function ModelSelector() {
    const [data, setData] = useState<ModelsResponse | null>(null);
    const [selecting, setSelecting] = useState('');
    const [hfModels, setHfModels] = useState<HFModel[]>([]);
    const [hfSearching, setHfSearching] = useState(false);
    const [hfQuery, setHfQuery] = useState('bitcoin price prediction transformer');
    const [hfSearched, setHfSearched] = useState(false);
    const [downloading, setDownloading] = useState('');

    const loadModels = useCallback(() => {
        fetch('http://127.0.0.1:8420/api/models')
            .then(r => r.json())
            .then(setData)
            .catch(() => { });
    }, []);

    useEffect(() => { loadModels(); }, [loadModels]);

    const selectModel = async (arch: string) => {
        setSelecting(arch);
        try {
            const result = await apiPost<{ status: string; message: string }>('/api/models/select', { arch });
            toast.success(result.message);
            loadModels();
        } catch (e: any) {
            toast.error(e.message);
        }
        setSelecting('');
    };

    const searchHuggingFace = async () => {
        setHfSearching(true);
        setHfSearched(false);
        try {
            const result = await apiPost<{ models: HFModel[] }>('/api/models/hf-search', { query: hfQuery });
            setHfModels(result.models || []);
            setHfSearched(true);
        } catch (e: any) {
            toast.error('HuggingFace search failed: ' + e.message);
        }
        setHfSearching(false);
    };

    const downloadHfModel = async (modelId: string) => {
        setDownloading(modelId);
        try {
            const result = await apiPost<{ status: string; message: string }>('/api/models/hf-download', { model_id: modelId });
            toast.success(result.message || `Downloaded ${modelId}`);
            loadModels();
        } catch (e: any) {
            toast.error(e.message);
        }
        setDownloading('');
    };

    if (!data) return null;

    const needsRestart = data.active_arch !== data.running_arch;

    return (
        <div className="card animate-in">
            <div className="flex items-center gap-8" style={{ marginBottom: 4 }}>
                <div className="card-title" style={{ margin: 0 }}>🧠 Model Architecture</div>
                <span style={{
                    fontSize: 9, fontWeight: 700, padding: '2px 6px', borderRadius: 4,
                    background: 'linear-gradient(135deg, #f59e0b, #ef4444)',
                    color: '#fff', letterSpacing: '0.05em', textTransform: 'uppercase',
                }}>BETA</span>
            </div>

            {/* GPU Info Bar */}
            <div style={{
                padding: '6px 10px', borderRadius: 6, marginBottom: 12,
                background: 'rgba(99,102,241,0.08)', fontSize: 11, color: 'var(--text-2)',
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            }}>
                <span>🖥️ {data.gpu.name}</span>
                <span className="mono">{data.gpu.vram_free_gb} / {data.gpu.vram_total_gb} GB VRAM free</span>
            </div>

            {/* Restart Warning */}
            {needsRestart && (
                <div style={{
                    padding: '8px 12px', borderRadius: 8, marginBottom: 12,
                    background: 'rgba(245,158,11,0.12)', border: '1px solid rgba(245,158,11,0.3)',
                    fontSize: 11, color: '#f59e0b',
                }}>
                    ⚠️ Model changed to <b>{data.active_arch}</b> — restart the server to apply.
                    Currently running: <b>{data.running_arch}</b>
                </div>
            )}

            {/* Local Models Grid */}
            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: 'var(--text-1)' }}>Local Models</div>
            <div className="flex-col gap-6" style={{ marginBottom: 16 }}>
                {data.models.map(m => (
                    <div
                        key={m.key}
                        onClick={() => !selecting && selectModel(m.key)}
                        style={{
                            padding: '10px 14px',
                            borderRadius: 10,
                            cursor: selecting ? 'wait' : 'pointer',
                            background: m.is_active ? 'rgba(99,102,241,0.12)' : 'rgba(255,255,255,0.02)',
                            border: m.is_active ? '1px solid var(--accent)' : '1px solid rgba(255,255,255,0.06)',
                            transition: 'all 0.2s',
                            opacity: selecting === m.key ? 0.6 : 1,
                        }}
                    >
                        <div className="flex items-center justify-between" style={{ marginBottom: 4 }}>
                            <div className="flex items-center gap-8">
                                <div style={{
                                    width: 16, height: 16, borderRadius: '50%',
                                    border: m.is_active ? '5px solid var(--accent)' : '2px solid rgba(255,255,255,0.2)',
                                    transition: 'all 0.2s',
                                }} />
                                <span style={{ fontSize: 13, fontWeight: 600 }}>{m.label}</span>
                            </div>
                            <div className="flex items-center gap-6">
                                {m.has_weights ? (
                                    <span style={{
                                        fontSize: 9, padding: '2px 6px', borderRadius: 4,
                                        background: 'rgba(34,197,94,0.15)', color: '#22c55e',
                                        fontWeight: 600,
                                    }}>READY</span>
                                ) : (
                                    <span style={{
                                        fontSize: 9, padding: '2px 6px', borderRadius: 4,
                                        background: 'rgba(255,255,255,0.06)', color: 'var(--text-3)',
                                        fontWeight: 600,
                                    }}>NO WEIGHTS</span>
                                )}
                                <span className="mono" style={{
                                    fontSize: 10, color: 'var(--text-3)',
                                }}>{m.params} params</span>
                            </div>
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--text-2)', marginLeft: 24, marginBottom: 4 }}>
                            {m.description}
                        </div>
                        {m.file_size_mb > 0 && (
                            <div style={{ fontSize: 10, color: 'var(--text-3)', marginLeft: 24 }}>
                                💾 {m.file_size_mb} MB on disk
                            </div>
                        )}
                        <div style={{ marginLeft: 24, maxWidth: 200 }}>
                            <VramBar
                                used={data.gpu.vram_total_gb - data.gpu.vram_free_gb}
                                total={data.gpu.vram_total_gb}
                                required={m.vram_gb}
                            />
                        </div>
                    </div>
                ))}
            </div>

            {/* HuggingFace Marketplace */}
            <div style={{
                borderTop: '1px solid rgba(255,255,255,0.06)',
                paddingTop: 16, marginTop: 8,
            }}>
                <div className="flex items-center gap-8" style={{ marginBottom: 8 }}>
                    <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-1)' }}>
                        🤗 HuggingFace Model Hub
                    </span>
                    <span style={{
                        fontSize: 9, padding: '2px 6px', borderRadius: 4,
                        background: 'rgba(255,213,79,0.12)', color: '#ffd54f',
                        fontWeight: 600,
                    }}>EXPERIMENTAL</span>
                </div>
                <p style={{ fontSize: 11, color: 'var(--text-2)', marginBottom: 10 }}>
                    Search for pre-trained financial/crypto models on HuggingFace.
                    Community models may significantly outperform local training.
                </p>
                <div className="flex items-center gap-8" style={{ marginBottom: 12 }}>
                    <input
                        className="input"
                        value={hfQuery}
                        onChange={e => setHfQuery(e.target.value)}
                        placeholder="Search HuggingFace models…"
                        onKeyDown={e => e.key === 'Enter' && searchHuggingFace()}
                        style={{ flex: 1, fontSize: 12 }}
                    />
                    <button
                        className="btn btn-sm btn-primary"
                        onClick={searchHuggingFace}
                        disabled={hfSearching || !hfQuery.trim()}
                        style={{ minWidth: 80 }}
                    >
                        {hfSearching ? '⏳ ...' : '🔍 Search'}
                    </button>
                </div>

                {/* HF Results */}
                {hfSearched && hfModels.length === 0 && (
                    <div style={{
                        padding: '12px', borderRadius: 8, textAlign: 'center',
                        background: 'rgba(255,255,255,0.02)', fontSize: 11, color: 'var(--text-3)',
                    }}>
                        No compatible models found. Try different search terms like "crypto", "finance", "time-series".
                    </div>
                )}

                {hfModels.length > 0 && (
                    <div className="flex-col gap-6">
                        {hfModels.map(m => (
                            <div key={m.id} style={{
                                padding: '10px 14px', borderRadius: 10,
                                background: 'rgba(255,255,255,0.02)',
                                border: '1px solid rgba(255,255,255,0.06)',
                                transition: 'all 0.2s',
                            }}>
                                <div className="flex items-center justify-between" style={{ marginBottom: 4 }}>
                                    <div>
                                        <div style={{ fontSize: 13, fontWeight: 600 }}>{m.id}</div>
                                        <div style={{ fontSize: 10, color: 'var(--text-3)' }}>by {m.author}</div>
                                    </div>
                                    <div className="flex items-center gap-8">
                                        <span style={{ fontSize: 10, color: 'var(--text-3)' }}>
                                            ⬇️ {m.downloads >= 1000 ? `${(m.downloads / 1000).toFixed(1)}k` : m.downloads}
                                        </span>
                                        <span style={{ fontSize: 10, color: 'var(--text-3)' }}>
                                            ❤️ {m.likes}
                                        </span>
                                        <button
                                            className="btn btn-sm btn-primary"
                                            onClick={(e) => { e.stopPropagation(); downloadHfModel(m.id); }}
                                            disabled={downloading === m.id}
                                            style={{ fontSize: 10, minWidth: 70 }}
                                        >
                                            {downloading === m.id ? '⏳...' : '📥 Download'}
                                        </button>
                                    </div>
                                </div>
                                {m.description && (
                                    <div style={{ fontSize: 11, color: 'var(--text-2)', marginTop: 2 }}>
                                        {m.description.slice(0, 150)}{m.description.length > 150 ? '…' : ''}
                                    </div>
                                )}
                                {m.tags.length > 0 && (
                                    <div className="flex items-center gap-4" style={{ marginTop: 4, flexWrap: 'wrap' }}>
                                        {m.tags.slice(0, 5).map(t => (
                                            <span key={t} style={{
                                                fontSize: 9, padding: '1px 5px', borderRadius: 3,
                                                background: 'rgba(99,102,241,0.10)', color: 'var(--accent)',
                                                fontWeight: 500,
                                            }}>{t}</span>
                                        ))}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}


/* ═════════════════════════════════════════════════════════════════
   MAIN SETTINGS PAGE
   ═════════════════════════════════════════════════════════════════ */

export default function Settings() {
    const [betaEnabled, setBetaEnabled] = useState(false);

    useEffect(() => {
        fetch('http://127.0.0.1:8420/api/beta-features')
            .then(r => r.json())
            .then(d => {
                setBetaEnabled(d?.model_selector?.enabled || false);
            })
            .catch(() => { });
    }, []);

    const handleBetaChange = (key: string, enabled: boolean) => {
        if (key === 'model_selector') setBetaEnabled(enabled);
    };

    return (
        <div className="flex-col gap-16">
            <div>
                <h1 style={{ fontSize: 18, fontWeight: 700, marginBottom: 4 }}>Settings</h1>
                <p style={{ fontSize: 12, color: 'var(--text-2)' }}>Manage API keys, AI provider priority, and system configuration</p>
            </div>

            {/* Beta Features Toggle — always visible */}
            <BetaFeaturesPanel onBetaChange={handleBetaChange} />

            {/* Model Selector — only visible when beta is enabled */}
            {betaEnabled && <ModelSelector />}

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
