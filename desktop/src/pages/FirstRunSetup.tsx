import { useState } from 'react';
import { apiPost } from '../hooks/useApi';
import { toast } from '../stores/toastStore';
import { IconNexus, IconCheck, IconKey, IconEye, IconEyeOff } from '../components/Icons';

type Props = { onComplete: () => void };

const STEPS = ['Welcome', 'API Keys', 'Ready'] as const;

export default function FirstRunSetup({ onComplete }: Props) {
    const [step, setStep] = useState(0);
    const [binanceKey, setBinanceKey] = useState('');
    const [binanceSecret, setBinanceSecret] = useState('');
    const [openaiKey, setOpenaiKey] = useState('');
    const [showSecret, setShowSecret] = useState(false);
    const [saving, setSaving] = useState(false);

    const saveKeys = async () => {
        setSaving(true);
        try {
            const keys: Record<string, string> = {};
            if (binanceKey.trim()) keys.BINANCE_API_KEY = binanceKey.trim();
            if (binanceSecret.trim()) keys.BINANCE_API_SECRET = binanceSecret.trim();
            if (openaiKey.trim()) keys.OPENAI_API_KEY = openaiKey.trim();

            for (const [key, value] of Object.entries(keys)) {
                await apiPost('/api/settings/keys', { key, value });
            }

            toast.success('API keys saved');
            setStep(2);
        } catch (e: any) {
            toast.error(e.message);
        } finally {
            setSaving(false);
        }
    };

    const finish = async () => {
        try {
            await apiPost('/api/settings', { first_run_done: true });
        } catch { /* ignore */ }
        onComplete();
    };

    return (
        <div className="setup-root">
            <div className="setup-card animate-in">
                {/* Progress */}
                <div className="setup-steps">
                    {STEPS.map((_, i) => (
                        <div key={i} className={`setup-step ${i < step ? 'done' : i === step ? 'current' : ''}`} />
                    ))}
                </div>

                {/* Step 0: Welcome */}
                {step === 0 && (
                    <div className="flex-col items-center gap-16" style={{ textAlign: 'center' }}>
                        <IconNexus style={{ width: 48, height: 48, color: 'var(--accent)' }} />
                        <div>
                            <h1 style={{ fontSize: 20, fontWeight: 700, marginBottom: 8 }}>Welcome to Nexus Shadow-Quant</h1>
                            <p style={{ fontSize: 13, color: 'var(--text-1)', maxWidth: 360, margin: '0 auto' }}>
                                Institutional-grade Bitcoin intelligence with real-time prediction,
                                quant analysis, and paper trading.
                            </p>
                        </div>
                        <button className="btn btn-primary" onClick={() => setStep(1)}>Get Started</button>
                    </div>
                )}

                {/* Step 1: API Keys */}
                {step === 1 && (
                    <div className="flex-col gap-16">
                        <div>
                            <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 4 }}>Configure API Keys</h2>
                            <p style={{ fontSize: 12, color: 'var(--text-2)' }}>
                                You can skip this step and configure keys later in Settings.
                            </p>
                        </div>

                        <div className="flex-col gap-10">
                            <div>
                                <label style={{ fontSize: 11, color: 'var(--text-1)', display: 'block', marginBottom: 4 }}>
                                    <IconKey style={{ width: 12, height: 12, display: 'inline', marginRight: 4 }} />
                                    Binance API Key
                                </label>
                                <input className="input" value={binanceKey} onChange={e => setBinanceKey(e.target.value)} placeholder="Enter API key…" />
                            </div>
                            <div>
                                <label style={{ fontSize: 11, color: 'var(--text-1)', display: 'block', marginBottom: 4 }}>Binance Secret</label>
                                <div className="flex gap-8">
                                    <input className="input" type={showSecret ? 'text' : 'password'} value={binanceSecret} onChange={e => setBinanceSecret(e.target.value)} placeholder="Enter secret…" />
                                    <button className="btn btn-sm btn-ghost" onClick={() => setShowSecret(s => !s)}>
                                        {showSecret ? <IconEyeOff style={{ width: 14, height: 14 }} /> : <IconEye style={{ width: 14, height: 14 }} />}
                                    </button>
                                </div>
                            </div>
                            <div>
                                <label style={{ fontSize: 11, color: 'var(--text-1)', display: 'block', marginBottom: 4 }}>OpenAI API Key</label>
                                <input className="input" value={openaiKey} onChange={e => setOpenaiKey(e.target.value)} placeholder="sk-…" />
                            </div>
                        </div>

                        <div className="flex justify-between">
                            <button className="btn btn-ghost" onClick={() => setStep(2)}>Skip</button>
                            <button className="btn btn-primary" onClick={saveKeys} disabled={saving}>
                                {saving ? 'Saving…' : 'Save & Continue'}
                            </button>
                        </div>
                    </div>
                )}

                {/* Step 2: Ready */}
                {step === 2 && (
                    <div className="flex-col items-center gap-16" style={{ textAlign: 'center' }}>
                        <div style={{
                            width: 56, height: 56, borderRadius: '50%',
                            background: 'var(--positive-dim)', display: 'flex',
                            alignItems: 'center', justifyContent: 'center',
                        }}>
                            <IconCheck style={{ width: 28, height: 28, color: 'var(--positive)' }} />
                        </div>
                        <div>
                            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>You're All Set</h2>
                            <p style={{ fontSize: 13, color: 'var(--text-1)' }}>
                                Nexus is ready. The system will begin fetching market data and training models.
                            </p>
                        </div>
                        <button className="btn btn-primary" onClick={finish}>Launch Dashboard</button>
                    </div>
                )}
            </div>
        </div>
    );
}
