import { useState, useEffect, useRef } from 'react';

type FirstRunStatus = {
    needs_setup: boolean;
    running: boolean;
    complete: boolean;
    progress: {
        stage?: string;
        progress?: number;
        message?: string;
        step?: string;
        candles?: number;
    };
};

type SystemCheck = {
    gpu_name: string;
    vram_gb: number;
    gpu_ok: boolean;
    gpu_compute: number;
    disk_free_gb: number;
    disk_ok: boolean;
    ram_gb: number;
    ram_ok: boolean;
    cuda_version: string;
    errors: string[];
    warnings: string[];
};

import { API_BASE } from '../hooks/useApi';
const API = API_BASE;

export default function FirstRunSetup({ onComplete }: { onComplete: () => void }) {
    const [step, setStep] = useState<'welcome' | 'system' | 'setup' | 'done'>('welcome');
    const [systemCheck, setSystemCheck] = useState<SystemCheck | null>(null);
    const [setupStatus, setSetupStatus] = useState<FirstRunStatus | null>(null);
    const [logs, setLogs] = useState<string[]>([]);
    const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const logsEndRef = useRef<HTMLDivElement>(null);

    // Auto-scroll logs
    useEffect(() => {
        logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [logs]);

    // Poll setup status when running
    useEffect(() => {
        if (step !== 'setup') return;

        pollRef.current = setInterval(async () => {
            try {
                const res = await fetch(`${API}/api/first-run-status`);
                const data: FirstRunStatus = await res.json();
                setSetupStatus(data);

                if (data.progress?.message) {
                    setLogs(prev => {
                        const last = prev[prev.length - 1];
                        if (last !== data.progress.message) {
                            return [...prev, data.progress.message!];
                        }
                        return prev;
                    });
                }

                if (data.complete) {
                    setStep('done');
                    if (pollRef.current) clearInterval(pollRef.current);
                }
            } catch { /* retry */ }
        }, 1000);

        return () => { if (pollRef.current) clearInterval(pollRef.current); };
    }, [step]);

    const runSystemCheck = async () => {
        setStep('system');
        try {
            const res = await fetch(`${API}/api/system-check`);
            const data = await res.json();
            setSystemCheck(data);
        } catch {
            setSystemCheck({
                gpu_name: 'Unknown', vram_gb: 0, gpu_ok: false, gpu_compute: 0,
                disk_free_gb: 0, disk_ok: false, ram_gb: 0, ram_ok: false,
                cuda_version: 'N/A', errors: ['Failed to connect to backend'], warnings: []
            });
        }
    };

    const startSetup = async () => {
        setStep('setup');
        setLogs(['Starting first-run setup...']);
        try {
            await fetch(`${API}/api/first-run`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ days: 1095 }) });
        } catch {
            setLogs(prev => [...prev, 'ERROR: Failed to start setup process']);
        }
    };

    const progressPct = setupStatus?.progress?.progress ?? 0;
    const currentStage = setupStatus?.progress?.stage || 'initializing';

    return (
        <div style={{
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            height: '100%', padding: '40px', color: 'var(--text-1)', fontFamily: 'var(--font-mono)',
        }}>
            {/* ─── Welcome ─── */}
            {step === 'welcome' && (
                <div style={{ textAlign: 'center', maxWidth: 600 }}>
                    <div style={{ fontSize: 48, marginBottom: 16 }}>⚡</div>
                    <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 8, color: 'var(--accent)' }}>
                        Welcome to Nexus Shadow-Quant
                    </h1>
                    <p style={{ color: 'var(--text-3)', marginBottom: 32, lineHeight: 1.6 }}>
                        First-time setup is required. This will download 3 years of BTC market data
                        and train the prediction models on your GPU. This process takes approximately
                        <strong style={{ color: 'var(--warning)' }}> 15-30 minutes</strong>.
                    </p>
                    <div style={{
                        background: 'var(--surface-2)', borderRadius: 12, padding: '20px 24px',
                        marginBottom: 32, textAlign: 'left', border: '1px solid var(--border)'
                    }}>
                        <h3 style={{ fontSize: 13, marginBottom: 12, color: 'var(--text-2)' }}>What will happen:</h3>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                            {['System compatibility check (GPU, RAM, Disk)', 'Download 3 years of BTC/USDT historical data', 'Download FinBERT sentiment model', 'Train XGBoost prediction model', 'Train LSTM neural network'].map((item, i) => (
                                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text-3)' }}>
                                    <span style={{ color: 'var(--accent)', fontWeight: 700 }}>{i + 1}.</span> {item}
                                </div>
                            ))}
                        </div>
                    </div>
                    <button
                        onClick={runSystemCheck}
                        style={{
                            background: 'var(--accent)', color: '#000', border: 'none', borderRadius: 8,
                            padding: '12px 32px', fontSize: 14, fontWeight: 700, cursor: 'pointer',
                            fontFamily: 'var(--font-mono)', transition: 'opacity 0.2s',
                        }}
                        onMouseOver={e => (e.currentTarget.style.opacity = '0.85')}
                        onMouseOut={e => (e.currentTarget.style.opacity = '1')}
                    >
                        Begin Setup →
                    </button>
                </div>
            )}

            {/* ─── System Check ─── */}
            {step === 'system' && (
                <div style={{ textAlign: 'center', maxWidth: 550 }}>
                    <h2 style={{ fontSize: 18, marginBottom: 24, color: 'var(--text-1)' }}>System Check</h2>
                    {!systemCheck ? (
                        <p style={{ color: 'var(--text-3)' }}>Checking system requirements...</p>
                    ) : (
                        <>
                            <div style={{
                                background: 'var(--surface-2)', borderRadius: 12, padding: 20,
                                textAlign: 'left', border: '1px solid var(--border)', marginBottom: 24
                            }}>
                                {[
                                    { label: 'GPU', value: systemCheck.gpu_name, ok: systemCheck.gpu_ok },
                                    { label: 'VRAM', value: `${systemCheck.vram_gb} GB`, ok: systemCheck.vram_gb >= 6 },
                                    { label: 'CUDA', value: systemCheck.cuda_version, ok: systemCheck.cuda_version !== 'N/A' },
                                    { label: 'Disk', value: `${systemCheck.disk_free_gb} GB free`, ok: systemCheck.disk_ok },
                                    { label: 'RAM', value: `${systemCheck.ram_gb} GB`, ok: systemCheck.ram_ok },
                                ].map((item, i) => (
                                    <div key={i} style={{
                                        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                        padding: '8px 0', borderBottom: i < 4 ? '1px solid var(--border)' : 'none',
                                        fontSize: 13,
                                    }}>
                                        <span style={{ color: 'var(--text-3)' }}>{item.label}</span>
                                        <span style={{ color: item.ok ? 'var(--positive)' : 'var(--negative)', fontWeight: 600 }}>
                                            {item.ok ? '✓' : '✗'} {item.value}
                                        </span>
                                    </div>
                                ))}
                            </div>

                            {systemCheck.errors.length > 0 && (
                                <div style={{
                                    background: 'rgba(255,50,50,0.1)', border: '1px solid var(--negative)',
                                    borderRadius: 8, padding: 12, marginBottom: 20, textAlign: 'left',
                                }}>
                                    {systemCheck.errors.map((err, i) => (
                                        <p key={i} style={{ color: 'var(--negative)', fontSize: 12, margin: '4px 0' }}>⚠️ {err}</p>
                                    ))}
                                </div>
                            )}

                            <button
                                onClick={startSetup}
                                disabled={systemCheck.errors.length > 0}
                                style={{
                                    background: systemCheck.errors.length > 0 ? 'var(--surface-3)' : 'var(--accent)',
                                    color: systemCheck.errors.length > 0 ? 'var(--text-4)' : '#000',
                                    border: 'none', borderRadius: 8, padding: '12px 32px', fontSize: 14,
                                    fontWeight: 700, cursor: systemCheck.errors.length > 0 ? 'not-allowed' : 'pointer',
                                    fontFamily: 'var(--font-mono)',
                                }}
                            >
                                {systemCheck.errors.length > 0 ? 'System Requirements Not Met' : 'Start Setup →'}
                            </button>
                        </>
                    )}
                </div>
            )}

            {/* ─── Setup Running ─── */}
            {step === 'setup' && (
                <div style={{ width: '100%', maxWidth: 600 }}>
                    <h2 style={{ fontSize: 18, marginBottom: 8, textAlign: 'center', color: 'var(--text-1)' }}>
                        Setting Up Nexus
                    </h2>
                    <p style={{ textAlign: 'center', color: 'var(--text-3)', fontSize: 12, marginBottom: 24 }}>
                        Stage: <strong style={{ color: 'var(--accent)' }}>{currentStage}</strong>
                    </p>

                    {/* Progress bar */}
                    <div style={{
                        background: 'var(--surface-2)', borderRadius: 8, height: 8,
                        overflow: 'hidden', marginBottom: 24,
                    }}>
                        <div style={{
                            background: 'var(--accent)', height: '100%', borderRadius: 8,
                            width: `${Math.min(100, progressPct)}%`,
                            transition: 'width 0.5s ease',
                        }} />
                    </div>

                    {/* Log panel */}
                    <div style={{
                        background: 'var(--surface-1)', border: '1px solid var(--border)',
                        borderRadius: 8, height: 250, overflowY: 'auto', padding: 12,
                        fontFamily: 'var(--font-mono)', fontSize: 11, lineHeight: 1.6,
                    }}>
                        {logs.map((log, i) => (
                            <div key={i} style={{ color: log.includes('ERROR') ? 'var(--negative)' : 'var(--text-3)' }}>
                                <span style={{ color: 'var(--text-4)', marginRight: 8 }}>
                                    [{String(i + 1).padStart(3, '0')}]
                                </span>
                                {log}
                            </div>
                        ))}
                        <div ref={logsEndRef} />
                    </div>
                </div>
            )}

            {/* ─── Done ─── */}
            {step === 'done' && (
                <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: 48, marginBottom: 16 }}>🎉</div>
                    <h2 style={{ fontSize: 20, marginBottom: 8, color: 'var(--positive)' }}>Setup Complete!</h2>
                    <p style={{ color: 'var(--text-3)', marginBottom: 24, fontSize: 13 }}>
                        All models trained and data downloaded. You're ready to go.
                    </p>
                    <button
                        onClick={onComplete}
                        style={{
                            background: 'var(--positive)', color: '#000', border: 'none', borderRadius: 8,
                            padding: '12px 32px', fontSize: 14, fontWeight: 700, cursor: 'pointer',
                            fontFamily: 'var(--font-mono)',
                        }}
                    >
                        Launch Dashboard →
                    </button>
                </div>
            )}
        </div>
    );
}
