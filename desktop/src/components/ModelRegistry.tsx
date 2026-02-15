import { useEffect, useState } from 'react';

type Checkpoint = { epoch: number; accuracy: number; filename: string };

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
    best_checkpoint_acc: number | null;
    epochs_trained: number;
    checkpoint_count: number;
    checkpoints: Checkpoint[];
};

type ModelsResponse = {
    models: ModelInfo[];
    active_arch: string;
    running_arch: string | null;
    gpu: { name: string; vram_total_gb: number; vram_free_gb: number };
};

const ACC_COLORS = {
    high: '#69F0AE',   // ≥75%
    mid: '#FFAB40',    // ≥60%
    low: '#FF5252',    // <60%
    none: 'var(--text-3)',
};

function accColor(acc: number | null): string {
    if (acc == null || acc === 0) return ACC_COLORS.none;
    if (acc >= 75) return ACC_COLORS.high;
    if (acc >= 60) return ACC_COLORS.mid;
    return ACC_COLORS.low;
}

export function ModelRegistry() {
    const [data, setData] = useState<ModelsResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [expandedModel, setExpandedModel] = useState<string | null>(null);

    useEffect(() => {
        let active = true;
        const fetchModels = async () => {
            try {
                const res = await fetch('http://127.0.0.1:8420/api/models');
                const json = await res.json();
                if (active) {
                    setData(json);
                    setLoading(false);
                }
            } catch {
                if (active) setLoading(false);
            }
        };
        fetchModels();
        // Refresh every 30s (in case training finishes and new checkpoints appear)
        const timer = setInterval(fetchModels, 30_000);
        return () => { active = false; clearInterval(timer); };
    }, []);

    if (loading) {
        return (
            <div className="card animate-in" style={{ minHeight: 120 }}>
                <div className="stat-label">Model Registry</div>
                <div className="stat-value xs text-muted" style={{ marginTop: 12 }}>Loading…</div>
            </div>
        );
    }

    if (!data || data.models.length === 0) {
        return (
            <div className="card animate-in">
                <div className="stat-label">Model Registry</div>
                <div className="stat-value xs text-muted" style={{ marginTop: 8 }}>No models found</div>
            </div>
        );
    }

    return (
        <div className="card animate-in" style={{ padding: 0 }}>
            {/* Header */}
            <div style={{
                padding: '10px 14px 6px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
            }}>
                <span className="stat-label" style={{ margin: 0 }}>Model Registry</span>
                <span className="mono" style={{
                    fontSize: 9,
                    padding: '2px 6px',
                    borderRadius: 3,
                    background: 'rgba(129,140,248,0.12)',
                    color: '#818cf8',
                    fontWeight: 600,
                }}>
                    {data.gpu.name}
                </span>
            </div>

            {/* Model rows */}
            <div style={{ padding: '0 4px 8px' }}>
                {data.models.map((m) => {
                    const isRunning = m.key === data.running_arch;
                    const isExpanded = expandedModel === m.key;
                    const bestAcc = m.best_checkpoint_acc;

                    return (
                        <div key={m.key}>
                            {/* Main row */}
                            <div
                                onClick={() => setExpandedModel(isExpanded ? null : m.key)}
                                style={{
                                    display: 'grid',
                                    gridTemplateColumns: '1fr auto auto',
                                    gap: 8,
                                    alignItems: 'center',
                                    padding: '7px 10px',
                                    margin: '2px 0',
                                    borderRadius: 6,
                                    cursor: m.checkpoint_count > 0 ? 'pointer' : 'default',
                                    background: isRunning
                                        ? 'rgba(129,140,248,0.08)'
                                        : 'transparent',
                                    borderLeft: isRunning
                                        ? '2px solid #818cf8'
                                        : '2px solid transparent',
                                    transition: 'background 0.15s',
                                }}
                            >
                                {/* Name + params */}
                                <div>
                                    <div style={{
                                        fontSize: 12,
                                        fontWeight: isRunning ? 600 : 400,
                                        color: isRunning ? '#fff' : 'var(--text-1)',
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 6,
                                    }}>
                                        {m.label.split('(')[0].trim()}
                                        {isRunning && (
                                            <span style={{
                                                fontSize: 8,
                                                padding: '1px 4px',
                                                borderRadius: 3,
                                                background: 'rgba(0,230,118,0.15)',
                                                color: '#69F0AE',
                                                fontWeight: 700,
                                                letterSpacing: 0.5,
                                            }}>ACTIVE</span>
                                        )}
                                    </div>
                                    <div className="mono" style={{
                                        fontSize: 9,
                                        color: 'var(--text-3)',
                                        marginTop: 1,
                                    }}>
                                        {m.params} · {m.vram_gb}GB VRAM
                                        {m.epochs_trained > 0 && ` · ${m.epochs_trained} epochs`}
                                    </div>
                                </div>

                                {/* Accuracy */}
                                <div className="mono" style={{
                                    fontSize: 15,
                                    fontWeight: 700,
                                    color: accColor(bestAcc),
                                    textAlign: 'right',
                                    minWidth: 55,
                                }}>
                                    {bestAcc != null && bestAcc > 0 ? `${bestAcc}%` : '—'}
                                </div>

                                {/* Status indicator */}
                                <div style={{
                                    width: 8,
                                    height: 8,
                                    borderRadius: '50%',
                                    background: m.has_weights ? '#69F0AE' : 'var(--text-3)',
                                    opacity: m.has_weights ? 0.8 : 0.3,
                                    flexShrink: 0,
                                }} title={m.has_weights ? 'Weights available' : 'No weights'} />
                            </div>

                            {/* Expanded checkpoint timeline */}
                            {isExpanded && m.checkpoints.length > 0 && (
                                <div style={{
                                    padding: '4px 12px 8px 20px',
                                    display: 'flex',
                                    flexWrap: 'wrap',
                                    gap: '3px 6px',
                                }}>
                                    {m.checkpoints.map((ckpt) => (
                                        <div key={ckpt.epoch} className="mono" style={{
                                            fontSize: 9,
                                            padding: '2px 5px',
                                            borderRadius: 3,
                                            background: ckpt.accuracy === bestAcc
                                                ? 'rgba(105,240,174,0.12)'
                                                : 'rgba(255,255,255,0.04)',
                                            color: ckpt.accuracy === bestAcc
                                                ? '#69F0AE'
                                                : 'var(--text-2)',
                                            border: ckpt.accuracy === bestAcc
                                                ? '1px solid rgba(105,240,174,0.25)'
                                                : '1px solid rgba(255,255,255,0.06)',
                                        }}>
                                            E{ckpt.epoch}: {ckpt.accuracy}%
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
