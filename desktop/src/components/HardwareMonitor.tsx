import { useState, useEffect, useCallback } from 'react';

type HwData = {
    cpu: {
        percent: number;
        per_core: number[];
        cores_physical: number;
        cores_logical: number;
        freq_mhz: number;
        temp_c?: number;
    };
    ram: {
        total_gb: number; used_gb: number; percent: number;
        app_used_gb?: number; others_used_gb?: number;
    };
    gpu: {
        available: boolean;
        name?: string;
        vram_total_gb?: number;
        vram_used_gb?: number;
        vram_percent?: number;
        temp_c?: number;
        utilization_percent?: number;
        fan_speed_percent?: number;
        power_draw_w?: number;
        power_limit_w?: number;
    };
    disk: { total_gb: number; used_gb: number; percent: number };
    uptime: string;
};

/* ── helpers ─────────────────────────────────────────────── */
function pctColor(p: number) {
    if (p > 90) return '#FF5252';
    if (p > 70) return '#FFD740';
    return '#00E676';
}
function tempColor(t: number) {
    if (t > 80) return '#FF5252';
    if (t > 65) return '#FFD740';
    return '#00E676';
}

/* ── Thin horizontal bar ─────────────────────────────────── */
function Bar({ pct, color }: { pct: number; color: string }) {
    return (
        <div style={{
            height: 3, borderRadius: 2, background: 'rgba(255,255,255,0.06)',
            overflow: 'hidden', marginTop: 4,
        }}>
            <div style={{
                height: '100%', borderRadius: 2, width: `${Math.max(2, pct)}%`,
                background: `linear-gradient(90deg, ${color}66, ${color})`,
                transition: 'width 0.6s ease',
            }} />
        </div>
    );
}

/* ── Stat Row ────────────────────────────────────────────── */
function Row({ label, value, sub, color }: {
    label: string; value: string; sub?: string; color?: string;
}) {
    return (
        <div style={{
            display: 'flex', alignItems: 'baseline',
            justifyContent: 'space-between', gap: 8,
        }}>
            <span style={{ fontSize: 11, color: 'var(--text-2)' }}>{label}</span>
            <span style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
                <span className="mono" style={{
                    fontSize: 13, fontWeight: 600, color: color ?? 'var(--text-1)',
                }}>
                    {value}
                </span>
                {sub && (
                    <span className="mono" style={{ fontSize: 10, color: 'var(--text-2)' }}>
                        {sub}
                    </span>
                )}
            </span>
        </div>
    );
}


/* ═══════════════════════════════════════════════════════════════
   MAIN
   ═══════════════════════════════════════════════════════════════ */

export default function HardwareMonitor() {
    const [data, setData] = useState<HwData | null>(null);

    const refresh = useCallback(() => {
        fetch('http://127.0.0.1:8420/api/hardware')
            .then(r => r.json())
            .then(setData)
            .catch(() => { });
    }, []);

    useEffect(() => {
        refresh();
        const iv = setInterval(refresh, 2000);
        return () => clearInterval(iv);
    }, [refresh]);

    if (!data) {
        return (
            <div className="card card-compact animate-in" style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                height: '100%', fontSize: 11, color: 'var(--text-2)',
            }}>
                Connecting…
            </div>
        );
    }

    const { gpu, cpu, ram, disk } = data;

    /* section wrapper */
    const sec = (children: React.ReactNode, highlight?: boolean) => (
        <div style={{
            padding: '10px 12px',
            borderRadius: 8,
            background: highlight ? 'rgba(139,92,246,0.05)' : 'rgba(255,255,255,0.02)',
            border: highlight ? '1px solid rgba(139,92,246,0.12)' : '1px solid var(--border)',
        }}>
            {children}
        </div>
    );

    return (
        <div className="card card-compact animate-in">
            {/* ── Header ── */}
            <div className="card-header">
                <span className="card-title">Hardware</span>
                <span className="mono" style={{ fontSize: 10, color: 'var(--text-2)' }}>
                    {data.uptime}
                </span>
            </div>

            <div className="flex-col gap-6" style={{ fontSize: 12 }}>

                {/* ═══ GPU Section ═══ */}
                {gpu.available && sec(
                    <>
                        {/* GPU name row */}
                        <div style={{
                            display: 'flex', alignItems: 'center', gap: 8,
                            marginBottom: 10,
                        }}>
                            <span style={{
                                fontSize: 9, fontWeight: 700, color: '#8b5cf6',
                                background: 'rgba(139,92,246,0.12)',
                                padding: '2px 6px', borderRadius: 3,
                                letterSpacing: 0.3, fontFamily: 'var(--font-mono)',
                            }}>GPU</span>
                            <span style={{ fontSize: 11, color: 'var(--text-1)', fontWeight: 500 }}>
                                {gpu.name}
                            </span>
                        </div>

                        {/* GPU stats grid — 3 columns */}
                        <div style={{
                            display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)',
                            gap: 8, marginBottom: 10,
                        }}>
                            {/* Temp */}
                            <div style={{
                                padding: '6px 8px', borderRadius: 6, textAlign: 'center',
                                background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border)',
                            }}>
                                <div style={{ fontSize: 9, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 2 }}>
                                    Temp
                                </div>
                                <div className="mono" style={{
                                    fontSize: 16, fontWeight: 700,
                                    color: gpu.temp_c != null ? tempColor(gpu.temp_c) : 'var(--text-2)',
                                }}>
                                    {gpu.temp_c != null ? `${gpu.temp_c}°` : '—'}
                                </div>
                            </div>

                            {/* Load */}
                            <div style={{
                                padding: '6px 8px', borderRadius: 6, textAlign: 'center',
                                background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border)',
                            }}>
                                <div style={{ fontSize: 9, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 2 }}>
                                    Load
                                </div>
                                <div className="mono" style={{
                                    fontSize: 16, fontWeight: 700,
                                    color: gpu.utilization_percent != null ? pctColor(100 - gpu.utilization_percent) : 'var(--text-2)',
                                }}>
                                    {gpu.utilization_percent != null ? `${gpu.utilization_percent}%` : '—'}
                                </div>
                            </div>

                            {/* Power */}
                            <div style={{
                                padding: '6px 8px', borderRadius: 6, textAlign: 'center',
                                background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border)',
                            }}>
                                <div style={{ fontSize: 9, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 2 }}>
                                    Power
                                </div>
                                <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: '#FFD740' }}>
                                    {gpu.power_draw_w != null ? `${Math.round(gpu.power_draw_w)}W` : '—'}
                                </div>
                            </div>
                        </div>

                        {/* VRAM bar */}
                        <Row
                            label="VRAM"
                            value={`${(gpu.vram_used_gb ?? 0).toFixed(1)} GB`}
                            sub={`/ ${(gpu.vram_total_gb ?? 0).toFixed(1)} GB`}
                        />
                        <Bar pct={gpu.vram_percent ?? 0} color="#8b5cf6" />

                        {/* Fan speed row */}
                        {(gpu.fan_speed_percent != null || gpu.power_limit_w != null) && (
                            <div style={{
                                display: 'flex', justifyContent: 'space-between',
                                marginTop: 8, fontSize: 10, color: 'var(--text-2)',
                            }}>
                                {gpu.fan_speed_percent != null && (
                                    <span className="mono">Fan {gpu.fan_speed_percent}%</span>
                                )}
                                {gpu.power_limit_w != null && (
                                    <span className="mono">TDP {gpu.power_limit_w}W</span>
                                )}
                            </div>
                        )}
                    </>,
                    true, /* highlighted section */
                )}

                {/* ═══ CPU Section ═══ */}
                {sec(
                    <>
                        <div style={{
                            display: 'flex', alignItems: 'center', gap: 8,
                            marginBottom: 10,
                        }}>
                            <span style={{
                                fontSize: 9, fontWeight: 700, color: '#6366f1',
                                background: 'rgba(99,102,241,0.12)',
                                padding: '2px 6px', borderRadius: 3,
                                letterSpacing: 0.3, fontFamily: 'var(--font-mono)',
                            }}>CPU</span>
                            <span className="mono" style={{ fontSize: 10, color: 'var(--text-2)' }}>
                                {cpu.cores_physical}C / {cpu.cores_logical}T
                                {cpu.freq_mhz > 0 ? ` · ${(cpu.freq_mhz / 1000).toFixed(1)} GHz` : ''}
                            </span>
                            <span className="mono" style={{
                                marginLeft: 'auto', fontSize: 13, fontWeight: 600,
                                color: pctColor(100 - cpu.percent),
                            }}>
                                {cpu.percent.toFixed(0)}%
                            </span>
                        </div>

                        <Bar pct={cpu.percent} color="#6366f1" />

                        {/* Per-core grid */}
                        {cpu.per_core.length > 0 && (
                            <div style={{
                                display: 'grid',
                                gridTemplateColumns: `repeat(${Math.min(cpu.per_core.length, 16)}, 1fr)`,
                                gap: 3, marginTop: 8,
                            }}>
                                {cpu.per_core.map((pct, i) => (
                                    <div key={i} title={`Core ${i}: ${pct.toFixed(0)}%`} style={{
                                        height: 20, borderRadius: 3,
                                        background: 'rgba(255,255,255,0.04)',
                                        position: 'relative', overflow: 'hidden',
                                    }}>
                                        <div style={{
                                            position: 'absolute', bottom: 0, left: 0, right: 0,
                                            height: `${pct}%`, borderRadius: 3,
                                            background: pct > 90 ? '#FF5252'
                                                : pct > 60 ? '#FFD740' : '#6366f1',
                                            transition: 'height 0.6s ease',
                                        }} />
                                    </div>
                                ))}
                            </div>
                        )}
                    </>
                )}

                {/* ═══ Memory + Storage ═══ */}
                {sec(
                    <>
                        {/* RAM label row */}
                        <div style={{
                            display: 'flex', alignItems: 'baseline',
                            justifyContent: 'space-between', gap: 8,
                        }}>
                            <span style={{ fontSize: 11, color: 'var(--text-2)' }}>RAM</span>
                            <span className="mono" style={{ fontSize: 10, color: 'var(--text-2)' }}>
                                {ram.used_gb.toFixed(1)} / {ram.total_gb.toFixed(1)} GB
                            </span>
                        </div>

                        {/* Dual-color RAM bar: green = our app, red = others */}
                        <div style={{
                            height: 6, borderRadius: 3, background: 'rgba(255,255,255,0.06)',
                            overflow: 'hidden', marginTop: 4, display: 'flex',
                        }}>
                            {/* Others (red/orange) — rendered first (left side) */}
                            {ram.others_used_gb != null && ram.total_gb > 0 && (
                                <div style={{
                                    height: '100%',
                                    width: `${(ram.others_used_gb / ram.total_gb) * 100}%`,
                                    background: 'linear-gradient(90deg, #FF525266, #FF5252)',
                                    transition: 'width 0.6s ease',
                                }} />
                            )}
                            {/* Our app (green) */}
                            {ram.app_used_gb != null && ram.total_gb > 0 && (
                                <div style={{
                                    height: '100%',
                                    width: `${(ram.app_used_gb / ram.total_gb) * 100}%`,
                                    background: 'linear-gradient(90deg, #00E67666, #00E676)',
                                    transition: 'width 0.6s ease',
                                }} />
                            )}
                        </div>

                        {/* Legend */}
                        <div style={{
                            display: 'flex', gap: 12, marginTop: 5,
                            fontSize: 10, color: 'var(--text-2)',
                        }}>
                            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                                <span style={{ width: 6, height: 6, borderRadius: 2, background: '#00E676', display: 'inline-block' }} />
                                <span className="mono">
                                    Predictor {ram.app_used_gb != null ? `${ram.app_used_gb.toFixed(1)} GB` : '—'}
                                </span>
                            </span>
                            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                                <span style={{ width: 6, height: 6, borderRadius: 2, background: '#FF5252', display: 'inline-block' }} />
                                <span className="mono">
                                    Other {ram.others_used_gb != null ? `${ram.others_used_gb.toFixed(1)} GB` : '—'}
                                </span>
                            </span>
                        </div>

                        <div style={{ marginTop: 10 }}>
                            <Row
                                label="Disk"
                                value={`${disk.used_gb.toFixed(0)} GB`}
                                sub={`/ ${disk.total_gb.toFixed(0)} GB`}
                                color={pctColor(100 - disk.percent)}
                            />
                            <Bar pct={disk.percent} color="#22c55e" />
                        </div>
                    </>
                )}

            </div>
        </div>
    );
}
