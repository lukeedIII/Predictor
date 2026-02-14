import { useLiveConnected } from '../stores/liveStore';
import { useApi } from '../hooks/useApi';
import { IconCpu, IconWifi } from './Icons';

type AiService = {
    connected: boolean;
    models?: string[];
    active_model?: string | null;
    model_count?: number;
    key_preview?: string | null;
    reason?: string;
};

type HealthData = {
    gpu_available?: boolean;
    gpu_name?: string;
    gpu_memory_used?: number;
    gpu_memory_total?: number;
    gpu_vram_total_gb?: number;
    gpu_vram_used_gb?: number;
    model_trained?: boolean;
    uptime?: string;
    uptime_seconds?: number;
    is_retraining?: boolean;
    retrain_count?: number;
    next_retrain_countdown?: string;
    api_keys?: Record<string, boolean>;
    ai_services?: {
        ollama?: AiService;
        openai?: AiService;
        gemini?: AiService;
    };
};

export default function SystemHealth() {
    const { connected, wsConnected } = useLiveConnected();
    const { data } = useApi<HealthData>('/api/health', 10_000);

    const dot = (ok: boolean) => <span className={`status-dot ${ok ? 'online' : 'offline'}`} />;

    const gpuPct = data?.gpu_vram_total_gb
        ? ((data.gpu_vram_used_gb ?? 0) / data.gpu_vram_total_gb * 100).toFixed(0)
        : null;

    const ai = data?.ai_services;

    return (
        <div className="card card-compact animate-in">
            <div className="card-header">
                <span className="card-title">System</span>
                {data?.uptime && (
                    <span className="mono" style={{ fontSize: 10, color: 'var(--text-2)' }}>
                        ⏱ {data.uptime}
                    </span>
                )}
            </div>
            <div className="flex-col gap-8" style={{ fontSize: 12 }}>
                <div className="flex items-center justify-between">
                    <span className="flex items-center gap-4">
                        <IconWifi style={{ width: 12, height: 12, color: 'var(--text-2)' }} />
                        <span style={{ color: 'var(--text-1)' }}>Backend</span>
                    </span>
                    {dot(connected)}
                </div>
                <div className="flex items-center justify-between">
                    <span className="flex items-center gap-4">
                        <IconWifi style={{ width: 12, height: 12, color: 'var(--text-2)' }} />
                        <span style={{ color: 'var(--text-1)' }}>Binance WS</span>
                    </span>
                    {dot(wsConnected)}
                </div>
                {data?.gpu_name != null && (
                    <div className="flex items-center justify-between">
                        <span className="flex items-center gap-4">
                            <IconCpu style={{ width: 12, height: 12, color: 'var(--text-2)' }} />
                            <span style={{ color: 'var(--text-1)' }}>GPU</span>
                        </span>
                        <span className="mono" style={{ fontSize: 11, color: 'var(--text-0)' }}>
                            {gpuPct ? `${gpuPct}%` : 'Ready'}
                        </span>
                    </div>
                )}
                {data?.model_trained != null && (
                    <div className="flex items-center justify-between">
                        <span style={{ color: 'var(--text-1)' }}>Model</span>
                        <span className={data.model_trained ? 'badge badge-long' : 'badge badge-warning'}>
                            {data.model_trained ? 'Trained' : 'Untrained'}
                        </span>
                    </div>
                )}

                {/* ── AI Services ── */}
                {ai && (
                    <>
                        <div style={{
                            borderTop: '1px solid rgba(255,255,255,0.06)',
                            margin: '4px 0 2px',
                            paddingTop: 6,
                        }}>
                            <span style={{
                                fontSize: 9,
                                fontWeight: 700,
                                letterSpacing: '1.2px',
                                textTransform: 'uppercase',
                                color: 'var(--text-2)',
                            }}>AI Providers</span>
                        </div>

                        {/* Ollama */}
                        <div className="flex items-center justify-between">
                            <span className="flex items-center gap-4">
                                <span style={{ fontSize: 12 }}>🦙</span>
                                <span style={{ color: 'var(--text-1)' }}>Ollama</span>
                            </span>
                            {ai.ollama?.connected ? (
                                <span className="flex items-center gap-4">
                                    <span className="mono" style={{
                                        fontSize: 9,
                                        color: '#0ECB81',
                                        maxWidth: 90,
                                        overflow: 'hidden',
                                        textOverflow: 'ellipsis',
                                        whiteSpace: 'nowrap',
                                    }}>
                                        {ai.ollama.active_model
                                            ? ai.ollama.active_model.split(':')[0]
                                            : `${ai.ollama.model_count} models`}
                                    </span>
                                    {dot(true)}
                                </span>
                            ) : (
                                <span className="flex items-center gap-4">
                                    <span style={{ fontSize: 9, color: 'var(--text-2)' }}>
                                        {ai.ollama?.reason || 'Off'}
                                    </span>
                                    {dot(false)}
                                </span>
                            )}
                        </div>

                        {/* OpenAI */}
                        <div className="flex items-center justify-between">
                            <span className="flex items-center gap-4">
                                <span style={{ fontSize: 12 }}>⚡</span>
                                <span style={{ color: 'var(--text-1)' }}>OpenAI</span>
                            </span>
                            {ai.openai?.connected ? (
                                <span className="flex items-center gap-4">
                                    <span className="mono" style={{ fontSize: 9, color: '#0ECB81' }}>
                                        Key set
                                    </span>
                                    {dot(true)}
                                </span>
                            ) : (
                                <span className="flex items-center gap-4">
                                    <span style={{ fontSize: 9, color: 'var(--text-2)' }}>No key</span>
                                    {dot(false)}
                                </span>
                            )}
                        </div>

                        {/* Gemini */}
                        <div className="flex items-center justify-between">
                            <span className="flex items-center gap-4">
                                <span style={{ fontSize: 12 }}>💎</span>
                                <span style={{ color: 'var(--text-1)' }}>Gemini</span>
                            </span>
                            {ai.gemini?.connected ? (
                                <span className="flex items-center gap-4">
                                    <span className="mono" style={{ fontSize: 9, color: '#0ECB81' }}>
                                        Key set
                                    </span>
                                    {dot(true)}
                                </span>
                            ) : (
                                <span className="flex items-center gap-4">
                                    <span style={{ fontSize: 9, color: 'var(--text-2)' }}>No key</span>
                                    {dot(false)}
                                </span>
                            )}
                        </div>
                    </>
                )}

                {/* ── Next Retrain Countdown ── */}
                {data?.next_retrain_countdown != null && (
                    <div className="flex items-center justify-between">
                        <span style={{ color: 'var(--text-1)' }}>
                            {data.is_retraining ? '🔄 Retraining' : '⏳ Next Train'}
                        </span>
                        <span className="mono" style={{
                            fontSize: 11,
                            color: data.is_retraining ? '#FF9100' : data.next_retrain_countdown === 'imminent' ? '#00E676' : 'var(--text-0)',
                            fontWeight: data.is_retraining ? 600 : 400,
                        }}>
                            {data.is_retraining ? 'in progress...' : data.next_retrain_countdown}
                        </span>
                    </div>
                )}
                {data?.retrain_count != null && data.retrain_count > 0 && (
                    <div className="flex items-center justify-between">
                        <span style={{ color: 'var(--text-1)' }}>Retrains</span>
                        <span className="mono" style={{ fontSize: 11, color: 'var(--text-0)' }}>
                            {data.retrain_count}×
                        </span>
                    </div>
                )}
            </div>
        </div>
    );
}
