import { useEffect, useState } from 'react';
import { IconMonitor, IconBrain, IconCrosshair, IconChart } from './Icons';

interface HealthData {
    gpu_name: string;
    gpu_vram_total_gb: number;
    gpu_vram_used_gb: number;
    model_trained: boolean;
    model_version: string;
    feature_count: number;
    validation_accuracy: number;
    ensemble_weights: { xgb: number; lstm: number };
    model_age_hours?: number;
    model_last_trained?: string;
    data_size_mb?: number;
}

export default function SystemHealth() {
    const [health, setHealth] = useState<HealthData | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchHealth = async () => {
            try {
                const res = await fetch('http://localhost:8420/api/system-health');
                if (res.ok) setHealth(await res.json());
            } catch { /* ignore */ }
            setLoading(false);
        };
        fetchHealth();
        const interval = setInterval(fetchHealth, 15000);
        return () => clearInterval(interval);
    }, []);

    if (loading || !health) {
        return <div className="system-health loading">Loading system health...</div>;
    }

    const vramPct = health.gpu_vram_total_gb > 0
        ? (health.gpu_vram_used_gb / health.gpu_vram_total_gb) * 100
        : 0;

    const formatAge = (hours?: number) => {
        if (!hours) return 'Never';
        if (hours < 1) return `${Math.round(hours * 60)}m ago`;
        if (hours < 24) return `${Math.round(hours)}h ago`;
        return `${Math.round(hours / 24)}d ago`;
    };

    return (
        <div className="system-health">
            <h3>System Health</h3>
            <div className="sh-grid">
                {/* GPU */}
                <div className="sh-card">
                    <span className="sh-icon"><IconMonitor size={20} /></span>
                    <div className="sh-info">
                        <span className="sh-label">GPU</span>
                        <span className="sh-value">{health.gpu_name}</span>
                        <div className="sh-progress-track">
                            <div
                                className="sh-progress-fill"
                                style={{
                                    width: `${vramPct}%`,
                                    background: vramPct > 80 ? '#e57373' : vramPct > 50 ? '#ffd54f' : '#81c784',
                                }}
                            />
                        </div>
                        <span className="sh-sub">
                            {health.gpu_vram_used_gb}GB / {health.gpu_vram_total_gb}GB VRAM
                        </span>
                    </div>
                </div>

                {/* Model */}
                <div className="sh-card">
                    <span className="sh-icon"><IconBrain size={20} /></span>
                    <div className="sh-info">
                        <span className="sh-label">Model</span>
                        <span className="sh-value">{health.model_version} • {health.feature_count} features</span>
                        <span className="sh-sub">
                            {health.model_trained ? `Trained ${formatAge(health.model_age_hours)}` : 'Not trained'}
                        </span>
                    </div>
                </div>

                {/* Accuracy */}
                <div className="sh-card">
                    <span className="sh-icon"><IconCrosshair size={20} /></span>
                    <div className="sh-info">
                        <span className="sh-label">Accuracy</span>
                        <span className={`sh-value ${health.validation_accuracy > 52 ? 'positive' : 'neutral'}`}>
                            {health.validation_accuracy.toFixed(1)}%
                        </span>
                        <span className="sh-sub">
                            XGB: {(health.ensemble_weights.xgb * 100).toFixed(0)}% • LSTM: {(health.ensemble_weights.lstm * 100).toFixed(0)}%
                        </span>
                    </div>
                </div>

                {/* Data */}
                <div className="sh-card">
                    <span className="sh-icon"><IconChart size={20} /></span>
                    <div className="sh-info">
                        <span className="sh-label">Data</span>
                        <span className="sh-value">{health.data_size_mb ? `${health.data_size_mb} MB` : 'No data'}</span>
                        <span className="sh-sub">Parquet format</span>
                    </div>
                </div>
            </div>
        </div>
    );
}
