import { useState, useEffect, useRef, useCallback } from 'react';

// ─── Swiss Cities ────────────────────────────────
const CITIES = [
    { name: 'Zürich', lat: 47.3769, lon: 8.5417 },
    { name: 'Bern', lat: 46.9481, lon: 7.4474 },
    { name: 'Geneva', lat: 46.2044, lon: 6.1432 },
    { name: 'Lucerne', lat: 47.0502, lon: 8.3093 },
    { name: 'Basel', lat: 47.5596, lon: 7.5886 },
];

const STORAGE_KEY = 'nexus-weather-city-idx';

// ─── Weather codes → meta ────────────────────────
type WeatherType = 'clear' | 'cloudy' | 'fog' | 'drizzle' | 'rain' | 'snow' | 'storm';

function weatherMeta(code: number): { icon: string; label: string; type: WeatherType } {
    if (code === 0) return { icon: '☀️', label: 'Clear Sky', type: 'clear' };
    if (code <= 3) return { icon: '⛅', label: 'Cloudy', type: 'cloudy' };
    if (code <= 48) return { icon: '🌫️', label: 'Fog', type: 'fog' };
    if (code <= 55) return { icon: '🌦️', label: 'Drizzle', type: 'drizzle' };
    if (code <= 67) return { icon: '🌧️', label: 'Rain', type: 'rain' };
    if (code <= 77) return { icon: '🌨️', label: 'Snow', type: 'snow' };
    if (code <= 86) return { icon: '🌨️', label: 'Snow Showers', type: 'snow' };
    if (code <= 99) return { icon: '⛈️', label: 'Thunderstorm', type: 'storm' };
    return { icon: '🌡️', label: '—', type: 'cloudy' };
}

type CityWeather = {
    name: string;
    temp: number;
    code: number;
    wind: number;
    humidity: number;
};

// ─── Particle System ─────────────────────────────
type Particle = {
    x: number;
    y: number;
    vx: number;
    vy: number;
    size: number;
    opacity: number;
    life: number;
};

function createParticles(type: WeatherType, w: number, h: number): Particle[] {
    const particles: Particle[] = [];
    // More particles for every type — always visually alive
    const count = type === 'storm' ? 100
        : type === 'rain' ? 80
            : type === 'drizzle' ? 40
                : type === 'snow' ? 50
                    : type === 'clear' ? 30   // stars + glow orbs
                        : type === 'fog' ? 20
                            : type === 'cloudy' ? 18  // drifting cloud blobs
                                : 12;

    for (let i = 0; i < count; i++) {
        const isStar = type === 'clear' && i < 22;
        particles.push({
            x: Math.random() * w,
            y: Math.random() * h,
            vx: type === 'rain' || type === 'storm' ? -0.5 + Math.random() * -0.5
                : type === 'cloudy' ? 0.1 + Math.random() * 0.2
                    : (Math.random() - 0.5) * 0.3,
            vy: type === 'rain' ? 3 + Math.random() * 5
                : type === 'storm' ? 5 + Math.random() * 6
                    : type === 'drizzle' ? 1.5 + Math.random() * 2
                        : type === 'snow' ? 0.3 + Math.random() * 0.8
                            : type === 'clear' ? (isStar ? 0 : 0.05 + Math.random() * 0.1)
                                : type === 'cloudy' ? 0.02 + Math.random() * 0.04
                                    : 0.03 + Math.random() * 0.05,
            size: type === 'snow' ? 2 + Math.random() * 3
                : type === 'rain' || type === 'storm' ? 1
                    : type === 'clear' ? (isStar ? 0.5 + Math.random() * 1.5 : 3 + Math.random() * 5)
                        : type === 'fog' ? 8 + Math.random() * 16
                            : type === 'cloudy' ? 6 + Math.random() * 12
                                : 4 + Math.random() * 8,
            opacity: type === 'clear' ? (isStar ? 0.3 + Math.random() * 0.7 : 0.08 + Math.random() * 0.15)
                : type === 'fog' ? 0.03 + Math.random() * 0.06
                    : type === 'cloudy' ? 0.04 + Math.random() * 0.08
                        : 0.3 + Math.random() * 0.5,
            life: Math.random() * Math.PI * 2, // phase offset for twinkling
        });
    }
    return particles;
}

function drawParticles(ctx: CanvasRenderingContext2D, particles: Particle[], type: WeatherType, w: number, h: number) {
    ctx.clearRect(0, 0, w, h);
    const t = Date.now() * 0.001;

    // Background gradient based on weather
    const grad = ctx.createLinearGradient(0, 0, w, h);
    if (type === 'clear') {
        grad.addColorStop(0, 'rgba(15, 23, 42, 0.3)');
        grad.addColorStop(1, 'rgba(30, 41, 59, 0.15)');
    } else if (type === 'rain' || type === 'drizzle') {
        grad.addColorStop(0, 'rgba(59, 130, 246, 0.08)');
        grad.addColorStop(1, 'rgba(30, 64, 175, 0.05)');
    } else if (type === 'storm') {
        grad.addColorStop(0, 'rgba(88, 28, 135, 0.10)');
        grad.addColorStop(1, 'rgba(30, 58, 138, 0.08)');
    } else if (type === 'snow') {
        grad.addColorStop(0, 'rgba(186, 230, 253, 0.08)');
        grad.addColorStop(1, 'rgba(147, 197, 253, 0.05)');
    } else if (type === 'fog') {
        grad.addColorStop(0, 'rgba(148, 163, 184, 0.08)');
        grad.addColorStop(1, 'rgba(100, 116, 139, 0.05)');
    } else {
        grad.addColorStop(0, 'rgba(148, 163, 184, 0.06)');
        grad.addColorStop(1, 'rgba(71, 85, 105, 0.04)');
    }
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    for (let pi = 0; pi < particles.length; pi++) {
        const p = particles[pi];

        if (type === 'clear') {
            const isStar = pi < 22;
            if (isStar) {
                // ★ Twinkling stars — pulsing opacity
                const twinkle = 0.3 + Math.abs(Math.sin(t * (1.5 + pi * 0.3) + p.life)) * 0.7;
                ctx.globalAlpha = twinkle;
                // Star core
                ctx.fillStyle = '#F8FAFC';
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fill();
                // Star cross glow
                ctx.globalAlpha = twinkle * 0.3;
                ctx.strokeStyle = '#E2E8F0';
                ctx.lineWidth = 0.5;
                ctx.beginPath();
                ctx.moveTo(p.x - p.size * 3, p.y);
                ctx.lineTo(p.x + p.size * 3, p.y);
                ctx.moveTo(p.x, p.y - p.size * 3);
                ctx.lineTo(p.x, p.y + p.size * 3);
                ctx.stroke();
            } else {
                // Warm floating glow orbs
                const pulse = 0.5 + Math.sin(t * 0.8 + p.life) * 0.3;
                ctx.globalAlpha = p.opacity * pulse;
                ctx.fillStyle = 'rgba(255, 215, 0, 0.6)';
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = 'rgba(255, 200, 50, 0.12)';
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size * 4, 0, Math.PI * 2);
                ctx.fill();
                p.x += p.vx;
                p.y += p.vy;
            }
        } else if (type === 'rain' || type === 'drizzle') {
            ctx.globalAlpha = p.opacity;
            ctx.strokeStyle = type === 'drizzle' ? '#93C5FD' : '#60A5FA';
            ctx.lineWidth = type === 'drizzle' ? 0.5 : 1;
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p.x + p.vx * 3, p.y + p.vy * 2.5);
            ctx.stroke();
            p.x += p.vx;
            p.y += p.vy;
        } else if (type === 'storm') {
            ctx.globalAlpha = p.opacity;
            ctx.strokeStyle = '#818CF8';
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p.x + p.vx * 4, p.y + p.vy * 3);
            ctx.stroke();
            p.x += p.vx;
            p.y += p.vy;
            // Lightning flash (slightly more frequent)
            if (Math.random() < 0.003) {
                ctx.globalAlpha = 0.2;
                ctx.fillStyle = '#E0E7FF';
                ctx.fillRect(0, 0, w, h);
            }
        } else if (type === 'snow') {
            const drift = Math.sin(p.life + t * 1.2) * 0.4;
            ctx.globalAlpha = p.opacity;
            ctx.fillStyle = '#E0F2FE';
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fill();
            // Soft glow around each flake
            ctx.globalAlpha = p.opacity * 0.2;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size * 2.5, 0, Math.PI * 2);
            ctx.fill();
            p.x += p.vx + drift;
            p.y += p.vy;
        } else if (type === 'fog') {
            const pulse = 0.7 + Math.sin(t * 0.5 + p.life) * 0.3;
            ctx.globalAlpha = p.opacity * pulse;
            ctx.fillStyle = 'rgba(148, 163, 184, 1)';
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fill();
            p.x += p.vx;
            p.y += p.vy;
        } else {
            // Cloudy — drifting gray cloud blobs
            const pulse = 0.6 + Math.sin(t * 0.4 + p.life) * 0.4;
            ctx.globalAlpha = p.opacity * pulse;
            ctx.fillStyle = 'rgba(100, 116, 139, 1)';
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size * (0.8 + Math.sin(t * 0.3 + pi) * 0.2), 0, Math.PI * 2);
            ctx.fill();
            p.x += p.vx;
            p.y += p.vy;
        }

        // Wrap around
        if (p.y > h + 20) { p.y = -20; p.x = Math.random() * w; }
        if (p.y < -20) { p.y = h + 20; p.x = Math.random() * w; }
        if (p.x > w + 20) p.x = -20;
        if (p.x < -20) p.x = w + 20;
    }

    ctx.globalAlpha = 1;
}

// ─── Component ───────────────────────────────────
export default function SwissWeather() {
    const [cities, setCities] = useState<CityWeather[]>([]);
    const [idx, setIdx] = useState(() => {
        try { return parseInt(localStorage.getItem(STORAGE_KEY) || '0') || 0; } catch { return 0; }
    });
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const particlesRef = useRef<Particle[]>([]);
    const animRef = useRef<number>(0);
    const typeRef = useRef<WeatherType>('cloudy');

    // Persist city selection
    useEffect(() => { localStorage.setItem(STORAGE_KEY, String(idx)); }, [idx]);

    useEffect(() => {
        async function fetchWeather() {
            try {
                const lats = CITIES.map(c => c.lat).join(',');
                const lons = CITIES.map(c => c.lon).join(',');
                const url = `https://api.open-meteo.com/v1/forecast?latitude=${lats}&longitude=${lons}&current=temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m&timezone=Europe/Zurich`;
                const resp = await fetch(url);
                const data = await resp.json();
                const results: CityWeather[] = (Array.isArray(data) ? data : [data]).map((d: any, i: number) => ({
                    name: CITIES[i].name,
                    temp: d.current?.temperature_2m ?? 0,
                    code: d.current?.weather_code ?? 0,
                    wind: d.current?.wind_speed_10m ?? 0,
                    humidity: d.current?.relative_humidity_2m ?? 0,
                }));
                setCities(results);
            } catch (e) {
                console.error('Weather fetch error:', e);
            }
        }
        fetchWeather();
        const timer = setInterval(fetchWeather, 600_000);
        return () => clearInterval(timer);
    }, []);

    // Canvas animation loop
    const animate = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        const w = canvas.width;
        const h = canvas.height;

        drawParticles(ctx, particlesRef.current, typeRef.current, w, h);
        animRef.current = requestAnimationFrame(animate);
    }, []);

    // Init/reinit particles when city or weather changes
    useEffect(() => {
        const city = cities[idx];
        if (!city) return;
        const meta = weatherMeta(city.code);
        typeRef.current = meta.type;

        const canvas = canvasRef.current;
        if (canvas) {
            const rect = canvas.parentElement?.getBoundingClientRect();
            canvas.width = rect?.width ?? 200;
            canvas.height = rect?.height ?? 150;
            particlesRef.current = createParticles(meta.type, canvas.width, canvas.height);
        }
    }, [cities, idx]);

    // Start/stop animation
    useEffect(() => {
        animRef.current = requestAnimationFrame(animate);
        return () => cancelAnimationFrame(animRef.current);
    }, [animate]);

    // Resize canvas
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ro = new ResizeObserver(() => {
            const rect = canvas.parentElement?.getBoundingClientRect();
            if (rect) {
                canvas.width = rect.width;
                canvas.height = rect.height;
                particlesRef.current = createParticles(typeRef.current, rect.width, rect.height);
            }
        });
        ro.observe(canvas.parentElement!);
        return () => ro.disconnect();
    }, []);

    const city = cities[idx];
    const meta = city ? weatherMeta(city.code) : { icon: '⏳', label: 'Loading', type: 'cloudy' as WeatherType };

    const tempColor = !city ? 'var(--text-2)'
        : city.temp <= 0 ? '#64B5F6'
            : city.temp <= 10 ? '#81D4FA'
                : city.temp <= 20 ? '#FFD740'
                    : city.temp <= 30 ? '#FF9800'
                        : '#FF5252';

    const cycleCity = (dir: number) => {
        if (cities.length === 0) return;
        setIdx(prev => (prev + dir + cities.length) % cities.length);
    };

    return (
        <div className="weather-card" style={{ position: 'relative', overflow: 'hidden', height: '100%' }}>
            {/* Animated canvas background */}
            <canvas
                ref={canvasRef}
                style={{
                    position: 'absolute', inset: 0,
                    width: '100%', height: '100%',
                    pointerEvents: 'none', zIndex: 0,
                }}
            />

            <div style={{ position: 'relative', zIndex: 1, padding: 12 }}>
                <div className="stat-label" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span>🇨🇭 Switzerland</span>
                </div>
                {city ? (
                    <>
                        <div className="stat-value" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <span style={{ fontSize: 20 }}>{meta.icon}</span>
                            <span className="mono" style={{ color: tempColor }}>
                                {city.temp.toFixed(1)}°C
                            </span>
                        </div>
                        <div className="stat-sub" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ fontWeight: 600 }}>{city.name}</span>
                            <span style={{ opacity: 0.6 }}>·</span>
                            <span>{meta.label}</span>
                            <span style={{ opacity: 0.6 }}>·</span>
                            <span>💨 {city.wind.toFixed(0)}km/h</span>
                        </div>
                        {/* City selector */}
                        <div style={{
                            display: 'flex', gap: 4, marginTop: 6,
                            alignItems: 'center', justifyContent: 'center',
                        }}>
                            <button
                                onClick={() => cycleCity(-1)}
                                style={{
                                    all: 'unset', cursor: 'pointer', fontSize: 10,
                                    color: 'var(--text-2)', padding: '0 4px',
                                }}
                                title="Previous city"
                            >◀</button>
                            {cities.map((c, i) => (
                                <button
                                    key={c.name}
                                    onClick={() => setIdx(i)}
                                    style={{
                                        all: 'unset', cursor: 'pointer',
                                        width: i === idx ? 14 : 5, height: 5,
                                        borderRadius: 3,
                                        background: i === idx ? 'var(--accent)' : 'rgba(255,255,255,0.15)',
                                        transition: 'all 0.3s ease',
                                    }}
                                    title={c.name}
                                />
                            ))}
                            <button
                                onClick={() => cycleCity(1)}
                                style={{
                                    all: 'unset', cursor: 'pointer', fontSize: 10,
                                    color: 'var(--text-2)', padding: '0 4px',
                                }}
                                title="Next city"
                            >▶</button>
                        </div>
                    </>
                ) : (
                    <div className="stat-value" style={{ color: 'var(--text-2)', fontSize: 13 }}>Loading...</div>
                )}
            </div>
        </div>
    );
}
