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
    const count = type === 'rain' || type === 'storm' ? 60
        : type === 'drizzle' ? 30
            : type === 'snow' ? 40
                : type === 'clear' ? 8
                    : type === 'fog' ? 12
                        : type === 'cloudy' ? 6
                            : 0;

    for (let i = 0; i < count; i++) {
        particles.push({
            x: Math.random() * w,
            y: Math.random() * h,
            vx: type === 'rain' || type === 'storm' ? -0.5 + Math.random() * -0.5 : (Math.random() - 0.5) * 0.3,
            vy: type === 'rain' ? 3 + Math.random() * 4
                : type === 'storm' ? 5 + Math.random() * 5
                    : type === 'drizzle' ? 1.5 + Math.random() * 2
                        : type === 'snow' ? 0.3 + Math.random() * 0.8
                            : type === 'clear' ? 0.1 + Math.random() * 0.2
                                : 0.05 + Math.random() * 0.1,
            size: type === 'snow' ? 2 + Math.random() * 3
                : type === 'rain' || type === 'storm' ? 1
                    : type === 'clear' ? 1 + Math.random() * 2
                        : 4 + Math.random() * 8,
            opacity: type === 'fog' ? 0.03 + Math.random() * 0.06
                : type === 'cloudy' ? 0.04 + Math.random() * 0.06
                    : 0.3 + Math.random() * 0.5,
            life: Math.random(),
        });
    }
    return particles;
}

function drawParticles(ctx: CanvasRenderingContext2D, particles: Particle[], type: WeatherType, w: number, h: number) {
    ctx.clearRect(0, 0, w, h);

    // Background gradient based on weather
    const grad = ctx.createLinearGradient(0, 0, w, h);
    if (type === 'clear') {
        grad.addColorStop(0, 'rgba(255, 200, 50, 0.06)');
        grad.addColorStop(1, 'rgba(255, 140, 0, 0.03)');
    } else if (type === 'rain' || type === 'drizzle') {
        grad.addColorStop(0, 'rgba(59, 130, 246, 0.06)');
        grad.addColorStop(1, 'rgba(30, 64, 175, 0.04)');
    } else if (type === 'storm') {
        grad.addColorStop(0, 'rgba(88, 28, 135, 0.08)');
        grad.addColorStop(1, 'rgba(30, 58, 138, 0.06)');
    } else if (type === 'snow') {
        grad.addColorStop(0, 'rgba(186, 230, 253, 0.06)');
        grad.addColorStop(1, 'rgba(147, 197, 253, 0.04)');
    } else if (type === 'fog') {
        grad.addColorStop(0, 'rgba(148, 163, 184, 0.06)');
        grad.addColorStop(1, 'rgba(100, 116, 139, 0.04)');
    } else {
        grad.addColorStop(0, 'rgba(148, 163, 184, 0.04)');
        grad.addColorStop(1, 'rgba(71, 85, 105, 0.03)');
    }
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    for (const p of particles) {
        ctx.globalAlpha = p.opacity;

        if (type === 'rain' || type === 'drizzle') {
            // Rain drops — thin diagonal lines
            ctx.strokeStyle = '#60A5FA';
            ctx.lineWidth = type === 'drizzle' ? 0.5 : 1;
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p.x + p.vx * 3, p.y + p.vy * 2.5);
            ctx.stroke();
        } else if (type === 'storm') {
            // Storm — thick rain + occasional "flash"
            ctx.strokeStyle = '#818CF8';
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p.x + p.vx * 4, p.y + p.vy * 3);
            ctx.stroke();

            // Lightning flash (rare)
            if (Math.random() < 0.001) {
                ctx.globalAlpha = 0.15;
                ctx.fillStyle = '#E0E7FF';
                ctx.fillRect(0, 0, w, h);
            }
        } else if (type === 'snow') {
            // Snow — soft circles
            ctx.fillStyle = '#E0F2FE';
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fill();
        } else if (type === 'clear') {
            // Sun — warm floating dots
            ctx.fillStyle = `rgba(255, 215, 0, ${p.opacity * 0.6})`;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fill();
            // Glow
            ctx.fillStyle = `rgba(255, 200, 50, ${p.opacity * 0.15})`;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size * 4, 0, Math.PI * 2);
            ctx.fill();
        } else if (type === 'fog') {
            // Fog — large blurry blobs
            ctx.fillStyle = `rgba(148, 163, 184, ${p.opacity})`;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fill();
        } else {
            // Cloudy — soft gray blobs
            ctx.fillStyle = `rgba(100, 116, 139, ${p.opacity})`;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fill();
        }

        // Update position
        p.x += p.vx;
        p.y += p.vy;

        // Snow sway
        if (type === 'snow') {
            p.x += Math.sin(p.life * 6 + Date.now() * 0.001) * 0.3;
        }

        // Wrap around
        if (p.y > h + 10) { p.y = -10; p.x = Math.random() * w; }
        if (p.y < -10) { p.y = h + 10; p.x = Math.random() * w; }
        if (p.x > w + 10) p.x = -10;
        if (p.x < -10) p.x = w + 10;
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
