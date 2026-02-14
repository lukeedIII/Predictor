import { useState, useEffect } from 'react';

// ─── Swiss Cities ────────────────────────────────
const CITIES = [
    { name: 'Zürich', lat: 47.3769, lon: 8.5417 },
    { name: 'Bern', lat: 46.9481, lon: 7.4474 },
    { name: 'Geneva', lat: 46.2044, lon: 6.1432 },
    { name: 'Lucerne', lat: 47.0502, lon: 8.3093 },
    { name: 'Basel', lat: 47.5596, lon: 7.5886 },
];

// ─── Weather codes → icon + label ────────────────
function weatherMeta(code: number): { icon: string; label: string } {
    if (code === 0) return { icon: '☀️', label: 'Clear' };
    if (code <= 3) return { icon: '⛅', label: 'Cloudy' };
    if (code <= 48) return { icon: '🌫️', label: 'Fog' };
    if (code <= 55) return { icon: '🌦️', label: 'Drizzle' };
    if (code <= 65) return { icon: '🌧️', label: 'Rain' };
    if (code <= 67) return { icon: '🌧️', label: 'Freezing' };
    if (code <= 75) return { icon: '🌨️', label: 'Snow' };
    if (code <= 77) return { icon: '❄️', label: 'Grains' };
    if (code <= 82) return { icon: '⛈️', label: 'Showers' };
    if (code <= 86) return { icon: '🌨️', label: 'Snow' };
    if (code <= 99) return { icon: '⛈️', label: 'Storm' };
    return { icon: '🌡️', label: '—' };
}

type CityWeather = {
    name: string;
    temp: number;
    code: number;
    wind: number;
    humidity: number;
};

export default function SwissWeather() {
    const [cities, setCities] = useState<CityWeather[]>([]);
    const [idx, setIdx] = useState(0);

    useEffect(() => {
        async function fetchWeather() {
            try {
                const lats = CITIES.map(c => c.lat).join(',');
                const lons = CITIES.map(c => c.lon).join(',');
                const url = `https://api.open-meteo.com/v1/forecast?latitude=${lats}&longitude=${lons}&current=temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m&timezone=Europe/Zurich`;
                const resp = await fetch(url);
                const data = await resp.json();

                // Open-Meteo returns array for multi-location
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
        const timer = setInterval(fetchWeather, 600_000); // refresh every 10 min
        return () => clearInterval(timer);
    }, []);

    // Auto-rotate between cities every 4 seconds
    useEffect(() => {
        if (cities.length === 0) return;
        const timer = setInterval(() => {
            setIdx(prev => (prev + 1) % cities.length);
        }, 4000);
        return () => clearInterval(timer);
    }, [cities.length]);

    const city = cities[idx];
    const meta = city ? weatherMeta(city.code) : { icon: '⏳', label: 'Loading' };

    // Temperature color
    const tempColor = !city ? 'var(--text-2)'
        : city.temp <= 0 ? '#64B5F6'
            : city.temp <= 10 ? '#81D4FA'
                : city.temp <= 20 ? '#FFD740'
                    : city.temp <= 30 ? '#FF9800'
                        : '#FF5252';

    return (
        <div className="card card-compact animate-in" style={{ position: 'relative', overflow: 'hidden' }}>
            {/* Background animation */}
            <div className="weather-bg" data-code={city?.code ?? 0} />

            <div style={{ position: 'relative', zIndex: 1 }}>
                <div className="stat-label" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span>🇨🇭 Switzerland</span>
                    <span style={{ fontSize: 8, opacity: 0.5, fontWeight: 400 }}>
                        {cities.length > 0 ? `${idx + 1}/${cities.length}` : ''}
                    </span>
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
                        {/* City dots */}
                        <div style={{
                            display: 'flex',
                            gap: 4,
                            marginTop: 6,
                            justifyContent: 'center',
                        }}>
                            {cities.map((c, i) => (
                                <button
                                    key={c.name}
                                    onClick={() => setIdx(i)}
                                    style={{
                                        all: 'unset',
                                        cursor: 'pointer',
                                        width: i === idx ? 14 : 5,
                                        height: 5,
                                        borderRadius: 3,
                                        background: i === idx ? 'var(--accent)' : 'rgba(255,255,255,0.15)',
                                        transition: 'all 0.3s ease',
                                    }}
                                    title={c.name}
                                />
                            ))}
                        </div>
                    </>
                ) : (
                    <div className="stat-value" style={{ color: 'var(--text-2)', fontSize: 13 }}>Loading...</div>
                )}
            </div>
        </div>
    );
}
