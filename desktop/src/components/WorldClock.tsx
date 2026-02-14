import { useState, useEffect } from 'react';

// ── Financial Hubs ───────────────────────────────
const CLOCKS = [
    { city: 'New York', tz: 'America/New_York', flag: '🇺🇸', market: 'NYSE' },
    { city: 'London', tz: 'Europe/London', flag: '🇬🇧', market: 'LSE' },
    { city: 'Zürich', tz: 'Europe/Zurich', flag: '🇨🇭', market: 'SIX' },
    { city: 'Moscow', tz: 'Europe/Moscow', flag: '🇷🇺', market: 'MOEX' },
    { city: 'Tokyo', tz: 'Asia/Tokyo', flag: '🇯🇵', market: 'TSE' },
    { city: 'Shanghai', tz: 'Asia/Shanghai', flag: '🇨🇳', market: 'SSE' },
];

const STORAGE_KEY = 'nexus-clock-city-idx';

type ClockData = { time: string; seconds: string; isOpen: boolean };

function getClockData(tz: string): ClockData {
    const now = new Date();
    const fmt = new Intl.DateTimeFormat('en-GB', {
        timeZone: tz, hour: '2-digit', minute: '2-digit', hour12: false,
    });
    const secFmt = new Intl.DateTimeFormat('en-GB', {
        timeZone: tz, second: '2-digit',
    });
    const time = fmt.format(now);
    const seconds = secFmt.format(now);

    const parts = new Intl.DateTimeFormat('en-US', {
        timeZone: tz, hour: 'numeric', minute: 'numeric', weekday: 'short', hour12: false,
    }).formatToParts(now);

    const hour = parseInt(parts.find(p => p.type === 'hour')?.value || '0');
    const minute = parseInt(parts.find(p => p.type === 'minute')?.value || '0');
    const day = parts.find(p => p.type === 'weekday')?.value || '';
    const totalMin = hour * 60 + minute;
    const isWeekday = !['Sat', 'Sun'].includes(day);
    const isOpen = isWeekday && totalMin >= 570 && totalMin <= 960;

    return { time, seconds, isOpen };
}

export default function WorldClock() {
    const [idx, setIdx] = useState(() => {
        try { return parseInt(localStorage.getItem(STORAGE_KEY) || '0') || 0; } catch { return 0; }
    });
    const [data, setData] = useState<ClockData[]>([]);

    useEffect(() => { localStorage.setItem(STORAGE_KEY, String(idx)); }, [idx]);

    useEffect(() => {
        function tick() { setData(CLOCKS.map(c => getClockData(c.tz))); }
        tick();
        const id = setInterval(tick, 1000);
        return () => clearInterval(id);
    }, []);

    const city = CLOCKS[idx];
    const d = data[idx];
    const cycleCity = (dir: number) => {
        setIdx(prev => (prev + dir + CLOCKS.length) % CLOCKS.length);
    };

    // Other clocks (not the selected one)
    const others = CLOCKS.map((c, i) => ({ ...c, data: data[i], i })).filter((_, i) => i !== idx);

    return (
        <div style={{ padding: '8px 10px', height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Featured clock */}
            <div className="stat-label" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span>🕐 World Clock</span>
            </div>
            {d ? (
                <>
                    <div className="stat-value" style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
                        <span style={{ fontSize: 14 }}>{city.flag}</span>
                        <span className="mono" style={{ fontSize: 22, fontWeight: 700, color: 'var(--text-0)', letterSpacing: 1 }}>
                            {d.time}
                        </span>
                        <span className="mono" style={{ fontSize: 12, color: 'var(--text-2)', opacity: 0.6 }}>
                            :{d.seconds}
                        </span>
                        <span className={`clock-dot ${d.isOpen ? 'clock-open' : 'clock-closed'}`}
                            title={d.isOpen ? `${city.market} Open` : `${city.market} Closed`}
                            style={{ marginLeft: 4 }} />
                    </div>
                    <div className="stat-sub" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span style={{ fontWeight: 600 }}>{city.city}</span>
                        <span style={{ opacity: 0.5 }}>·</span>
                        <span style={{ color: d.isOpen ? '#34D399' : 'var(--text-2)' }}>
                            {city.market} {d.isOpen ? 'Open' : 'Closed'}
                        </span>
                    </div>

                    {/* Other clocks ticker */}
                    <div className="clock-others">
                        {others.map(o => (
                            <button key={o.city} className="clock-other-btn" onClick={() => setIdx(o.i)}
                                title={`Switch to ${o.city}`}>
                                <span className="clock-flag">{o.flag}</span>
                                <span className="clock-time mono">{o.data?.time || '--:--'}</span>
                            </button>
                        ))}
                    </div>

                    {/* Dot selector */}
                    <div style={{
                        display: 'flex', gap: 4, marginTop: 4,
                        alignItems: 'center', justifyContent: 'center',
                    }}>
                        <button onClick={() => cycleCity(-1)}
                            style={{ all: 'unset', cursor: 'pointer', fontSize: 10, color: 'var(--text-2)', padding: '0 4px' }}
                            title="Previous city">◀</button>
                        {CLOCKS.map((_, i) => (
                            <button key={i} onClick={() => setIdx(i)}
                                style={{
                                    all: 'unset', cursor: 'pointer',
                                    width: i === idx ? 14 : 5, height: 5, borderRadius: 3,
                                    background: i === idx ? 'var(--accent)' : 'rgba(255,255,255,0.15)',
                                    transition: 'all 0.3s ease',
                                }}
                                title={CLOCKS[i].city} />
                        ))}
                        <button onClick={() => cycleCity(1)}
                            style={{ all: 'unset', cursor: 'pointer', fontSize: 10, color: 'var(--text-2)', padding: '0 4px' }}
                            title="Next city">▶</button>
                    </div>
                </>
            ) : (
                <div className="stat-value" style={{ color: 'var(--text-2)', fontSize: 13 }}>Loading...</div>
            )}
        </div>
    );
}
