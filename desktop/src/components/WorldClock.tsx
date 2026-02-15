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

    // Other clocks (not selected)
    const others = CLOCKS.map((c, i) => ({ ...c, data: data[i], i })).filter((_, i) => i !== idx);

    return (
        <div style={{ padding: '6px 10px', height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            {/* Label */}
            <div className="stat-label" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 2 }}>
                <span>🕐 World Clock</span>
            </div>

            {d ? (
                <>
                    {/* Featured: flag + time + seconds + market dot — single compact line */}
                    <div className="stat-value" style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
                        <span style={{ fontSize: 13 }}>{city.flag}</span>
                        <span className="mono" style={{ fontSize: 22, fontWeight: 700, color: 'var(--text-0)', letterSpacing: 1 }}>
                            {d.time}
                        </span>
                        <span className="mono" style={{ fontSize: 11, color: 'var(--text-2)', opacity: 0.6 }}>
                            :{d.seconds}
                        </span>
                        <span className={`clock-dot ${d.isOpen ? 'clock-open' : 'clock-closed'}`}
                            title={d.isOpen ? `${city.market} Open` : `${city.market} Closed`}
                            style={{ marginLeft: 2 }} />
                    </div>

                    {/* City + Market — inline with ◀ ▶ arrows */}
                    <div className="stat-sub" style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 2 }}>
                        <button onClick={() => cycleCity(-1)}
                            style={{ all: 'unset', cursor: 'pointer', fontSize: 8, color: 'var(--text-2)', padding: 0, lineHeight: 1 }}
                            title="Previous city">◀</button>
                        <span style={{ fontWeight: 600, fontSize: 10.5 }}>{city.city}</span>
                        <span style={{ opacity: 0.4, fontSize: 10 }}>·</span>
                        <span style={{ color: d.isOpen ? '#34D399' : 'var(--text-2)', fontSize: 10 }}>
                            {city.market} {d.isOpen ? 'Open' : 'Closed'}
                        </span>
                        <button onClick={() => cycleCity(1)}
                            style={{ all: 'unset', cursor: 'pointer', fontSize: 8, color: 'var(--text-2)', padding: 0, lineHeight: 1 }}
                            title="Next city">▶</button>
                    </div>

                    {/* Other clocks — compact row with flags only */}
                    <div className="clock-others" style={{ display: 'flex', gap: 3, flexWrap: 'wrap', marginTop: 'auto' }}>
                        {others.map(o => (
                            <button key={o.city} className="clock-other-btn" onClick={() => setIdx(o.i)}
                                title={`${o.city} — ${o.data?.time || '--:--'}`}
                                style={{
                                    all: 'unset', cursor: 'pointer', display: 'inline-flex',
                                    alignItems: 'center', gap: 2, padding: '1px 5px',
                                    borderRadius: 4, fontSize: 9,
                                    background: 'var(--bg-3)', border: '1px solid var(--border)',
                                }}>
                                <span style={{ fontSize: 10 }}>{o.flag}</span>
                                <span className="mono" style={{ color: 'var(--text-1)', fontSize: 9 }}>
                                    {o.data?.time || '--:--'}
                                </span>
                            </button>
                        ))}
                    </div>
                </>
            ) : (
                <div className="stat-value" style={{ color: 'var(--text-2)', fontSize: 13 }}>Loading...</div>
            )}
        </div>
    );
}
