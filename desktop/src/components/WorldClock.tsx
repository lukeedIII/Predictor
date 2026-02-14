import { useState, useEffect } from 'react';

// ── Financial Hubs with Timezone ─────────────────
const CLOCKS = [
    { city: 'New York', tz: 'America/New_York', flag: '🇺🇸', market: 'NYSE' },
    { city: 'London', tz: 'Europe/London', flag: '🇬🇧', market: 'LSE' },
    { city: 'Zürich', tz: 'Europe/Zurich', flag: '🇨🇭', market: 'SIX' },
    { city: 'Moscow', tz: 'Europe/Moscow', flag: '🇷🇺', market: 'MOEX' },
    { city: 'Tokyo', tz: 'Asia/Tokyo', flag: '🇯🇵', market: 'TSE' },
    { city: 'Shanghai', tz: 'Asia/Shanghai', flag: '🇨🇳', market: 'SSE' },
];

function getTime(tz: string): { time: string; isOpen: boolean } {
    const now = new Date();
    const fmt = new Intl.DateTimeFormat('en-GB', {
        timeZone: tz,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
    });
    const time = fmt.format(now);

    // Check if within typical market hours (9:30–16:00 local)
    const parts = new Intl.DateTimeFormat('en-US', {
        timeZone: tz,
        hour: 'numeric',
        minute: 'numeric',
        weekday: 'short',
        hour12: false,
    }).formatToParts(now);

    const hour = parseInt(parts.find(p => p.type === 'hour')?.value || '0');
    const minute = parseInt(parts.find(p => p.type === 'minute')?.value || '0');
    const day = parts.find(p => p.type === 'weekday')?.value || '';
    const totalMin = hour * 60 + minute;
    const isWeekday = !['Sat', 'Sun'].includes(day);
    const isOpen = isWeekday && totalMin >= 570 && totalMin <= 960; // 9:30–16:00

    return { time, isOpen };
}

export default function WorldClock() {
    const [times, setTimes] = useState<{ time: string; isOpen: boolean }[]>([]);

    useEffect(() => {
        function tick() {
            setTimes(CLOCKS.map(c => getTime(c.tz)));
        }
        tick();
        const id = setInterval(tick, 1000);
        return () => clearInterval(id);
    }, []);

    return (
        <div style={{ padding: 8, height: '100%', display: 'flex', flexDirection: 'column' }}>
            <div className="world-clock-grid">
                {CLOCKS.map((c, i) => {
                    const t = times[i];
                    return (
                        <div key={c.city} className="clock-row">
                            <div className="clock-city">
                                <span className="clock-flag">{c.flag}</span>
                                <span className="clock-name">{c.city}</span>
                            </div>
                            <div className="clock-right">
                                <span className={`clock-dot ${t?.isOpen ? 'clock-open' : 'clock-closed'}`}
                                    title={t?.isOpen ? `${c.market} Open` : `${c.market} Closed`} />
                                <span className="clock-time mono">{t?.time || '--:--:--'}</span>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
