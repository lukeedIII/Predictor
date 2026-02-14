import { useEffect, useState } from 'react';
import { HashRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom';
import { connectLiveWS, useLivePrice, useLiveConnected, useLiveChangePct } from './stores/liveStore';
import { Toaster } from './stores/toastStore';
import { IconDashboard, IconChart, IconBot, IconSettings, IconMinimize, IconMaximize, IconClose, IconNexus, IconKeyboard, IconGpu } from './components/Icons';

/* Lazy pages */
import Dashboard from './pages/Dashboard';
import PaperTrading from './pages/PaperTrading';
import NexusAgent from './pages/NexusAgent';
import Settings from './pages/Settings';
import GpuFarm from './pages/GpuFarm';
import FirstRunSetup from './pages/FirstRunSetup';

/* ─── Titlebar ─────────────────────────────────────── */
function Titlebar() {
    const api = (window as any).electronAPI;
    return (
        <header className="titlebar">
            <div className="titlebar-brand">
                <IconNexus style={{ width: 16, height: 16 }} />
                <span>Nexus Shadow-Quant</span>
            </div>
            <div className="titlebar-controls">
                <button className="titlebar-btn" onClick={() => api?.minimize()} aria-label="Minimize">
                    <IconMinimize style={{ width: 14, height: 14 }} />
                </button>
                <button className="titlebar-btn" onClick={() => api?.maximize()} aria-label="Maximize">
                    <IconMaximize style={{ width: 14, height: 14 }} />
                </button>
                <button className="titlebar-btn close" onClick={() => api?.close()} aria-label="Close">
                    <IconClose style={{ width: 14, height: 14 }} />
                </button>
            </div>
        </header>
    );
}

/* ─── Sidebar ──────────────────────────────────────── */
const NAV = [
    { to: '/', icon: IconDashboard, label: 'Dashboard' },
    { to: '/trading', icon: IconChart, label: 'Paper Trading' },
    { to: '/gpu-farm', icon: IconGpu, label: 'GPU Farm' },
    { to: '/agent', icon: IconBot, label: 'Nexus Agent' },
    { to: '/settings', icon: IconSettings, label: 'Settings' },
] as const;

function Sidebar({ onToggleShortcuts }: { onToggleShortcuts: () => void }) {
    return (
        <nav className="sidebar">
            <div className="sidebar-top">
                {NAV.map(({ to, icon: Icon, label }) => (
                    <NavLink
                        key={to}
                        to={to}
                        end={to === '/'}
                        className={({ isActive }) => `nav-btn${isActive ? ' active' : ''}`}
                        data-tooltip={label}
                    >
                        <Icon />
                    </NavLink>
                ))}
            </div>
            <div className="sidebar-bottom">
                <button className="nav-btn" data-tooltip="Shortcuts" onClick={onToggleShortcuts}>
                    <IconKeyboard />
                </button>
            </div>
        </nav>
    );
}

/* ─── Status Bar ───────────────────────────────────── */
function StatusBar() {
    const price = useLivePrice();
    const changePct = useLiveChangePct();
    const { connected, wsConnected } = useLiveConnected();
    const [clock, setClock] = useState('');

    useEffect(() => {
        const tick = () => setClock(new Date().toLocaleTimeString('en-US', { hour12: false }));
        tick();
        const id = setInterval(tick, 1000);
        return () => clearInterval(id);
    }, []);

    const pctClass = (changePct ?? 0) >= 0 ? 'text-positive' : 'text-negative';

    return (
        <footer className="statusbar">
            <span className="flex items-center gap-4">
                <span className={`status-dot ${connected ? 'online' : 'offline'}`} />
                {connected ? 'Connected' : 'Disconnected'}
            </span>
            {wsConnected && <span className="flex items-center gap-4"><span className="status-dot online" />Binance</span>}
            <div className="statusbar-right">
                {price != null && (
                    <span className="mono">
                        BTC ${price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        {changePct != null && (
                            <span className={pctClass} style={{ marginLeft: 6 }}>
                                {changePct >= 0 ? '+' : ''}{changePct.toFixed(2)}%
                            </span>
                        )}
                    </span>
                )}
                <span className="mono">{clock}</span>
            </div>
        </footer>
    );
}

/* ─── Shortcuts Overlay ────────────────────────────── */
function ShortcutsOverlay({ show, onClose }: { show: boolean; onClose: () => void }) {
    if (!show) return null;
    const shortcuts = [
        ['Navigate Dashboard', '1'],
        ['Navigate Paper Trading', '2'],
        ['Navigate Nexus Agent', '3'],
        ['Navigate Settings', '4'],
        ['Show Shortcuts', '?'],
    ];
    return (
        <div className="overlay-backdrop" onClick={onClose}>
            <div className="overlay-panel" onClick={e => e.stopPropagation()}>
                <h2>Keyboard Shortcuts</h2>
                {shortcuts.map(([label, key]) => (
                    <div className="shortcut-row" key={key}>
                        <span>{label}</span>
                        <kbd>{key}</kbd>
                    </div>
                ))}
            </div>
        </div>
    );
}

/* ─── Route Content ────────────────────────────────── */
function PageContent() {
    const location = useLocation();
    return (
        <main className="page" key={location.pathname}>
            <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/trading" element={<PaperTrading />} />
                <Route path="/gpu-farm" element={<GpuFarm />} />
                <Route path="/agent" element={<NexusAgent />} />
                <Route path="/settings" element={<Settings />} />
            </Routes>
        </main>
    );
}

/* ─── Inner Shell (must be inside Router for hooks) ─── */
function AppShell({ showShortcuts, setShowShortcuts }: { showShortcuts: boolean; setShowShortcuts: React.Dispatch<React.SetStateAction<boolean>> }) {
    return (
        <>
            <div className="app-layout">
                <Titlebar />
                <Sidebar onToggleShortcuts={() => setShowShortcuts(s => !s)} />
                <PageContent />
                <StatusBar />
            </div>
            <ShortcutsOverlay show={showShortcuts} onClose={() => setShowShortcuts(false)} />
            <Toaster />
        </>
    );
}

/* ─── App ──────────────────────────────────────────── */
export default function App() {
    const [showShortcuts, setShowShortcuts] = useState(false);
    const [firstRunDone, setFirstRunDone] = useState<boolean | null>(null);

    // Connect WS once at mount
    useEffect(() => {
        const cleanup = connectLiveWS();
        return cleanup;
    }, []);

    // Check first-run
    useEffect(() => {
        (async () => {
            try {
                const res = await fetch('http://127.0.0.1:8420/api/settings');
                if (res.ok) {
                    const data = await res.json();
                    setFirstRunDone(data?.first_run_done === true);
                } else {
                    setFirstRunDone(true); // assume done if API is down
                }
            } catch {
                setFirstRunDone(true);
            }
        })();
    }, []);

    // Still loading first-run check
    if (firstRunDone === null) return null;

    // Show first-run wizard
    if (!firstRunDone) {
        return <FirstRunSetup onComplete={() => setFirstRunDone(true)} />;
    }

    return (
        <HashRouter>
            <AppShell showShortcuts={showShortcuts} setShowShortcuts={setShowShortcuts} />
        </HashRouter>
    );
}
