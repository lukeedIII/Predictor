import { useState, useEffect, useRef, useCallback, Component, type ReactNode, type ErrorInfo } from 'react';
import { HashRouter, Routes, Route, NavLink } from 'react-router-dom';
import { useApi } from './hooks/useApi';
import { useKeyboardShortcuts, SHORTCUTS } from './hooks/useKeyboardShortcuts';
import { connectLiveWS, useLivePrice, useLiveChangePct, useLiveConnected } from './stores/liveStore';
import { Toaster } from './stores/toastStore';
import {
  IconZap, IconDashboard, IconTrending, IconBot, IconSettings,
  IconClock, IconMonitor, IconBrain, IconActivity, IconKeyboard, IconX,
} from './components/Icons';
import Dashboard from './pages/Dashboard';
import PaperTrading from './pages/PaperTrading';
import NexusAgent from './pages/NexusAgent';
import Settings from './pages/Settings';
import FirstRunSetup from './pages/FirstRunSetup';
import './index.css';

// ─── Error Boundary ──────────────────────────────
class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(err: Error) { return { error: err }; }
  componentDidCatch(err: Error, info: ErrorInfo) { console.error('[ErrorBoundary]', err, info); }
  render() {
    if (this.state.error) {
      return (
        <div style={{
          padding: 40, color: 'var(--negative)', fontFamily: 'var(--font-mono)',
          fontSize: 13, display: 'flex', flexDirection: 'column', gap: 12,
        }}>
          <div style={{ fontSize: 24 }}>⚠️ Render Error</div>
          <pre style={{ whiteSpace: 'pre-wrap', color: 'var(--text-3)' }}>
            {this.state.error.message}\n{this.state.error.stack}
          </pre>
          <button onClick={() => this.setState({ error: null })} style={{
            padding: '8px 16px', background: 'var(--accent)', color: '#fff',
            border: 'none', borderRadius: 6, cursor: 'pointer', width: 'fit-content',
          }}>Retry</button>
        </div>
      );
    }
    return this.props.children;
  }
}

// ─── Titlebar ────────────────────────────────────
function Titlebar() {
  const api = window.electronAPI;
  return (
    <div className="titlebar">
      <span className="titlebar-title"><IconZap size={14} style={{ marginRight: 6, verticalAlign: -2 }} /> NEXUS SHADOW-QUANT</span>
      {api && (
        <div className="titlebar-controls">
          <button className="titlebar-btn minimize" onClick={() => api.minimize()} aria-label="Minimize window" />
          <button className="titlebar-btn maximize" onClick={() => api.maximize()} aria-label="Maximize window" />
          <button className="titlebar-btn close" onClick={() => api.close()} aria-label="Close window" />
        </div>
      )}
    </div>
  );
}

// ─── Sidebar ─────────────────────────────────────
function Sidebar() {

  type StatusData = {
    ready: boolean;
    device?: string;
    model_trained?: boolean;
    bot_running?: boolean;
    positions_count?: number;
    version?: string;
  };

  const { data: status } = useApi<StatusData>('/api/status', 3000);

  const isReady = status?.ready ?? false;
  const statusClass = isReady ? 'online' : 'loading';

  return (
    <div className="sidebar">
      <div className="sidebar-logo"><IconZap size={22} /></div>
      <nav className="sidebar-nav">
        <NavLink to="/" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`} end>
          <span className="nav-icon" data-tooltip="Dashboard (Alt+1)" aria-label="Dashboard"><IconDashboard size={20} /></span>
        </NavLink>
        <NavLink to="/trading" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <span className="nav-icon" data-tooltip="Paper Trading (Alt+2)" aria-label="Paper Trading"><IconTrending size={20} /></span>
        </NavLink>
        <NavLink to="/agent" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <span className="nav-icon" data-tooltip="Nexus Agent (Alt+3)" aria-label="Nexus Agent"><IconBot size={20} /></span>
        </NavLink>
        <NavLink to="/settings" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <span className="nav-icon" data-tooltip="Settings (Alt+4)" aria-label="Settings"><IconSettings size={20} /></span>
        </NavLink>
      </nav>
      <div className="sidebar-status">
        <div className={`status-dot ${statusClass}`} data-tooltip={isReady ? 'System Online' : 'Loading...'} aria-label={isReady ? 'System Online' : 'Loading'} />
        <span style={{ fontSize: 9, color: 'var(--text-4)' }}>
          {status?.version || '...'}
        </span>
      </div>
    </div>
  );
}

// ─── Status Bar ──────────────────────────────────
function StatusBar() {
  type StatusData = {
    ready: boolean;
    device?: string;
    model_trained?: boolean;
    bot_running?: boolean;
    positions_count?: number;
  };

  const { data: status } = useApi<StatusData>('/api/status', 5000);
  const { connected: wsConnected, wsConnected: feedConnected } = useLiveConnected();
  const livePrice = useLivePrice();
  const changePct = useLiveChangePct();

  // UTC Clock — ticks every second
  const [utcTime, setUtcTime] = useState('');
  useEffect(() => {
    const tick = () => {
      const now = new Date();
      setUtcTime(now.toISOString().slice(11, 19)); // HH:MM:SS
    };
    tick();
    const interval = setInterval(tick, 1000);
    return () => clearInterval(interval);
  }, []);

  // Price flash animation — track previous price for direction
  const prevPriceRef = useRef<number | null>(null);
  const [priceDirection, setPriceDirection] = useState<'up' | 'down' | 'neutral'>('neutral');

  useEffect(() => {
    if (livePrice !== null && prevPriceRef.current !== null) {
      if (livePrice > prevPriceRef.current) setPriceDirection('up');
      else if (livePrice < prevPriceRef.current) setPriceDirection('down');
    }
    prevPriceRef.current = livePrice;

    // Reset flash after animation
    const timer = setTimeout(() => setPriceDirection('neutral'), 600);
    return () => clearTimeout(timer);
  }, [livePrice]);

  const priceColor = priceDirection === 'up' ? 'var(--positive)'
    : priceDirection === 'down' ? 'var(--negative)'
      : 'var(--text-1)';

  const changePctColor = (changePct ?? 0) >= 0 ? 'var(--positive)' : 'var(--negative)';

  return (
    <div className="status-bar">
      {/* UTC Clock */}
      <div className="status-bar-item status-bar-clock">
        <IconClock size={13} style={{ marginRight: 4, verticalAlign: -2 }} /> <span className="mono">{utcTime}</span> UTC
      </div>

      {/* Live BTC Price */}
      {livePrice !== null && livePrice > 0 && (
        <div className={`status-bar-item status-bar-price ${priceDirection !== 'neutral' ? 'price-flash-' + priceDirection : ''}`}>
          <span className="mono" style={{ color: priceColor, fontWeight: 600 }}>
            ${livePrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
          {changePct !== null && (
            <span className="mono" style={{ color: changePctColor, fontSize: '0.75rem', marginLeft: 6 }}>
              {changePct >= 0 ? '+' : ''}{changePct.toFixed(2)}%
            </span>
          )}
        </div>
      )}

      {/* Backend API status */}
      <div className="status-bar-item">
        <span style={{ color: status?.ready ? 'var(--positive)' : 'var(--warning)' }}>●</span>
        {status?.ready ? 'API' : 'Init...'}
      </div>

      {/* Our internal WS */}
      <div className="status-bar-item">
        <span style={{ color: wsConnected ? 'var(--positive)' : 'var(--text-4)' }}>●</span>
        WS {wsConnected ? 'Live' : 'Off'}
      </div>

      {/* Binance WS health */}
      <div className="status-bar-item">
        <span style={{ color: feedConnected ? 'var(--positive)' : 'var(--warning)' }}>●</span>
        Feed {feedConnected ? 'Live' : 'Off'}
      </div>

      {status?.device && (
        <div className="status-bar-item">
          <IconMonitor size={13} style={{ marginRight: 4, verticalAlign: -2 }} /> {status.device.toUpperCase()}
        </div>
      )}
      {status?.model_trained !== undefined && (
        <div className="status-bar-item">
          <IconBrain size={13} style={{ marginRight: 4, verticalAlign: -2 }} /> {status.model_trained ? 'Model Ready' : 'Untrained'}
        </div>
      )}
      {status?.bot_running !== undefined && (
        <div className="status-bar-item">
          <IconBot size={13} style={{ marginRight: 4, verticalAlign: -2 }} /> {status.bot_running ? 'Auto-Trading' : 'Standby'}
        </div>
      )}
      {status?.positions_count !== undefined && status.positions_count > 0 && (
        <div className="status-bar-item">
          <IconActivity size={13} style={{ marginRight: 4, verticalAlign: -2 }} /> {status.positions_count} Open
        </div>
      )}
    </div>
  );
}

// ─── Shortcuts Help Overlay ──────────────────────
function ShortcutsHelp({ show, onClose }: { show: boolean; onClose: () => void }) {
  const modalRef = useRef<HTMLDivElement>(null);
  const previousFocus = useRef<HTMLElement | null>(null);

  // Focus trap + ESC handler
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      onClose();
      return;
    }
    if (e.key === 'Tab' && modalRef.current) {
      const focusable = modalRef.current.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey) {
        if (document.activeElement === first) { e.preventDefault(); last.focus(); }
      } else {
        if (document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
    }
  }, [onClose]);

  useEffect(() => {
    if (show) {
      previousFocus.current = document.activeElement as HTMLElement;
      document.addEventListener('keydown', handleKeyDown);
      // Focus the close button after a tick so the modal is rendered
      requestAnimationFrame(() => {
        const btn = modalRef.current?.querySelector<HTMLElement>('.shortcuts-close');
        btn?.focus();
      });
    }
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      if (!show && previousFocus.current) {
        previousFocus.current.focus();
        previousFocus.current = null;
      }
    };
  }, [show, handleKeyDown]);

  if (!show) return null;
  return (
    <div className="shortcuts-overlay" onClick={onClose} role="presentation">
      <div
        ref={modalRef}
        className="shortcuts-modal"
        onClick={e => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Keyboard shortcuts"
      >
        <div className="shortcuts-header">
          <span><IconKeyboard size={16} style={{ marginRight: 6, verticalAlign: -3 }} /> Keyboard Shortcuts</span>
          <button className="shortcuts-close" onClick={onClose} aria-label="Close shortcuts panel"><IconX size={16} /></button>
        </div>
        <div className="shortcuts-list">
          {SHORTCUTS.map(s => (
            <div key={s.keys} className="shortcut-row">
              <kbd className="shortcut-key">{s.keys}</kbd>
              <span className="shortcut-label">{s.label}</span>
            </div>
          ))}
        </div>
        <div className="shortcuts-hint">Press Alt + / to toggle this panel</div>
      </div>
    </div>
  );
}

import { API_BASE } from './hooks/useApi';

// ─── App ─────────────────────────────────────────
export default function App() {
  const [needsSetup, setNeedsSetup] = useState<boolean | null>(null);

  useEffect(() => {
    let attempts = 0;
    const MAX_RETRIES = 30; // 60s total
    const checkFirstRun = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/first-run-status`);
        const data = await res.json();
        setNeedsSetup(data.needs_setup);
      } catch {
        attempts++;
        if (attempts < MAX_RETRIES) {
          setTimeout(checkFirstRun, 2000);
        } else {
          // Give up — show the main app (backend may be unreachable)
          setNeedsSetup(false);
        }
      }
    };
    checkFirstRun();
  }, []);

  // Still checking...
  if (needsSetup === null) {
    return (
      <>
        <Titlebar />
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          height: '100vh', color: 'var(--text-3)', fontFamily: 'var(--font-mono)',
          fontSize: 14, flexDirection: 'column', gap: 12,
        }}>
          <IconZap size={32} style={{ color: 'var(--accent)' }} />
          <div>Connecting to backend...</div>
        </div>
      </>
    );
  }

  // First-run setup needed
  if (needsSetup) {
    return (
      <HashRouter>
        <Titlebar />
        <div className="app-layout">
          <Sidebar />
          <main className="main-content">
            <FirstRunSetup onComplete={() => setNeedsSetup(false)} />
          </main>
        </div>
        <StatusBar />
      </HashRouter>
    );
  }

  // Normal app
  return (
    <HashRouter>
      <AppShell />
    </HashRouter>
  );
}

// Inner shell — needs to be inside HashRouter so useNavigate works
function AppShell() {
  const { showHelp, setShowHelp } = useKeyboardShortcuts();

  // Start WebSocket connection on mount (replaces old WebSocketProvider)
  useEffect(() => {
    const disconnect = connectLiveWS();
    return disconnect;
  }, []);

  return (
    <>
      <Titlebar />
      <div className="app-layout">
        <Sidebar />
        <main className="main-content">
          <ErrorBoundary>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/trading" element={<PaperTrading />} />
              <Route path="/agent" element={<NexusAgent />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </ErrorBoundary>
        </main>
      </div>
      <StatusBar />
      <ShortcutsHelp show={showHelp} onClose={() => setShowHelp(false)} />
      <Toaster />
    </>
  );
}
