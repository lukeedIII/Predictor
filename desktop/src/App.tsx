import { useState, useEffect, useRef, Component, type ReactNode, type ErrorInfo } from 'react';
import { HashRouter, Routes, Route, NavLink } from 'react-router-dom';
import { useApi } from './hooks/useApi';
import { useKeyboardShortcuts, SHORTCUTS } from './hooks/useKeyboardShortcuts';
import { WebSocketProvider, useWebSocket } from './hooks/useWebSocket';
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
      <span className="titlebar-title">⚡ NEXUS SHADOW-QUANT</span>
      {api && (
        <div className="titlebar-controls">
          <button className="titlebar-btn minimize" onClick={() => api.minimize()} />
          <button className="titlebar-btn maximize" onClick={() => api.maximize()} />
          <button className="titlebar-btn close" onClick={() => api.close()} />
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
      <div className="sidebar-logo">⚡</div>
      <nav className="sidebar-nav">
        <NavLink to="/" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`} end>
          <span className="nav-icon" data-tooltip="Dashboard (Alt+1)">📊</span>
        </NavLink>
        <NavLink to="/trading" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <span className="nav-icon" data-tooltip="Paper Trading (Alt+2)">💹</span>
        </NavLink>
        <NavLink to="/agent" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <span className="nav-icon" data-tooltip="Nexus Agent (Alt+3)">🤖</span>
        </NavLink>
        <NavLink to="/settings" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <span className="nav-icon" data-tooltip="Settings (Alt+4)">⚙️</span>
        </NavLink>
      </nav>
      <div className="sidebar-status">
        <div className={`status-dot ${statusClass}`} data-tooltip={isReady ? 'System Online' : 'Loading...'} />
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
  const ws = useWebSocket();

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
    if (ws.price !== null && prevPriceRef.current !== null) {
      if (ws.price > prevPriceRef.current) setPriceDirection('up');
      else if (ws.price < prevPriceRef.current) setPriceDirection('down');
    }
    prevPriceRef.current = ws.price;

    // Reset flash after animation
    const timer = setTimeout(() => setPriceDirection('neutral'), 600);
    return () => clearTimeout(timer);
  }, [ws.price]);

  const priceColor = priceDirection === 'up' ? 'var(--positive)'
    : priceDirection === 'down' ? 'var(--negative)'
      : 'var(--text-1)';

  const changePctColor = (ws.change_pct ?? 0) >= 0 ? 'var(--positive)' : 'var(--negative)';

  return (
    <div className="status-bar">
      {/* UTC Clock */}
      <div className="status-bar-item status-bar-clock">
        🕐 <span className="mono">{utcTime}</span> UTC
      </div>

      {/* Live BTC Price */}
      {ws.price !== null && ws.price > 0 && (
        <div className={`status-bar-item status-bar-price ${priceDirection !== 'neutral' ? 'price-flash-' + priceDirection : ''}`}>
          <span className="mono" style={{ color: priceColor, fontWeight: 600 }}>
            ${ws.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
          {ws.change_pct !== null && (
            <span className="mono" style={{ color: changePctColor, fontSize: '0.75rem', marginLeft: 6 }}>
              {ws.change_pct >= 0 ? '+' : ''}{ws.change_pct.toFixed(2)}%
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
        <span style={{ color: ws.connected ? 'var(--positive)' : 'var(--text-4)' }}>●</span>
        WS {ws.connected ? 'Live' : 'Off'}
      </div>

      {/* Binance WS health */}
      <div className="status-bar-item">
        <span style={{ color: ws.ws_connected ? 'var(--positive)' : 'var(--warning)' }}>●</span>
        Feed {ws.ws_connected ? 'Live' : 'Off'}
      </div>

      {status?.device && (
        <div className="status-bar-item">
          🖥️ {status.device.toUpperCase()}
        </div>
      )}
      {status?.model_trained !== undefined && (
        <div className="status-bar-item">
          🧠 {status.model_trained ? 'Model Ready' : 'Untrained'}
        </div>
      )}
      {status?.bot_running !== undefined && (
        <div className="status-bar-item">
          🤖 {status.bot_running ? 'Auto-Trading' : 'Standby'}
        </div>
      )}
      {status?.positions_count !== undefined && status.positions_count > 0 && (
        <div className="status-bar-item">
          📈 {status.positions_count} Open
        </div>
      )}
    </div>
  );
}

// ─── Shortcuts Help Overlay ──────────────────────
function ShortcutsHelp({ show, onClose }: { show: boolean; onClose: () => void }) {
  if (!show) return null;
  return (
    <div className="shortcuts-overlay" onClick={onClose}>
      <div className="shortcuts-modal" onClick={e => e.stopPropagation()}>
        <div className="shortcuts-header">
          <span>⌨️ Keyboard Shortcuts</span>
          <button className="shortcuts-close" onClick={onClose}>✕</button>
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
          <div style={{ fontSize: 32 }}>⚡</div>
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
  return (
    <WebSocketProvider>
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
    </WebSocketProvider>
  );
}
