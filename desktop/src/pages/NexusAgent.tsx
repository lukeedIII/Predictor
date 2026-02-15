import { useState, useRef, useEffect, useCallback, type FormEvent } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { IconSend, IconPlus, IconTrash, IconChatBubble, IconChevronLeft, IconTrend, IconTrendDown } from '../components/Icons';
import { useLivePrice, useLiveChangePct, useLivePrediction, useLivePositions } from '../stores/liveStore';

const API = 'http://127.0.0.1:8420';

/* ── Types ─────────────────────────────────────────── */
type Message = { role: 'user' | 'agent'; content: string; provider?: string };
type Session = {
    session_id: string;
    title: string;
    message_count: number;
    started_at: string;
    last_message_at: string;
};

/* ── Helpers ───────────────────────────────────────── */
function relativeTime(iso: string): string {
    if (!iso) return '';
    const d = new Date(iso + 'Z');
    const diff = (Date.now() - d.getTime()) / 1000;
    if (diff < 60) return 'just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
}

/* ── Market Ticker Component ──────────────────────── */
function MarketTicker() {
    const price = useLivePrice();
    const changePct = useLiveChangePct();
    const prediction = useLivePrediction();
    const positions = useLivePositions();

    const pctStr = changePct != null ? `${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%` : '--';
    const pctClass = (changePct ?? 0) >= 0 ? 'up' : 'down';
    const predDir = (prediction as any)?.direction ?? null;
    const predConf = (prediction as any)?.confidence ?? null;
    const openPos = positions?.length ?? 0;

    return (
        <div className="market-ticker">
            <div className="ticker-item">
                <span className="ticker-label">BTC/USDT</span>
                <span className="ticker-value" style={{ fontVariantNumeric: 'tabular-nums' }}>
                    ${price != null ? price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '—'}
                </span>
            </div>
            <div className="ticker-divider" />
            <div className="ticker-item">
                <span className="ticker-label">24h</span>
                <span className={`ticker-value ${pctClass}`}>{pctStr}</span>
            </div>
            <div className="ticker-divider" />
            <div className="ticker-item">
                <span className="ticker-label">Signal</span>
                {predDir ? (
                    <span className={`ticker-value ${predDir === 'LONG' ? 'up' : 'down'}`} style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                        {predDir === 'LONG' ? <IconTrend size={13} /> : <IconTrendDown size={13} />}
                        {predDir} {predConf != null ? `${Number(predConf).toFixed(2)}%` : ''}
                    </span>
                ) : <span className="ticker-value muted">—</span>}
            </div>
            <div className="ticker-divider" />
            <div className="ticker-item">
                <span className="ticker-label">Positions</span>
                <span className={`ticker-value ${openPos > 0 ? 'up' : 'muted'}`}>{openPos}</span>
            </div>
        </div>
    );
}

/* ── Main Component ───────────────────────────────── */
export default function NexusAgent() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [streaming, setStreaming] = useState(false);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [sessions, setSessions] = useState<Session[]>([]);
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLTextAreaElement>(null);

    const scrollToBottom = useCallback(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, []);
    useEffect(scrollToBottom, [messages, scrollToBottom]);

    /* ── Load sessions list ─────────────────────── */
    const loadSessions = useCallback(async () => {
        try {
            const res = await fetch(`${API}/api/agent/sessions`);
            if (!res.ok) return;
            const data = await res.json();
            setSessions(data.sessions ?? []);
            if (!sessionId && data.current_session_id) {
                setSessionId(data.current_session_id);
            }
        } catch { /* backend not ready */ }
    }, [sessionId]);

    /* ── Load messages for a given session ───────── */
    const loadHistory = useCallback(async (sid: string | null) => {
        try {
            const url = sid
                ? `${API}/api/agent/history?session_id=${sid}&limit=50`
                : `${API}/api/agent/history?limit=50`;
            const res = await fetch(url);
            if (!res.ok) return;
            const data = await res.json();
            if (data.messages && data.messages.length > 0) {
                const history: Message[] = data.messages.map((m: any) => ({
                    role: m.role === 'user' ? 'user' as const : 'agent' as const,
                    content: m.content ?? m.text ?? '',
                }));
                setMessages(history);
            } else {
                setMessages([]);
            }
        } catch { /* backend not ready */ }
    }, []);

    /* ── Initial mount ──────────────────────────── */
    useEffect(() => {
        (async () => {
            // Fetch sessions and current session
            const res = await fetch(`${API}/api/agent/sessions`).catch(() => null);
            if (!res || !res.ok) return;
            const data = await res.json();
            setSessions(data.sessions ?? []);
            const sid = data.current_session_id;
            if (sid) {
                setSessionId(sid);
                loadHistory(sid);
            }
        })();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    /* ── New Chat ────────────────────────────────── */
    const startNewChat = useCallback(async () => {
        try {
            const res = await fetch(`${API}/api/agent/new-session`, { method: 'POST' });
            if (!res.ok) return;
            const data = await res.json();
            setSessionId(data.session_id);
            setMessages([]);
            inputRef.current?.focus();
            loadSessions();
        } catch { /* */ }
    }, [loadSessions]);

    /* ── Switch Session ──────────────────────────── */
    const switchSession = useCallback(async (sid: string) => {
        setSessionId(sid);
        await loadHistory(sid);
    }, [loadHistory]);

    /* ── Delete Session ──────────────────────────── */
    const deleteSession = useCallback(async (sid: string, e: React.MouseEvent) => {
        e.stopPropagation();
        try {
            await fetch(`${API}/api/agent/sessions/${sid}`, { method: 'DELETE' });
            setSessions(prev => prev.filter(s => s.session_id !== sid));
            if (sessionId === sid) {
                // Deleted active session — start new one
                startNewChat();
            }
        } catch { /* */ }
    }, [sessionId, startNewChat]);

    /* ── Send Message ────────────────────────────── */
    const sendMessage = useCallback(async (e?: FormEvent) => {
        e?.preventDefault();
        const text = input.trim();
        if (!text || streaming) return;

        setInput('');
        setMessages(prev => [...prev, { role: 'user', content: text }]);
        setStreaming(true);

        try {
            const res = await fetch(`${API}/api/agent/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, session_id: sessionId }),
            });

            if (!res.ok) throw new Error(`HTTP ${res.status}`);

            const contentType = res.headers.get('content-type') ?? '';
            if (contentType.includes('text/event-stream') || contentType.includes('text/plain')) {
                const reader = res.body?.getReader();
                const decoder = new TextDecoder();
                let agentText = '';
                let providerLabel = '';
                setMessages(prev => [...prev, { role: 'agent', content: '' }]);

                if (reader) {
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        const chunk = decoder.decode(value, { stream: true });
                        const lines = chunk.split('\n');
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                const data = line.slice(6);
                                if (data === '[DONE]') continue;
                                try {
                                    const parsed = JSON.parse(data);
                                    if (parsed.meta?.provider) {
                                        providerLabel = parsed.meta.provider;
                                    } else {
                                        agentText += parsed.content ?? parsed.text ?? '';
                                    }
                                } catch {
                                    agentText += data;
                                }
                            }
                        }
                        setMessages(prev => {
                            const copy = [...prev];
                            copy[copy.length - 1] = { role: 'agent', content: agentText, provider: providerLabel || undefined };
                            return copy;
                        });
                    }
                }
            } else {
                const data = await res.json();
                const reply = data.reply ?? data.response ?? data.message ?? 'No response';
                setMessages(prev => [...prev, { role: 'agent', content: reply }]);
            }

            // Refresh sessions list (title may have changed)
            loadSessions();
        } catch (err: any) {
            setMessages(prev => [...prev, { role: 'agent', content: `Error: ${err.message}` }]);
        } finally {
            setStreaming(false);
            inputRef.current?.focus();
        }
    }, [input, streaming, sessionId, loadSessions]);

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    /* ── Rich markdown rendering ────────────────── */
    const renderContent = (content: string) => {
        return (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {content}
            </ReactMarkdown>
        );
    };

    const quickActions = [
        'What is the current market regime?',
        'Explain the latest prediction',
        'Show quant analysis summary',
        'What are the risk levels?',
    ];

    return (
        <div className="nexus-chat-layout">
            {/* ── Sidebar ──────────────────────────── */}
            <div className={`chat-sidebar ${sidebarOpen ? 'open' : 'collapsed'}`}>
                <div className="sidebar-header">
                    <button className="btn btn-primary sidebar-new-chat" onClick={startNewChat} title="New Chat">
                        <IconPlus size={15} />
                        {sidebarOpen && <span>New Chat</span>}
                    </button>
                    <button className="btn btn-ghost sidebar-toggle" onClick={() => setSidebarOpen(p => !p)} title={sidebarOpen ? 'Collapse' : 'Expand'}>
                        <IconChevronLeft size={16} style={{ transform: sidebarOpen ? undefined : 'rotate(180deg)', transition: 'transform .2s' }} />
                    </button>
                </div>
                {sidebarOpen && (
                    <div className="sidebar-sessions">
                        {sessions.length === 0 && (
                            <div className="sidebar-empty">No conversations yet</div>
                        )}
                        {sessions.map(s => (
                            <div
                                key={s.session_id}
                                className={`session-item ${s.session_id === sessionId ? 'active' : ''}`}
                                onClick={() => switchSession(s.session_id)}
                            >
                                <IconChatBubble size={14} style={{ flexShrink: 0, opacity: 0.5 }} />
                                <div className="session-info">
                                    <div className="session-title">{s.title}</div>
                                    <div className="session-meta">{s.message_count} msgs · {relativeTime(s.last_message_at)}</div>
                                </div>
                                <button
                                    className="session-delete"
                                    onClick={(e) => deleteSession(s.session_id, e)}
                                    title="Delete"
                                >
                                    <IconTrash size={13} />
                                </button>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* ── Main Chat Area ───────────────────── */}
            <div className="chat-main">
                {/* Market Ticker */}
                <MarketTicker />

                {/* Messages */}
                <div className="chat-messages">
                    {messages.length === 0 && (
                        <div className="empty-state" style={{ flex: 1 }}>
                            <div style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-1)', marginBottom: 4 }}>Dr. Nexus</div>
                            <div style={{ fontSize: 12, maxWidth: 300 }}>
                                Ask me anything about market conditions, predictions, or trading strategies.
                            </div>
                            <div className="flex gap-8" style={{ marginTop: 16, flexWrap: 'wrap', justifyContent: 'center' }}>
                                {quickActions.map(q => (
                                    <button key={q} className="btn btn-sm" onClick={() => { setInput(q); inputRef.current?.focus(); }}>
                                        {q}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}
                    {messages.map((msg, i) => (
                        <div key={i} className={`chat-bubble ${msg.role} animate-in`}>
                            {msg.role === 'agent' ? (
                                <div className="nexus-md">{renderContent(msg.content)}</div>
                            ) : msg.content}
                            {msg.role === 'agent' && msg.provider && (
                                <div className="provider-badge">via {msg.provider}</div>
                            )}
                        </div>
                    ))}
                    {streaming && messages.length > 0 && messages[messages.length - 1].content === '' && (
                        <div className="chat-bubble agent">
                            <div className="typing-dots"><span /><span /><span /></div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input */}
                <div className="chat-input-row">
                    <textarea
                        ref={inputRef}
                        className="chat-input"
                        placeholder="Ask Dr. Nexus…"
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        rows={1}
                    />
                    <button className="btn btn-primary" onClick={() => sendMessage()} disabled={streaming || !input.trim()}>
                        <IconSend style={{ width: 16, height: 16 }} />
                    </button>
                </div>
            </div>
        </div>
    );
}
