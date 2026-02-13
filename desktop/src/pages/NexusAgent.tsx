import { useState, useRef, useEffect } from 'react';
import { apiPost, API_BASE } from '../hooks/useApi';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

/* ─── Types ──────────────────────────────────────── */
type Message = {
    id: number;
    role: 'user' | 'agent';
    content: string;
    provider?: string;
    timestamp: Date;
};

/* ─── Safe Markdown Renderer ─────────────────────── */
function NexusMarkdown({ content }: { content: string }) {
    return (
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {content}
        </ReactMarkdown>
    );
}

/* ─── Page ───────────────────────────────────────── */
export default function NexusAgent() {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: 0,
            role: 'agent',
            content: "## Welcome! I'm Dr. Nexus 🧬\n\nYour AI-powered quant analyst with **real-time access** to the entire platform state.\n\n### What I can analyze:\n- **Market State** — BTC price, AI predictions, confidence scores\n- **Your Positions** — open trades, PnL, risk exposure, TP/SL levels\n- **Portfolio Stats** — win rate, Sharpe ratio, drawdown, Kelly criterion\n- **Algorithm Internals** — XGBoost ensemble, Hurst regime, FFT cycles, order flow\n\n> Ask me anything about the current market conditions, your positions, or the algorithm's decision-making.\n\n---\n*Powered by GPT-4o · Live state injected with every message · **Persistent Memory Active***",
            provider: 'gpt-4o',
            timestamp: new Date(),
        },
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [memoryStats, setMemoryStats] = useState<{ total_messages: number; total_knowledge: number } | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const idRef = useRef(1);

    // Load chat history from backend on mount
    useEffect(() => {
        const loadHistory = async () => {
            try {
                const resp = await fetch(`${API_BASE}/api/agent/history?limit=30`);
                if (resp.ok) {
                    const data = await resp.json();
                    if (data.messages && data.messages.length > 0) {
                        const loaded: Message[] = data.messages.map((m: Record<string, unknown>, i: number) => ({
                            id: i + 1,
                            role: m.role as 'user' | 'agent',
                            content: m.content as string,
                            provider: (m.provider as string) || undefined,
                            timestamp: new Date(m.created_at as string),
                        }));
                        idRef.current = loaded.length + 1;
                        setMessages(prev => [...prev, ...loaded]);
                    }
                }
            } catch {
                // Backend not ready yet, skip
            }

            // Load memory stats
            try {
                const resp = await fetch(`${API_BASE}/api/agent/memory-stats`);
                if (resp.ok) {
                    setMemoryStats(await resp.json());
                }
            } catch {
                // Skip
            }
        };
        loadHistory();
    }, []);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const quickPrompts = [
        { emoji: '📊', text: 'Full market analysis' },
        { emoji: '💼', text: 'Position risk report' },
        { emoji: '🧬', text: 'How is the algorithm performing?' },
        { emoji: '⚠️', text: 'What risks should I watch?' },
        { emoji: '🎯', text: 'Should I open a trade now?' },
        { emoji: '📈', text: 'Explain the current regime' },
    ];

    const sendMessage = async (text?: string) => {
        const msg = (text || input).trim();
        if (!msg || loading) return;

        const userMsg: Message = {
            id: idRef.current++,
            role: 'user',
            content: msg,
            timestamp: new Date(),
        };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        try {
            const resp = await apiPost('/api/agent/chat', {
                message: msg,
                session_id: sessionId,
            });
            const data = resp as { reply: string; provider: string; session_id: string };

            // Track session_id from backend
            if (data.session_id && !sessionId) {
                setSessionId(data.session_id);
            }

            const agentMsg: Message = {
                id: idRef.current++,
                role: 'agent',
                content: data.reply || 'No response received.',
                provider: data.provider,
                timestamp: new Date(),
            };
            setMessages(prev => [...prev, agentMsg]);

            // Refresh memory stats
            try {
                const statsResp = await fetch(`${API_BASE}/api/agent/memory-stats`);
                if (statsResp.ok) setMemoryStats(await statsResp.json());
            } catch { /* ignore */ }
        } catch (err) {
            const errorMsg: Message = {
                id: idRef.current++,
                role: 'agent',
                content: '⚠️ Failed to reach the AI. Check your connection and try again.',
                timestamp: new Date(),
            };
            setMessages(prev => [...prev, errorMsg]);
        }

        setLoading(false);
        inputRef.current?.focus();
    };

    const startNewSession = async () => {
        try {
            const resp = await apiPost('/api/agent/new-session', {});
            const data = resp as { session_id: string };
            setSessionId(data.session_id);
            setMessages([{
                id: 0,
                role: 'agent',
                content: "## New Session Started 🧬\n\nI still remember everything from our previous conversations! My **knowledge bank** carries over.\n\nHow can I help you?",
                provider: 'gpt-4o',
                timestamp: new Date(),
            }]);
            idRef.current = 1;
        } catch { /* ignore */ }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const canSend = !loading && input.trim().length > 0;

    return (
        <div className="animate-in chat-layout">
            {/* Header */}
            <div className="chat-header">
                <div>
                    <div className="page-title">🧬 Dr. Nexus</div>
                    <div className="page-subtitle">
                        AI Quant Analyst · GPT-4o · Persistent Memory
                    </div>
                </div>
                <div className="chat-actions">
                    {memoryStats && (memoryStats.total_messages > 0 || memoryStats.total_knowledge > 0) && (
                        <div className="chat-memory-badge">
                            🧠 {memoryStats.total_knowledge} insights · {memoryStats.total_messages} msgs
                        </div>
                    )}
                    <button className="chat-new-btn" onClick={startNewSession}>
                        + New Chat
                    </button>
                    <div className="chat-live-badge">
                        <div className="chat-live-dot" />
                        Live
                    </div>
                </div>
            </div>

            {/* Quick Prompts */}
            <div className="chat-quick-prompts">
                {quickPrompts.map((prompt, i) => (
                    <button
                        key={i}
                        className="chat-quick-btn"
                        onClick={() => sendMessage(`${prompt.emoji} ${prompt.text}`)}
                        disabled={loading}
                    >
                        {prompt.emoji} {prompt.text}
                    </button>
                ))}
            </div>

            {/* Messages */}
            <div className="chat-messages">
                {messages.map(msg => (
                    <div key={msg.id} className={`chat-msg-row ${msg.role}`}>
                        {/* Agent avatar */}
                        {msg.role === 'agent' && (
                            <div className="chat-avatar">🧬</div>
                        )}

                        <div className={`chat-bubble ${msg.role}`}>
                            {msg.role === 'agent' ? (
                                <div className="dr-nexus-response">
                                    <NexusMarkdown content={msg.content} />
                                </div>
                            ) : (
                                <div className="font-500">{msg.content}</div>
                            )}
                            <div className={`chat-meta ${msg.role}`}>
                                <span>{msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                                {msg.provider && (
                                    <span className="chat-provider-badge">{msg.provider}</span>
                                )}
                            </div>
                        </div>
                    </div>
                ))}

                {/* Typing indicator */}
                {loading && (
                    <div className="chat-typing">
                        <div className="chat-avatar">🧬</div>
                        <div className="chat-typing-bubble">
                            <div className="typing-dot" style={{ animationDelay: '0s' }} />
                            <div className="typing-dot" style={{ animationDelay: '0.15s' }} />
                            <div className="typing-dot" style={{ animationDelay: '0.3s' }} />
                            <span className="chat-typing-text">
                                Dr. Nexus is analyzing...
                            </span>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input Bar */}
            <div className="chat-input-bar">
                <input
                    ref={inputRef}
                    type="text"
                    className="chat-input"
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask Dr. Nexus about BTC, your positions, or the algorithm..."
                    disabled={loading}
                />
                <button
                    className={`chat-send-btn ${canSend ? 'active' : 'inactive'}`}
                    onClick={() => sendMessage()}
                    disabled={!canSend}
                >
                    {loading ? '⏳' : 'Send'}
                </button>
            </div>
        </div>
    );
}
