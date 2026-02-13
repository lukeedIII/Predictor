"""
Nexus Shadow-Quant — Dr. Nexus Memory System
===============================================
SQLite-based persistent memory for Dr. Nexus AI agent.
Stores conversation history AND extracted knowledge insights.

Tables:
  - conversations: Full chat messages (user + agent)
  - knowledge:     Extracted key facts/insights from conversations
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Optional

import config

DB_PATH = os.path.join(config.DATA_ROOT, "nexus_memory.db")


def _get_conn() -> sqlite3.Connection:
    """Get or create the SQLite connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'agent')),
                content TEXT NOT NULL,
                provider TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                source_msg_id INTEGER,
                confidence REAL DEFAULT 1.0,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (source_msg_id) REFERENCES conversations(id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
            CREATE INDEX IF NOT EXISTS idx_conv_created ON conversations(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_knowledge_cat ON knowledge(category);
        """)
        conn.commit()
        logging.info(f"Dr. Nexus memory initialized: {DB_PATH}")
    finally:
        conn.close()


# ── Conversation History ─────────────────────────────

def save_message(session_id: str, role: str, content: str, provider: str = None) -> int:
    """Save a chat message. Returns the message ID."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO conversations (session_id, role, content, provider) VALUES (?, ?, ?, ?)",
            (session_id, role, content, provider)
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_recent_messages(session_id: str = None, limit: int = 20) -> List[Dict]:
    """Get recent messages, optionally filtered by session."""
    conn = _get_conn()
    try:
        if session_id:
            rows = conn.execute(
                "SELECT id, session_id, role, content, provider, created_at "
                "FROM conversations WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, session_id, role, content, provider, created_at "
                "FROM conversations ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in reversed(rows)]
    finally:
        conn.close()


def get_all_sessions() -> List[Dict]:
    """Get list of all chat sessions with message counts."""
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT session_id, 
                   COUNT(*) as message_count,
                   MIN(created_at) as started_at,
                   MAX(created_at) as last_message_at
            FROM conversations 
            GROUP BY session_id 
            ORDER BY last_message_at DESC
            LIMIT 50
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Knowledge Store ──────────────────────────────────

KNOWLEDGE_CATEGORIES = [
    "market_insight",       # BTC market observations
    "trading_lesson",       # Lessons from trades (wins/losses)
    "user_preference",      # User's trading style preferences
    "model_observation",    # Observations about the AI model's behavior
    "risk_note",            # Risk alerts or patterns
    "general",              # Anything else worth remembering
]


def save_knowledge(category: str, content: str, source_msg_id: int = None,
                   confidence: float = 1.0) -> int:
    """Save a knowledge item. Returns the knowledge ID."""
    if category not in KNOWLEDGE_CATEGORIES:
        category = "general"
    conn = _get_conn()
    try:
        # Check for duplicate (same category + very similar content)
        existing = conn.execute(
            "SELECT id FROM knowledge WHERE category = ? AND content = ?",
            (category, content)
        ).fetchone()
        if existing:
            # Update timestamp instead of duplicating
            conn.execute(
                "UPDATE knowledge SET updated_at = datetime('now'), confidence = ? WHERE id = ?",
                (confidence, existing['id'])
            )
            conn.commit()
            return existing['id']
        
        cur = conn.execute(
            "INSERT INTO knowledge (category, content, source_msg_id, confidence) "
            "VALUES (?, ?, ?, ?)",
            (category, content, source_msg_id, confidence)
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_knowledge(category: str = None, limit: int = 30) -> List[Dict]:
    """Retrieve knowledge items, optionally by category."""
    conn = _get_conn()
    try:
        if category:
            rows = conn.execute(
                "SELECT id, category, content, confidence, created_at, updated_at "
                "FROM knowledge WHERE category = ? ORDER BY updated_at DESC LIMIT ?",
                (category, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, category, content, confidence, created_at, updated_at "
                "FROM knowledge ORDER BY updated_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_knowledge_summary() -> str:
    """Build a compact text summary of all knowledge for prompt injection."""
    items = get_knowledge(limit=50)
    if not items:
        return ""
    
    by_cat = {}
    for item in items:
        cat = item['category']
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(item['content'])
    
    lines = ["## Dr. Nexus Memory Bank"]
    for cat, entries in by_cat.items():
        label = cat.replace("_", " ").title()
        lines.append(f"\n### {label}")
        for e in entries[:10]:  # Max 10 per category
            lines.append(f"- {e}")
    
    return "\n".join(lines)


def get_conversation_context(session_id: str, max_messages: int = 10) -> str:
    """Build conversation context string for prompt injection."""
    msgs = get_recent_messages(session_id=session_id, limit=max_messages)
    if not msgs:
        return ""
    
    lines = ["## Recent Conversation"]
    for m in msgs:
        role = "User" if m['role'] == 'user' else "Dr. Nexus"
        # Truncate long messages for context window
        content = m['content'][:300]
        if len(m['content']) > 300:
            content += "..."
        lines.append(f"**{role}**: {content}")
    
    return "\n".join(lines)


# ── Stats ────────────────────────────────────────────

def get_stats() -> Dict:
    """Get memory system stats."""
    conn = _get_conn()
    try:
        msg_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        knowledge_count = conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
        session_count = conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM conversations"
        ).fetchone()[0]
        return {
            "total_messages": msg_count,
            "total_knowledge": knowledge_count,
            "total_sessions": session_count,
        }
    finally:
        conn.close()


# Initialize DB on import
try:
    init_db()
except Exception as e:
    logging.warning(f"Could not initialize memory DB: {e}")
