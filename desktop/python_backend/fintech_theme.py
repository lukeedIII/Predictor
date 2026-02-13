"""
FinTech Dark Mode Design System
================================
Modern, professional financial dashboard aesthetic.
Replaces the neon/gamer look with institutional-grade UI.

Design Tokens:
- Background: Deep Slate/Navy tones
- Typography: Inter (clean sans-serif)
- Accents: Muted/desaturated data colors
- Spacing: 8px grid system
- Surfaces: Subtle elevation via box-shadow (no borders)
"""

# ============================================================
#  COLOR PALETTE
# ============================================================
COLORS = {
    # Backgrounds (slate/navy gradient)
    'bg_primary':   '#0F1923',      # Deepest background
    'bg_secondary': '#131E2B',      # Page background
    'bg_gradient':  'linear-gradient(180deg, #0F1923 0%, #131E2B 50%, #0F1923 100%)',

    # Surfaces (elevated cards)
    'surface_0':    '#182332',      # Card background
    'surface_1':    '#1D2A3A',      # Elevated card
    'surface_2':    '#223242',      # Hover / active

    # Text hierarchy
    'text_primary':   '#E8ECF1',    # Primary text
    'text_secondary': '#8A9BB5',    # Labels, captions
    'text_tertiary':  '#566A84',    # Muted, disabled

    # Data Indicators (muted/desaturated)
    'positive':     '#34D399',      # Muted teal-green
    'negative':     '#F87171',      # Muted coral-red
    'warning':      '#FBBF24',      # Amber
    'info':         '#60A5FA',      # Soft blue
    'accent':       '#818CF8',      # Muted indigo
    'neutral':      '#6B7B93',      # Gray-blue

    # Borders / dividers
    'border':       'rgba(138, 155, 181, 0.12)',
    'border_subtle': 'rgba(138, 155, 181, 0.06)',

    # Shadows for elevation
    'shadow_sm':    '0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.2)',
    'shadow_md':    '0 4px 6px rgba(0,0,0,0.3), 0 2px 4px rgba(0,0,0,0.2)',
    'shadow_lg':    '0 10px 24px rgba(0,0,0,0.4), 0 4px 8px rgba(0,0,0,0.3)',
}

# ============================================================
#  GLOBAL CSS — Inject via st.markdown()
# ============================================================
GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Base ── */
.stApp {{
    background: {COLORS['bg_gradient']} !important;
    color: {COLORS['text_primary']} !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

/* ── Typography ── */
h1, h2, h3 {{
    font-family: 'Inter', sans-serif !important;
    color: {COLORS['text_primary']} !important;
    background: none !important;
    -webkit-text-fill-color: {COLORS['text_primary']} !important;
    letter-spacing: -0.02em !important;
    font-weight: 700 !important;
}}
h1 {{ font-size: 1.75rem !important; }}
h2 {{ font-size: 1.25rem !important; color: {COLORS['text_secondary']} !important; -webkit-text-fill-color: {COLORS['text_secondary']} !important; }}
h3 {{ font-size: 1rem !important; color: {COLORS['text_secondary']} !important; -webkit-text-fill-color: {COLORS['text_secondary']} !important; }}

/* ── Cards (Surface Elevation) ── */
div[data-testid="stMetric"],
div[data-testid="stHorizontalBlock"] > div {{
    background: {COLORS['surface_0']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: {COLORS['shadow_sm']} !important;
    backdrop-filter: none !important;
}}

/* ── Stat Card ── */
.stat-card {{
    background: {COLORS['surface_0']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin: 4px 0;
    box-shadow: {COLORS['shadow_sm']};
    transition: box-shadow 0.2s ease, background 0.2s ease;
}}
.stat-card:hover {{
    background: {COLORS['surface_1']};
    box-shadow: {COLORS['shadow_md']};
}}

/* ── Position Badges ── */
.position-badge {{
    padding: 6px 16px;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    display: inline-block;
}}
.long-badge {{
    background: rgba(52, 211, 153, 0.12);
    border: 1px solid rgba(52, 211, 153, 0.3);
    color: {COLORS['positive']};
}}
.short-badge {{
    background: rgba(248, 113, 113, 0.12);
    border: 1px solid rgba(248, 113, 113, 0.3);
    color: {COLORS['negative']};
}}
.flat-badge {{
    background: rgba(129, 140, 248, 0.10);
    border: 1px solid rgba(129, 140, 248, 0.25);
    color: {COLORS['accent']};
}}

/* ── Status LED ── */
@keyframes pulse-soft {{
    0%, 100% {{ opacity: 0.7; }}
    50% {{ opacity: 1; }}
}}
.pulse-led {{ animation: pulse-soft 3s ease-in-out infinite; display: inline-block; }}

/* ── News Feed ── */
.news-container {{
    background: {COLORS['surface_0']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 16px;
    max-height: 200px;
    overflow-y: auto;
}}
.news-item {{
    background: {COLORS['surface_1']};
    border-left: 3px solid;
    padding: 12px 16px;
    margin-bottom: 8px;
    border-radius: 8px;
}}
.news-source {{
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    color: {COLORS['text_secondary']};
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.news-content {{
    color: {COLORS['text_primary']};
    font-size: 0.85rem;
    margin-top: 4px;
    line-height: 1.4;
}}
.news-sentiment {{
    font-size: 0.75rem;
    margin-top: 4px;
    font-weight: 600;
}}

/* ── Scanning / Loading Placeholder ── */
.scanning-pattern {{
    background: {COLORS['surface_0']};
    border: 1px dashed {COLORS['border']};
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: {COLORS['text_tertiary']};
    font-weight: 500;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
}}

/* ── Progress Bar ── */
.stProgress > div > div > div > div {{
    background: linear-gradient(90deg, {COLORS['accent']} 0%, {COLORS['info']} 100%) !important;
    border-radius: 8px !important;
}}

/* ── Buttons ── */
.stButton > button {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    letter-spacing: 0.02em !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease !important;
}}

/* ── Selectbox ── */
div[data-baseweb="select"] {{
    font-family: 'Inter', sans-serif !important;
}}

/* ── Streamlit Chrome (hide) ── */
footer, header, .stDeployButton {{ visibility: hidden !important; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {COLORS['bg_primary']}; }}
::-webkit-scrollbar-thumb {{ background: {COLORS['text_tertiary']}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {COLORS['text_secondary']}; }}
</style>
"""

# ============================================================
#  PLOTLY THEME (Shared chart layout defaults)
# ============================================================
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color=COLORS['text_secondary'], size=11),
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor=COLORS['border'], zeroline=False),
    legend=dict(
        orientation="h", yanchor="top", y=1.02, xanchor="right", x=1,
        font=dict(size=10, color=COLORS['text_secondary']),
        bgcolor='rgba(0,0,0,0)'
    ),
    margin=dict(l=10, r=10, t=30, b=10),
)

# ============================================================
#  HELPER: Inline style shortcuts
# ============================================================
def label(text: str) -> str:
    """Small muted label text."""
    return f"<span style='color:{COLORS['text_secondary']};font-size:0.8rem;font-weight:500;letter-spacing:0.03em;'>{text}</span>"

def value(text: str, color: str = None, size: str = '1.2rem') -> str:
    """Bold data value."""
    c = color or COLORS['text_primary']
    return f"<span style='color:{c};font-size:{size};font-weight:700;'>{text}</span>"

def big_value(text: str, color: str = None) -> str:
    """Hero metric value."""
    c = color or COLORS['text_primary']
    return f"<span style='color:{c};font-size:2rem;font-weight:800;letter-spacing:-0.02em;'>{text}</span>"

def sub_text(text: str, color: str = None) -> str:
    """Secondary info below a metric."""
    c = color or COLORS['text_tertiary']
    return f"<span style='color:{c};font-size:0.75rem;font-weight:500;'>{text}</span>"

def card_html(content: str) -> str:
    """Wrap content in a stat-card div."""
    return f"<div class='stat-card'>{content}</div>"
