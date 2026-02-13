import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from predictor import NexusPredictor
from whale_monitor import WhaleMonitor
from math_core import MathCore
import config, time, os
import fintech_theme as theme
from datetime import datetime, timedelta

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="Nexus Shadow-Quant", layout="wide", initial_sidebar_state="auto")

# ========== LOADING SCREEN ==========
loading_header = st.empty()
loading_header.markdown(f"""
<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');</style>
<div style='text-align:center;font-family:Inter,sans-serif;color:{theme.COLORS["text_secondary"]};font-size:1.5rem;font-weight:700;margin:50px 0 20px 0;letter-spacing:-0.02em;'>Loading Nexus...</div>
""", unsafe_allow_html=True)

progress_bar = st.progress(0)
status_text = st.empty()

# ========== CSS INJECTION (FinTech Dark Mode) ==========
st.markdown(theme.GLOBAL_CSS, unsafe_allow_html=True)

# Step 1: Initialize (30%)
status_text.markdown("⚡ **Initializing engines...** `30%`")
progress_bar.progress(30)
whale_monitor = WhaleMonitor()
math_core = MathCore()

# Step 2: Load data (60%)
status_text.markdown("⚡ **Loading market data...** `60%`")
progress_bar.progress(60)

def load_data():
    # Prefer parquet (1.7MB, ~0.1s) over CSV (206MB, ~5s)
    if os.path.exists(config.MARKET_DATA_PARQUET_PATH):
        df = pd.read_parquet(config.MARKET_DATA_PARQUET_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    elif os.path.exists(config.MARKET_DATA_PATH):
        df = pd.read_csv(config.MARKET_DATA_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return pd.DataFrame()

df = load_data()
current_price = df['close'].iloc[-1] if not df.empty else 0
data_rows = len(df) if not df.empty else 0

# Step 3: Init predictor (80%)
status_text.markdown(f"⚡ **{data_rows:,} candles loaded. Initializing AI...** `80%`")
progress_bar.progress(80)

@st.cache_resource
def get_predictor():
    pred = NexusPredictor()
    return pred

predictor = get_predictor()
is_trained = predictor.is_trained

# Auto-train if model doesn't exist and we have data
if not is_trained and not df.empty and len(df) > 100:
    status_text.markdown("🧠 **Training AI model (first run)...** `85%`")
    progress_bar.progress(85)
    try:
        predictor.train()
        is_trained = True
    except Exception as e:
        logging.warning(f"Auto-training failed: {e}")

train_progress = 100 if is_trained else 0

# Step 4: Ready (100%)
status_text.markdown("✅ **Ready!** `100%`")
progress_bar.progress(100)
time.sleep(0.2)

# CLEAR ALL LOADING ELEMENTS
loading_header.empty()
progress_bar.empty()
status_text.empty()

# ========== TOP BANNER ==========
st.markdown("<h1 style='text-align: center; margin-bottom: 5px;'>Nexus Shadow-Quant Pro</h1>", unsafe_allow_html=True)
status_text_display = "Operational" if is_trained else f"Calibrating ({train_progress:.0f}%)"

# AUDIT STATUS BADGE
is_verified = predictor.is_statistically_verified if hasattr(predictor, 'is_statistically_verified') else False
audit_color = theme.COLORS['positive'] if is_verified else theme.COLORS['negative']
audit_text = "Verified" if is_verified else "Unverified"
audit_icon = "✓" if is_verified else "⚠"

st.markdown(f"""
<p style='text-align: center; font-family: Inter, sans-serif; color: {theme.COLORS['text_secondary']}; letter-spacing: 0.05em; font-size: 0.8rem; font-weight: 500;'>
<span class='pulse-led' style='color: {theme.COLORS['positive'] if is_trained else theme.COLORS['warning']};'>●</span>
{predictor.device.upper()} · {status_text_display}
<span style='margin-left: 16px; padding: 4px 12px; border-radius: 6px; background: {"rgba(52,211,153,0.1)" if is_verified else "rgba(248,113,113,0.1)"}; border: 1px solid {audit_color}; color: {audit_color}; font-weight: 600;'>
{audit_icon} {audit_text}
</span>
</p>
""", unsafe_allow_html=True)


# ========== 4-PILLAR METRICS BAR ==========
m1, m2, m3, m4 = st.columns(4)

with m1:
    if is_trained:
        pred = predictor.get_prediction()
        h = pred.get('hurst', 0.5)
        state = "Trending" if h > 0.55 else ("Reverting" if h < 0.45 else "Stochastic")
        s_color = theme.COLORS['positive'] if h > 0.55 else (theme.COLORS['negative'] if h < 0.45 else theme.COLORS['info'])
        st.markdown(f"""<div class='stat-card'>
            {theme.label('REGIME')}<br>
            {theme.value(state, s_color, '1.3rem')}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("<div class='scanning-pattern' style='height:80px;'>Scanning...</div>", unsafe_allow_html=True)

with m2:
    if is_trained:
        dir_color = theme.COLORS['positive'] if pred.get('direction') == 'UP' else theme.COLORS['negative']
        target_1h = pred.get('target_price_1h', pred.get('target_price', 0))
        target_2h = pred.get('target_price_2h', target_1h)
        st.markdown(f"""<div class='stat-card'>
            {theme.label('PREDICTION TARGETS')}<br>
            {theme.value(pred.get('direction', '--'), dir_color, '1.3rem')}<br>
            <span style='color:{theme.COLORS["accent"]};font-weight:600;font-size:0.85rem;'>1H:</span>
            <span style='color:{theme.COLORS["text_primary"]};font-size:0.85rem;'>${target_1h:,.2f}</span>
            <span style='margin-left:8px;color:{theme.COLORS["info"]};font-weight:600;font-size:0.85rem;'>2H:</span>
            <span style='color:{theme.COLORS["text_primary"]};font-size:0.85rem;'>${target_2h:,.2f}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("<div class='scanning-pattern' style='height:80px;'>Analyzing...</div>", unsafe_allow_html=True)

with m3:
    if is_trained:
        conf = float(pred.get('confidence', 0))
        st.markdown(f"""<div class='stat-card'>
            {theme.label('AI CONFIDENCE')}<br>
            {theme.value(f'{conf:.1f}%', theme.COLORS['text_primary'], '1.5rem')}
        </div>""", unsafe_allow_html=True)
        st.progress(float(conf / 100))
    else:
        st.markdown("<div class='scanning-pattern' style='height:80px;'>Calibrating...</div>", unsafe_allow_html=True)

with m4:
    acc = predictor.calculate_accuracy()
    acc_color = theme.COLORS['positive'] if acc >= 55 else (theme.COLORS['warning'] if acc >= 45 else theme.COLORS['negative'])
    acc_label = 'Realized Accuracy' if acc > 0 else 'Awaiting Data'
    st.markdown(f"""<div class='stat-card'>
        {theme.label(acc_label.upper())}<br>
        {theme.value(f'{acc:.1f}%', acc_color, '1.5rem')}<br>
        {theme.sub_text(f'24h Window · v{config.VERSION}')}
    </div>""", unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)

# ========== MAIN DASHBOARD: CHART + HUD ==========
col_chart, col_hud = st.columns([1.8, 1])

# === LEFT: PRICE CHART ===
with col_chart:
    st.subheader("📈 Price & Fibonacci")
    
    if not df.empty and len(df) > 10:
        fig = go.Figure()
        hist_df = df.tail(150)
        
        # 1. FIBONACCI RETRACEMENT LEVELS
        price_min, price_max = hist_df['close'].min(), hist_df['close'].max()
        diff = price_max - price_min
        fib_levels = [0.382, 0.5, 0.618]
        
        for lvl in fib_levels:
            price_lvl = price_max - (diff * lvl)
            fig.add_trace(go.Scatter(
                x=[hist_df['timestamp'].iloc[0], hist_df['timestamp'].iloc[-1]],
                y=[price_lvl, price_lvl],
                mode='lines',
                line=dict(color=f'rgba(129, 140, 248, 0.35)', width=1, dash='dash'),
                name=f"Fib {lvl}",
                hoverinfo='skip'
            ))
            fig.add_annotation(x=hist_df['timestamp'].iloc[-1], y=price_lvl,
                              text=f" {lvl}", showarrow=False, font=dict(size=10, color=theme.COLORS['accent']), xanchor='left')

        # 2. MONTE CARLO PROBABILITY CLOUD
        if is_trained:
            vol = df['close'].pct_change().std()
            if vol > 0:
                mc_paths = math_core.run_monte_carlo(current_price, vol, steps=60, simulations=500)
                future_times = [hist_df['timestamp'].iloc[-1] + pd.Timedelta(minutes=i) for i in range(60)]
                
                upper_95 = np.percentile(mc_paths, 95, axis=0)
                lower_05 = np.percentile(mc_paths, 5, axis=0)
                median = np.percentile(mc_paths, 50, axis=0)
                
                # Cloud fill
                fig.add_trace(go.Scatter(x=future_times, y=upper_95, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=future_times, y=lower_05, mode='lines', fill='tonexty',
                                        fillcolor='rgba(129, 140, 248, 0.08)', line=dict(width=0), name="Probability Cloud", hoverinfo='skip'))
                # Median line
                fig.add_trace(go.Scatter(x=future_times, y=median, mode='lines', line=dict(color=theme.COLORS['accent'], width=1, dash='dot'), name="MC Median"))
        
        # 2A. PREDICTION PATH (1H + 2H MARKERS)
        if is_trained:
            target_1h = pred.get('target_price_1h', pred.get('target_price', current_price))
            target_2h = pred.get('target_price_2h', target_1h)
            pred_color = theme.COLORS['positive'] if pred.get('direction') == 'UP' else theme.COLORS['negative']
            
            last_ts = hist_df['timestamp'].iloc[-1]
            t_1h = last_ts + pd.Timedelta(hours=1)
            t_2h = last_ts + pd.Timedelta(hours=2)
            
            # Prediction path line
            fig.add_trace(go.Scatter(
                x=[last_ts, t_1h, t_2h],
                y=[current_price, target_1h, target_2h],
                mode='lines+markers+text',
                line=dict(color=pred_color, width=3, dash='dash'),
                marker=dict(size=[8, 14, 14], symbol='diamond', color=pred_color, line=dict(width=2, color='white')),
                text=['NOW', '1H', '2H'],
                textposition='top center',
                textfont=dict(color=theme.COLORS['text_primary'], size=11, family='Inter'),
                name='Prediction Path',
                hovertemplate='<b>%{text}</b><br>$%{y:,.2f}<extra></extra>'
            ))

        # 3. KALMAN SIGNAL (NEON GLOW)
        smoothed = math_core.kalman_smooth(hist_df['close'].values)
        fig.add_trace(go.Scatter(
            x=hist_df['timestamp'], y=smoothed, name="Kalman Signal",
            line=dict(color=theme.COLORS['info'], width=3)
        ))
        # Glow effect
        fig.add_trace(go.Scatter(
            x=hist_df['timestamp'], y=smoothed, showlegend=False,
            line=dict(color='rgba(96, 165, 250, 0.2)', width=10), hoverinfo='skip'
        ))

        # 4. RAW PRICE
        fig.add_trace(go.Scatter(x=hist_df['timestamp'], y=hist_df['close'], name="Price", 
                                line=dict(color=theme.COLORS['text_tertiary'], width=1), opacity=0.5))

        fig.update_layout(
            **theme.PLOTLY_LAYOUT,
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown(f"<div class='scanning-pattern' style='height:500px;'>AWAITING DATAFEED... ({len(df)}/10 MIN ROWS)</div>", unsafe_allow_html=True)

# === RIGHT: QUANT INTELLIGENCE HUD ===
with col_hud:
    # TOP: FOURIER MARKET CYCLES
    st.subheader("📊 Market Cycles")
    if not df.empty and len(df) > 50:
        cycles = math_core.extract_cycles(df['close'].tail(100).values, top_n=3)
        labels = ['Short', 'Mid', 'Long']
        colors = [theme.COLORS['info'], theme.COLORS['accent'], theme.COLORS['positive']]
        
        c_fig = go.Figure()
        for i, (lbl, val, col) in enumerate(zip(labels, cycles, colors)):
            c_fig.add_trace(go.Bar(x=[lbl], y=[val], marker_color=col, name=lbl, text=f"{val:.2f}", textposition='outside'))
        
        c_fig.update_layout(**theme.PLOTLY_LAYOUT, height=180, showlegend=False)
        c_fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=30),
            yaxis=dict(showgrid=False, range=[0, max(cycles)*1.5 if max(cycles) > 0 else 1]),
        )
        st.plotly_chart(c_fig, use_container_width=True)
    else:
        st.markdown("<div class='scanning-pattern' style='height:180px;'>Extracting FFT...</div>", unsafe_allow_html=True)

    # HMM REGIME DETECTION
    st.subheader("🎯 Regime Detection")
    quant = predictor.last_quant_analysis
    if quant and quant.get('regime'):
        regime = quant['regime']
        regime_name = regime.get('regime', 'UNKNOWN')
        confidence = regime.get('confidence', 0)
        
        regime_styles = {
            'BULL': {'color': theme.COLORS['positive'], 'bg': 'rgba(52,211,153,0.08)'},
            'BEAR': {'color': theme.COLORS['negative'], 'bg': 'rgba(248,113,113,0.08)'},
            'SIDEWAYS': {'color': theme.COLORS['accent'], 'bg': 'rgba(129,140,248,0.08)'},
            'UNKNOWN': {'color': theme.COLORS['neutral'], 'bg': 'rgba(107,123,147,0.08)'}
        }
        style = regime_styles.get(regime_name, regime_styles['UNKNOWN'])
        
        st.markdown(f"""
        <div style="background: {style['bg']}; border: 1px solid {style['color']}30; border-radius: 12px; 
                    padding: 16px; text-align: center; margin-bottom: 16px;">
            {theme.value(regime_name, style['color'], '1.3rem')}
            <br>
            {theme.sub_text(f'{confidence:.0f}% Confidence')}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='scanning-pattern' style='height:100px;'>Training HMM...</div>", unsafe_allow_html=True)

    # ORDER FLOW IMBALANCE
    st.subheader("📊 Order Flow")
    if quant and quant.get('order_flow'):
        ofi = quant['order_flow']
        ofi_normalized = ofi.get('normalized', 0)
        ofi_signal = ofi.get('signal', 'NEUTRAL')
        
        if ofi_normalized > 0:
            bar_color = theme.COLORS['positive']
            label = 'Buy Pressure'
        elif ofi_normalized < 0:
            bar_color = theme.COLORS['negative']
            label = 'Sell Pressure'
        else:
            bar_color = theme.COLORS['neutral']
            label = 'Neutral'
        
        bar_width = int(abs(ofi_normalized) * 100)
        bar_direction = 'right' if ofi_normalized >= 0 else 'left'
        
        st.markdown(f"""
        <div style="background: {theme.COLORS['surface_0']}; border-radius: 12px; padding: 12px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: {theme.COLORS['negative']}; font-size: 0.75rem; font-weight: 600;">SELL</span>
                <span style="color: {bar_color}; font-weight: 700; font-size: 0.85rem;">{label}</span>
                <span style="color: {theme.COLORS['positive']}; font-size: 0.75rem; font-weight: 600;">BUY</span>
            </div>
            <div style="background: {theme.COLORS['surface_2']}; height: 16px; border-radius: 8px; position: relative; overflow: hidden;">
                <div style="position: absolute; left: 50%; width: 1px; height: 100%; background: {theme.COLORS['text_tertiary']}; opacity: 0.4;"></div>
                <div style="position: absolute; {'left: 50%' if ofi_normalized >= 0 else 'right: 50%'}; 
                            width: {bar_width}%; height: 100%; background: {bar_color}; 
                            border-radius: 8px; opacity: 0.7;"></div>
            </div>
            <div style="text-align: center; margin-top: 4px;">
                {theme.sub_text(f'{ofi_normalized:+.2f}')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='scanning-pattern' style='height:80px;'>Analyzing OFI...</div>", unsafe_allow_html=True)

    # BOTTOM: HURST GAUGE (CHAOS REGISTRY)
    st.subheader("🌀 Hurst Regime")
    if is_trained:
        h_val = pred.get('hurst', 0.5)
        
        if h_val > 0.55:
            gauge_color = theme.COLORS['positive']
            regime_text = "Trending"
        elif h_val < 0.45:
            gauge_color = theme.COLORS['negative']
            regime_text = "Mean Revert"
        else:
            gauge_color = theme.COLORS['info']
            regime_text = "Chaotic"
        
        g_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=h_val,
            number={'font': {'size': 28, 'family': 'Inter', 'color': gauge_color}},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"<b>{regime_text}</b>", 'font': {'size': 11, 'family': 'Inter', 'color': gauge_color}},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': theme.COLORS['text_tertiary'], 'tickfont': {'size': 8}},
                'bar': {'color': gauge_color, 'thickness': 0.3},
                'bgcolor': 'rgba(0,0,0,0)',
                'borderwidth': 1,
                'bordercolor': theme.COLORS['border'],
                'steps': [
                    {'range': [0, 0.45], 'color': 'rgba(248, 113, 113, 0.06)'},
                    {'range': [0.45, 0.55], 'color': 'rgba(96, 165, 250, 0.06)'},
                    {'range': [0.55, 1], 'color': 'rgba(52, 211, 153, 0.06)'}
                ],
                'threshold': {'line': {'color': theme.COLORS['text_primary'], 'width': 2}, 'thickness': 0.8, 'value': h_val}
            }
        ))
        g_fig.update_layout(
            **theme.PLOTLY_LAYOUT, height=160, margin=dict(l=15, r=15, t=30, b=5),
        )
        st.plotly_chart(g_fig, use_container_width=True)
    else:
        st.markdown("<div class='scanning-pattern' style='height:160px;'>Analyzing fractals...</div>", unsafe_allow_html=True)

    # NEW: JUMP RISK BADGE (Merton Jump Diffusion)
    st.subheader("⚡ Jump Risk")
    if quant and quant.get('jump_risk'):
        jump_risk = quant['jump_risk']
        risk_level = jump_risk.get('risk_level', 'UNKNOWN')
        intensity = jump_risk.get('jump_intensity', 0)
        
        risk_styles = {
            'HIGH': {'color': theme.COLORS['negative'], 'bg': 'rgba(248,113,113,0.08)'},
            'MEDIUM': {'color': theme.COLORS['warning'], 'bg': 'rgba(251,191,36,0.08)'},
            'LOW': {'color': theme.COLORS['positive'], 'bg': 'rgba(52,211,153,0.08)'},
            'UNKNOWN': {'color': theme.COLORS['neutral'], 'bg': 'rgba(107,123,147,0.08)'}
        }
        style = risk_styles.get(risk_level, risk_styles['UNKNOWN'])
        
        st.markdown(f"""
        <div style="background: {style['bg']}; border: 1px solid {style['color']}30; border-radius: 12px; 
                    padding: 16px; text-align: center; margin-bottom: 16px;">
            {theme.value(risk_level, style['color'], '1.1rem')}
            <br>
            {theme.sub_text(f'λ = {intensity:.3f}')}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='scanning-pattern' style='height:60px;'>Calibrating Merton...</div>", unsafe_allow_html=True)

    # NEW: ROUGHNESS H-GAUGE (Rough Volatility)
    st.subheader("📐 Volatility Roughness")
    if quant and quant.get('rough_vol'):
        rv = quant['rough_vol']
        H_val = rv.get('H', 0.5)
        interpretation = rv.get('interpretation', 'UNKNOWN')
        
        if H_val < 0.3:
            h_color = theme.COLORS['negative']
        elif H_val < 0.4:
            h_color = theme.COLORS['warning']
        elif H_val < 0.5:
            h_color = '#EAB308'
        else:
            h_color = theme.COLORS['positive']
        
        st.markdown(f"""
        <div style="background: {theme.COLORS['surface_0']}; border-radius: 12px; padding: 12px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                {theme.sub_text('H =')}
                {theme.value(f'{H_val:.3f}', h_color, '1.3rem')}
            </div>
            <div style="background: {theme.COLORS['surface_2']}; border-radius: 4px; height: 6px; margin-top: 8px;">
                <div style="background: linear-gradient(90deg, {theme.COLORS['negative']}, {theme.COLORS['warning']}, #EAB308, {theme.COLORS['positive']}); 
                            width: {min(H_val * 100, 100):.0f}%; height: 100%; border-radius: 4px;"></div>
            </div>
            <div style="text-align: center; margin-top: 4px;">
                {theme.sub_text(interpretation)}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='scanning-pattern' style='height:80px;'>Computing roughness...</div>", unsafe_allow_html=True)


# ========== FOOTER: CALIBRATION STATUS ==========

if not is_trained:
    st.markdown(f"""
    <div style="background: {theme.COLORS['surface_0']}; border: 1px solid {theme.COLORS['border']}; border-radius: 12px; padding: 16px; text-align: center; margin-top: 24px;">
        <span style="color: {theme.COLORS['accent']}; font-weight: 600; letter-spacing: 0.03em; font-size: 0.85rem;">Calibrating · {train_progress:.1f}% complete</span>
    </div>
    """, unsafe_allow_html=True)


# ========== NEWS FEED ==========
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("📰 News Intelligence")

try:
    from twitter_scraper import CryptoNewsScraper
    
    # Initialize and fetch news (CryptoPanic provides pre-scored sentiment)
    news_scraper = CryptoNewsScraper()
    news_items = news_scraper.fetch_news("BTC", limit=10)
    
    if news_items and len(news_items) > 0:
        # Create scrollable news container
        st.markdown("""
        <style>
        .news-container {
            background: """ + theme.COLORS['surface_0'] + """;
            border: 1px solid """ + theme.COLORS['border'] + """;
            border-radius: 12px;
            padding: 16px;
            max-height: 200px;
            overflow-y: auto;
        }
        .news-item {
            background: """ + theme.COLORS['surface_1'] + """;
            border-left: 3px solid;
            padding: 12px 16px;
            margin-bottom: 8px;
            border-radius: 8px;
        }
        .news-source {
            font-family: 'Inter', sans-serif;
            font-size: 0.7rem;
            font-weight: 600;
            color: """ + theme.COLORS['text_secondary'] + """;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .news-content {
            color: """ + theme.COLORS['text_primary'] + """;
            font-size: 0.85rem;
            margin-top: 4px;
            line-height: 1.4;
        }
        .news-sentiment {
            font-size: 0.75rem;
            margin-top: 4px;
            font-weight: 600;
        }
        </style>
        """, unsafe_allow_html=True)
        
        news_html = "<div class='news-container'>"
        
        # Limit to 8 most recent items
        for item in news_items[:8]:
            source = item.get('source', 'Unknown')
            title = item.get('title', '')[:150]  # Truncate long titles
            sent_score = item.get('sentiment_score', 0.0)
            
            # Use CryptoPanic's pre-scored sentiment
            if sent_score > 0.1:
                sent_color = theme.COLORS['positive']
                sent_label = 'Bullish'
                border_color = theme.COLORS['positive']
            elif sent_score < -0.1:
                sent_color = theme.COLORS['negative']
                sent_label = 'Bearish'
                border_color = theme.COLORS['negative']
            else:
                sent_color = theme.COLORS['neutral']
                sent_label = 'Neutral'
                border_color = theme.COLORS['neutral']
            
            news_html += f"""
            <div class='news-item' style='border-left-color: {border_color};'>
                <div class='news-source'>📡 {source}</div>
                <div class='news-content'>{title}</div>
                <div class='news-sentiment' style='color: {sent_color};'>{sent_label}</div>
            </div>
            """
        
        news_html += "</div>"
        st.markdown(news_html, unsafe_allow_html=True)
        
        # News summary stats (using CryptoPanic sentiment directly — no FinBERT needed)
        n_bullish = sum(1 for item in news_items if item.get('sentiment_score', 0) > 0.1)
        n_bearish = sum(1 for item in news_items if item.get('sentiment_score', 0) < -0.1)
        
        st.markdown(f"""
        <div style="text-align: center; margin-top: 8px; color: {theme.COLORS['text_tertiary']}; font-size: 0.8rem;">
            <span style="color: {theme.COLORS['text_secondary']}; font-weight: 600;">{len(news_items)} items</span> · 
            <span style="color: {theme.COLORS['positive']};">{n_bullish} Bullish</span> · 
            <span style="color: {theme.COLORS['negative']};">{n_bearish} Bearish</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='scanning-pattern' style='height:100px;'>
            FETCHING NEWS INTELLIGENCE...
        </div>
        """, unsafe_allow_html=True)
        
except Exception as e:
    st.markdown(f"""
    <div style="background: rgba(248, 113, 113, 0.06); border: 1px solid {theme.COLORS['negative']}30; border-radius: 12px; padding: 16px; text-align: center;">
        <span style="color: {theme.COLORS['negative']}; font-weight: 500;">News feed unavailable: {str(e)[:50]}</span>
    </div>
    """, unsafe_allow_html=True)


# ========== AUTO REFRESH ==========
time.sleep(config.UPDATE_INTERVAL_SEC if hasattr(config, 'UPDATE_INTERVAL_SEC') else 60)
st.rerun()

