/**
 * Icons — Inline SVG icon components
 * ====================================
 * Replaces emoji icons with crisp, accessible SVGs.
 * All icons accept `size` (default 20) and pass through
 * standard SVG/HTML attributes.
 *
 * Based on Lucide icon set (ISC license).
 */
import type { SVGProps } from 'react';

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function base(props: IconProps, d: string | string[]) {
    const { size = 20, ...rest } = props;
    const paths = Array.isArray(d) ? d : [d];
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            width={size}
            height={size}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
            {...rest}
        >
            {paths.map((p, i) => <path key={i} d={p} />)}
        </svg>
    );
}

// ─── Navigation ──────────────────────────────────
/** Zap / Lightning bolt — brand logo & accent */
export function IconZap(props: IconProps) {
    return base(props, 'M13 2L3 14h9l-1 8 10-12h-9l1-8z');
}

/** LayoutDashboard — Dashboard nav */
export function IconDashboard(props: IconProps) {
    return base(props, [
        'M3 3h7v9H3z',
        'M14 3h7v5h-7z',
        'M14 12h7v9h-7z',
        'M3 16h7v5H3z',
    ]);
}

/** TrendingUp — Paper Trading nav */
export function IconTrending(props: IconProps) {
    return base(props, ['M22 7l-8.5 8.5-5-5L2 17', 'M16 7h6v6']);
}

/** Bot / Agent — Nexus Agent nav */
export function IconBot(props: IconProps) {
    return base(props, [
        'M12 8V4H8',
        'M2 14a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v4a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2z',
        'M6 14v4',
        'M9.5 14v1',
        'M14.5 14v1',
        'M18 14v4',
    ]);
}

/** Settings / Gear — Settings nav */
export function IconSettings(props: IconProps) {
    return base(props, [
        'M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z',
        'M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z',
    ]);
}

// ─── Status Bar ──────────────────────────────────
/** Clock — UTC time display */
export function IconClock(props: IconProps) {
    return base(props, ['M12 12V8', 'M12 12l3 3', 'M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z']);
}

/** Monitor — Device indicator */
export function IconMonitor(props: IconProps) {
    return base(props, ['M2 3h20v14H2z', 'M8 21h8', 'M12 17v4']);
}

/** Brain — Model status */
export function IconBrain(props: IconProps) {
    return base(props, [
        'M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z',
        'M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z',
        'M12 5v13',
        'M7.5 7.5L16 16',
        'M16.5 7.5L8 16',
    ]);
}

/** Activity — Positions */
export function IconActivity(props: IconProps) {
    return base(props, 'M22 12h-4l-3 9L9 3l-3 9H2');
}

/** Keyboard — Shortcuts */
export function IconKeyboard(props: IconProps) {
    return base(props, [
        'M2 6a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2z',
        'M6 10h.01', 'M10 10h.01', 'M14 10h.01', 'M18 10h.01',
        'M8 14h8',
    ]);
}

// ─── Card Titles / Content ───────────────────────
/** BarChart3 — Performance/analytics */
export function IconChart(props: IconProps) {
    return base(props, ['M18 20V10', 'M12 20V4', 'M6 20v-6']);
}

/** LineChart — Equity curve / Price chart */
export function IconLineChart(props: IconProps) {
    return base(props, ['M3 3v18h18', 'M7 16l4-8 4 4 4-6']);
}

/** FileText — Trade history */
export function IconList(props: IconProps) {
    return base(props, [
        'M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z',
        'M14 2v6h6',
        'M8 13h8',
        'M8 17h8',
        'M8 9h1',
    ]);
}

/** Newspaper — News feed */
export function IconNews(props: IconProps) {
    return base(props, [
        'M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 0-2 0z',
        'M2 6h4',
        'M2 10h4',
        'M2 14h4',
        'M2 18h4',
        'M10 6h8',
        'M10 10h8',
        'M10 14h4',
    ]);
}

/** Crosshair — Regime / Target */
export function IconCrosshair(props: IconProps) {
    return base(props, [
        'M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z',
        'M22 12h-4', 'M6 12H2', 'M12 6V2', 'M12 22v-4',
    ]);
}

/** Zap (small) — Jump Risk / Energy */
export function IconBolt(props: IconProps) {
    return base(props, 'M13 2L3 14h9l-1 8 10-12h-9l1-8z');
}

/** AlertTriangle — Warning */
export function IconWarning(props: IconProps) {
    return base(props, [
        'M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z',
        'M12 9v4',
        'M12 17h.01',
    ]);
}

/** X — Close / Cancel */
export function IconX(props: IconProps) {
    return base(props, ['M18 6L6 18', 'M6 6l12 12']);
}

/** Play — Start bot */
export function IconPlay(props: IconProps) {
    return base(props, 'M5 3l14 9-14 9V3z');
}

/** Square — Stop bot */
export function IconSquare(props: IconProps) {
    return base(props, 'M3 3h18v18H3z');
}

/** XCircle — Close all */
export function IconXCircle(props: IconProps) {
    return base(props, [
        'M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z',
        'M15 9l-6 6', 'M9 9l6 6',
    ]);
}

/** ArrowUpRight — Long trade */
export function IconArrowUp(props: IconProps) {
    return base(props, ['M7 17L17 7', 'M7 7h10v10']);
}

/** ArrowDownRight — Short trade */
export function IconArrowDown(props: IconProps) {
    return base(props, ['M7 7l10 10', 'M17 7v10H7']);
}

/** Minimize — Window control */
export function IconMinus(props: IconProps) {
    return base(props, 'M5 12h14');
}

/** Maximize — Window control */
export function IconMaximize(props: IconProps) {
    return base(props, 'M3 3h18v18H3z');
}

/** ShieldCheck — System health */
export function IconShield(props: IconProps) {
    return base(props, [
        'M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z',
        'M9 12l2 2 4-4',
    ]);
}

/** CheckCircle — Success */
export function IconCheck(props: IconProps) {
    return base(props, [
        'M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z',
        'M9 12l2 2 4-4',
    ]);
}

/** Info — Info toast */
export function IconInfo(props: IconProps) {
    return base(props, [
        'M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z',
        'M12 16v-4',
        'M12 8h.01',
    ]);
}
