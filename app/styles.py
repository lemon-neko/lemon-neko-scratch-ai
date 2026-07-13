"""
Custom CSS for Streamlit app — AI 全栈可视化教学平台.

Dark cyberpunk dashboard design system:
  Background  #0a0e1a (deep navy black)
  Surface     #151f32 (card surface)
  Primary     #00d4aa (teal-cyan)
  Accent      #7c3aed (purple)
  Success     #22c55e (green)
  Warning     #f59e0b (amber)
  Error       #ef4444 (red)
  Text        #E2E8F0 (primary), #94A3B8 (secondary), #64748B (muted)

CJK font stack: Inter → PingFang SC → Microsoft YaHei → Noto Sans SC → system-ui
"""

STYLESHEET = """
/* ===== CSS Variables ===== */
:root {
    --bg: #0a0e1a;
    --bg-deep: #060a14;
    --bg-elevated: #0d1220;
    --surface: #151f32;
    --surface-alt: #1a2640;
    --border: rgba(255,255,255,0.06);
    --border-light: rgba(255,255,255,0.04);
    --border-glow: rgba(0,212,170,0.25);
    --primary: #00d4aa;
    --primary-light: #33e0bf;
    --primary-dark: #00a888;
    --accent: #7c3aed;
    --accent-light: #a78bfa;
    --color-cyan: #06b6d4;
    --color-blue: #3b82f6;
    --color-purple: #8b5cf6;
    --color-orange: #f97316;
    --color-pink: #ec4899;
    --success: #22c55e;
    --warning: #f59e0b;
    --error: #ef4444;
    --info: #06b6d4;
    --text-primary: #E2E8F0;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
    --shadow-md: 0 2px 8px rgba(0,0,0,0.05);
    --shadow-lg: 0 8px 24px rgba(0,0,0,0.1);
    --glow-primary: 0 0 20px rgba(0,212,170,0.15);
    --glow-accent: 0 0 20px rgba(124,58,237,0.15);
    --glow-success: 0 0 20px rgba(34,197,94,0.15);
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --nav-height: 56px;
    --font-cjk: "Inter", "PingFang SC", "Microsoft YaHei", "Noto Sans SC", system-ui, sans-serif;
    --font-mono: "JetBrains Mono", "Fira Code", "Consolas", monospace;
}

/* ===== Hide Streamlit Default Sidebar & Top Elements ===== */
section[data-testid="stSidebar"] {
    display: none !important;
}
section[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}
header[data-testid="stHeader"] {
    display: none !important;
}
header[data-testid="stTopHeader"] {
    display: none !important;
}
div[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

/* Ensure main content uses full width */
section[data-testid="stMain"] {
    padding-top: calc(var(--nav-height) + 16px) !important;
}

/* ===== Global Typography ===== */
.stMarkdown, .stText, .stHeader, .stBody, .stSidebar,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMarkdownContainer"] h4,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] span {
    font-family: var(--font-cjk);
    color: var(--text-primary);
}

::selection {
    background: var(--primary);
    color: #0a0e1a;
}

::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg);
}
::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted);
}

/* ===== Page Background with subtle grid ===== */
main {
    background-color: var(--bg) !important;
    background-image:
        linear-gradient(rgba(0,212,170,0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,170,0.015) 1px, transparent 1px);
    background-size: 40px 40px;
}

/* ===== Top Navigation Bar ===== */
.top-nav-bar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2rem;
    height: var(--nav-height);
    background: rgba(10,14,26,0.95);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
}

.top-nav-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    white-space: nowrap;
}

.top-nav-brand .brand-icon {
    font-size: 1.4rem;
}

.top-nav-tabs {
    display: flex;
    gap: 0.25rem;
    align-items: center;
    overflow-x: auto;
    scrollbar-width: none;
    -ms-overflow-style: none;
    max-width: 65%;
}

.top-nav-tabs::-webkit-scrollbar {
    display: none;
}

.top-nav-tab {
    display: inline-flex;
    align-items: center;
    padding: 0.4rem 1rem;
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text-secondary);
    text-decoration: none;
    border-radius: var(--radius-sm);
    transition: all 0.2s ease;
    white-space: nowrap;
    position: relative;
}

.top-nav-tab::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%) scaleX(0);
    width: 60%;
    height: 2px;
    background: var(--primary);
    border-radius: 1px;
    transition: transform 0.25s ease;
}

.top-nav-tab:hover {
    color: var(--primary);
    background: rgba(0,212,170,0.08);
}

.top-nav-tab:hover::after {
    transform: translateX(-50%) scaleX(0.5);
    opacity: 0.5;
}

.top-nav-tab.nav-active {
    color: var(--primary);
    background: rgba(0,212,170,0.1);
}

.top-nav-tab.nav-active::after {
    transform: translateX(-50%) scaleX(1);
    opacity: 1;
    box-shadow: 0 0 8px rgba(0,212,170,0.5);
}

.top-nav-tab:focus-visible {
    outline: 2px solid var(--primary);
    outline-offset: 2px;
    border-radius: var(--radius-sm);
}

/* Top nav search container */
.top-nav-search {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-left: 1rem;
}

.top-nav-search input {
    background: rgba(21,31,50,0.8);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.35rem 0.75rem;
    font-size: 0.8rem;
    color: var(--text-primary);
    outline: none;
    transition: border-color 0.2s ease;
    width: 180px;
}

.top-nav-search input:focus {
    border-color: var(--primary);
    box-shadow: 0 0 0 2px rgba(0,212,170,0.15);
}

.top-nav-search input::placeholder {
    color: var(--text-muted);
}

/* Push page content below fixed nav */
.stTitle {
    display: none !important;
}

/* ===== Section Titles (unified type scale) ===== */
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 2rem 0 1rem 0;
    line-height: 1.3;
    position: relative;
    padding-left: 1rem;
}

.section-title::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0.15em;
    bottom: 0.15em;
    width: 3px;
    border-radius: 2px;
    background: linear-gradient(180deg, var(--primary), var(--accent));
}

.section-title--lg {
    font-size: 1.5rem;
}

.section-title--sm {
    font-size: 1.1rem;
}

/* ===== Metric Glow Cards ===== */
.metric-glow-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0 2.5rem 0;
    flex-wrap: wrap;
}

.metric-glow {
    flex: 1;
    min-width: 160px;
    background: linear-gradient(135deg, var(--surface), rgba(21,31,50,0.8));
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.metric-glow::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--primary), var(--accent));
    opacity: 0;
    transition: opacity 0.3s ease;
}

.metric-glow:hover {
    border-color: var(--border-glow);
    box-shadow: var(--glow-primary);
    transform: translateY(-3px);
}

.metric-glow:hover::before {
    opacity: 1;
}

.metric-glow .metric-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary-light), var(--accent-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.25rem;
}

.metric-glow .metric-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ===== Metric Row Items ===== */
.metric-item {
    flex: 1;
    min-width: 140px;
    background: linear-gradient(135deg, var(--surface), rgba(21,31,50,0.8));
    border-radius: var(--radius-md);
    border: 1px solid var(--border);
    padding: 1.25rem;
    box-shadow: var(--shadow-sm);
    text-align: center;
    transition: all 0.25s ease;
}

.metric-item:hover {
    border-color: var(--border-glow);
    box-shadow: var(--glow-primary);
}

.metric-item-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}

.metric-item-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--primary);
}

.metric-item-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 0.5rem 0 1rem 0;
}

/* ===== Cards ===== */
.stCard {
    background: linear-gradient(135deg, var(--surface), rgba(21,31,50,0.9));
    border-radius: var(--radius-md);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
    transition: all 0.3s ease;
}

.stCard:hover {
    border-color: var(--border-glow);
    box-shadow: var(--glow-primary);
    transform: translateY(-2px);
}

.stCard .glow-border {
    border: 1px solid var(--border-glow);
    box-shadow: var(--glow-primary);
}

.stCard-header {
    font-weight: 700;
    font-size: 1rem;
    color: var(--text-primary);
    margin-bottom: 0.75rem;
}

/* ===== Page Navigation Grid ===== */
.page-nav-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin: 1.5rem 0 2rem 0;
}

/* ===== Page Navigation Cards ===== */
.page-nav-card {
    display: inline-block;
    background: linear-gradient(135deg, var(--surface), rgba(21,31,50,0.9));
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.25rem 1.5rem;
    margin: 0.5rem;
    width: calc(33.333% - 1rem);
    min-width: 240px;
    text-decoration: none;
    color: var(--text-primary);
    box-shadow: var(--shadow-sm);
    transition: all 0.3s ease;
    vertical-align: top;
    cursor: pointer;
    position: relative;
    overflow: hidden;
}

.page-nav-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--primary), var(--accent));
    opacity: 0;
    transition: opacity 0.3s ease;
}

.page-nav-card::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at var(--mx, 50%) var(--my, 50%), rgba(0,212,170,0.06), transparent 60%);
    opacity: 0;
    transition: opacity 0.3s ease;
    pointer-events: none;
}

.page-nav-card:hover {
    border-color: var(--primary);
    box-shadow: var(--glow-primary);
    transform: translateY(-4px);
}

.page-nav-card:hover::before {
    opacity: 1;
}

.page-nav-card:hover::after {
    opacity: 1;
}

.page-nav-card .nav-icon {
    font-size: 2rem;
    display: block;
    margin-bottom: 0.5rem;
}

.page-nav-card .nav-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
}

.page-nav-card .nav-desc {
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.4;
}

/* ===== Welcome Banner — Animated Gradient ===== */
.welcome-banner {
    background: linear-gradient(-45deg, #00a888, #00d4aa, #7c3aed, #06b6d4);
    background-size: 400% 400%;
    animation: bannerGradient 12s ease infinite;
    border-radius: var(--radius-lg);
    padding: 2.5rem 3rem;
    margin-bottom: 2.5rem;
    color: white;
    box-shadow: 0 8px 32px rgba(0,212,170,0.15), 0 2px 8px rgba(0,0,0,0.3);
    position: relative;
    overflow: hidden;
}

@keyframes bannerGradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.welcome-banner::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse at 20% 80%, rgba(124,58,237,0.3), transparent 50%),
        radial-gradient(ellipse at 80% 20%, rgba(0,212,170,0.2), transparent 50%);
    pointer-events: none;
}

.welcome-banner::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.05), transparent 70%);
    pointer-events: none;
    animation: bannerFloat 8s ease-in-out infinite;
}

@keyframes bannerFloat {
    0%, 100% { transform: translate(0, 0) rotate(0deg); }
    50% { transform: translate(-30px, 20px) rotate(10deg); }
}

.welcome-banner h1 {
    color: white !important;
    font-size: 2rem !important;
    font-weight: 800;
    margin-bottom: 0.75rem;
    text-shadow: 0 0 30px rgba(0,0,0,0.3);
    position: relative;
    z-index: 1;
    letter-spacing: -0.02em;
}

.welcome-banner p {
    color: rgba(255,255,255,0.92) !important;
    font-size: 1.1rem;
    line-height: 1.7;
    position: relative;
    z-index: 1;
    max-width: 60ch;
}

/* ===== Expander Styling ===== */
.streamlit-expanderHeader {
    font-weight: 600;
    font-size: 1.05em;
    color: var(--text-primary);
    background-color: var(--surface) !important;
    border-left: 3px solid var(--primary) !important;
    padding-left: 0.75rem;
    border-radius: var(--radius-sm);
    transition: all 0.2s ease;
}

.streamlit-expanderHeader:hover {
    background-color: rgba(0,212,170,0.05) !important;
    color: var(--primary-light) !important;
    border-left-color: var(--primary-light) !important;
}

.streamlit-expanderHeader:hover + .streamlit-expanderContent,
.streamlit-expanderHeader:hover ~ div {
    background-color: var(--surface) !important;
}

.streamlit-expanderContent {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
}

/* ===== Code Blocks ===== */
pre {
    font-size: 13px !important;
    background-color: #060a14 !important;
    border-radius: var(--radius-sm) !important;
    padding: 1rem !important;
    box-shadow: var(--shadow-sm) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

pre code {
    color: #E6EDF3 !important;
}

/* ===== DataFrames ===== */
.stDataFrame {
    font-size: 13px !important;
    border-radius: var(--radius-sm);
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
    color: var(--text-primary) !important;
}

/* Zebra striping for dataframes */
[data-testid="stDataFrame"] tbody tr:nth-child(even) {
    background-color: var(--surface-alt) !important;
}

[data-testid="stDataFrame"] tbody tr:hover {
    background-color: rgba(0,212,170,0.06) !important;
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background-color: #0d1220;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Sidebar header */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5 {
    color: var(--text-primary) !important;
}

/* Sidebar slider labels */
section[data-testid="stSidebar"] label {
    font-weight: 600;
    color: var(--text-primary);
}

/* ===== Metric Cards (st.metric) ===== */
.stMetric {
    background-color: var(--surface) !important;
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border) !important;
    padding: 0.75rem 1rem !important;
}

.stMetric > div {
    color: var(--text-primary) !important;
}

.metric-value {
    font-size: 1.5em !important;
    font-weight: 700;
    color: var(--primary) !important;
}

/* ===== Styled Info Boxes ===== */
.info-box {
    background: rgba(0,212,170,0.06);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    border-left: 4px solid var(--primary);
    transition: all 0.2s ease;
}

.info-box:hover {
    background: rgba(0,212,170,0.08);
}

.info-box--warning {
    background: rgba(245,158,11,0.06);
    border-left-color: var(--warning);
}

.info-box--warning:hover {
    background: rgba(245,158,11,0.08);
}

.info-box--success {
    background: rgba(34,197,94,0.06);
    border-left-color: var(--success);
}

.info-box--success:hover {
    background: rgba(34,197,94,0.08);
}

.info-box--error {
    background: rgba(239,68,68,0.06);
    border-left-color: var(--error);
}

.info-box--error:hover {
    background: rgba(239,68,68,0.08);
}

.info-box-title {
    font-weight: 700;
    color: var(--primary-light);
}

.info-box--warning .info-box-title { color: var(--warning); }
.info-box--success .info-box-title { color: var(--success); }
.info-box--error .info-box-title { color: var(--error); }

.info-box-body {
    color: var(--text-primary);
    margin-top: 0.5rem;
    line-height: 1.6;
}

/* ===== Info / Alert Boxes ===== */
.stAlert {
    border-radius: var(--radius-md) !important;
    border: none !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ===== Section Dividers ===== */
.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2.5rem 0;
}

/* ===== Badge / Tag ===== */
.badge {
    display: inline-block;
    background: var(--primary);
    color: #0a0e1a;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    margin-right: 0.5rem;
    vertical-align: middle;
    box-shadow: 0 0 8px rgba(0,212,170,0.15);
}

.badge-accent {
    background: var(--accent);
    color: white;
    box-shadow: 0 0 8px rgba(124,58,237,0.15);
}

.badge-outline {
    background: transparent;
    color: var(--primary);
    border: 1px solid var(--primary);
    box-shadow: none;
}

.badge-neon {
    background: transparent;
    color: var(--primary);
    border: 1px solid var(--primary);
    box-shadow: 0 0 10px rgba(0,212,170,0.15);
}

/* ===== Status Dot ===== */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 0.5rem;
    vertical-align: middle;
    box-shadow: 0 0 6px currentColor;
}

.status-dot.online {
    background: var(--success);
    color: var(--success);
}

.status-dot.offline {
    background: var(--text-muted);
    color: var(--text-muted);
}

.status-dot.warning {
    background: var(--warning);
    color: var(--warning);
}

.status-dot.error {
    background: var(--error);
    color: var(--error);
}

/* ===== Page Fade-in Animation ===== */
@media (prefers-reduced-motion: no-preference) {
    .block-container {
        animation: fadeIn 0.5s ease-out;
    }
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ===== Footer ===== */
.app-footer {
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border-light);
    text-align: center;
    color: var(--text-muted);
    font-size: 0.85rem;
}

/* ===== Sidebar Config Card ===== */
.config-card {
    background: linear-gradient(135deg, var(--surface), rgba(21,31,50,0.9));
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
}

.config-card h4.config-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* ===== Step Number Badge ===== */
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: var(--primary);
    color: #0a0e1a;
    font-size: 0.8rem;
    font-weight: 700;
    margin-right: 0.5rem;
    flex-shrink: 0;
    box-shadow: 0 0 10px rgba(0,212,170,0.15);
}

/* ===== Tech Stack Cards ===== */
.tech-card {
    background: linear-gradient(135deg, var(--surface), rgba(21,31,50,0.9));
    border-radius: var(--radius-md);
    padding: 1.25rem;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    text-align: center;
    transition: all 0.3s ease;
}

.tech-card:hover {
    border-color: var(--border-glow);
    box-shadow: var(--glow-primary);
    transform: translateY(-3px);
}

.tech-card .tech-icon {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    display: block;
}

.tech-card .tech-name {
    font-weight: 700;
    font-size: 1rem;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
}

.tech-card .tech-desc {
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ===== Learning Route Card ===== */
.route-card {
    display: flex;
    align-items: flex-start;
    background: linear-gradient(135deg, var(--surface), rgba(21,31,50,0.9));
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    transition: all 0.3s ease;
}

.route-card:hover {
    border-color: var(--border-glow);
    box-shadow: var(--glow-primary);
    transform: translateX(4px);
}

.route-card .route-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    color: white;
    font-weight: 700;
    font-size: 0.9rem;
    margin-right: 1rem;
    flex-shrink: 0;
    box-shadow: 0 0 10px rgba(0,212,170,0.15);
}

.route-card .route-content {
    flex: 1;
}

.route-card .route-title {
    font-weight: 700;
    font-size: 1rem;
    color: var(--text-primary);
}

.route-card .route-desc {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
}

/* ===== Comparison Cards (Encoder vs Decoder) ===== */
.comparison-card {
    background: linear-gradient(135deg, var(--surface), rgba(21,31,50,0.9));
    border-radius: var(--radius-md);
    padding: 1.25rem;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    height: 100%;
    transition: all 0.3s ease;
}

.comparison-card:hover {
    border-color: var(--border-glow);
    box-shadow: var(--glow-primary);
    transform: translateY(-2px);
}

.comparison-card h4 {
    font-weight: 700;
    color: var(--primary);
    margin-bottom: 0.75rem;
}

.comparison-card ul {
    padding-left: 1.25rem;
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.8;
}

/* ===== Info Panel ===== */
.info-panel {
    background: linear-gradient(135deg, rgba(0,212,170,0.05), rgba(124,58,237,0.05));
    border-radius: var(--radius-md);
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
    border-left: 4px solid var(--primary);
    box-shadow: var(--shadow-sm);
    transition: all 0.2s ease;
}

.info-panel:hover {
    background: linear-gradient(135deg, rgba(0,212,170,0.07), rgba(124,58,237,0.07));
}

/* ===== Code Output Block ===== */
.code-output {
    background: #060a14;
    border-radius: var(--radius-sm);
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace;
    font-size: 0.85rem;
    color: #E6EDF3;
    border: 1px solid rgba(255,255,255,0.06);
}

/* ===== Generated Text Block ===== */
.gen-text-box {
    background: linear-gradient(135deg, var(--surface), rgba(21,31,50,0.9));
    border: 1px solid var(--accent);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    box-shadow: var(--shadow-sm);
}

.gen-text-box .gen-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}

.gen-text-box .gen-content {
    font-size: 1.1rem;
    color: var(--text-primary);
    line-height: 1.6;
}

/* ===== Streamlit Native Element Overrides ===== */
/* Streamlit heading overrides */
.stMarkdown h1,
div[data-testid="stMarkdownContainer"] h1 {
    color: var(--text-primary) !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
}

.stMarkdown h2,
div[data-testid="stMarkdownContainer"] h2 {
    color: var(--text-primary) !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}

.stMarkdown h3,
div[data-testid="stMarkdownContainer"] h3 {
    color: var(--text-primary) !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
}

.stMarkdown h4,
div[data-testid="stMarkdownContainer"] h4 {
    color: var(--text-primary) !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}


/* Text inputs */
.stTextInput > div > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div,
.stNumberInput > div > div > div > input {
    background-color: var(--surface) !important;
    color: var(--text-primary) !important;
    border-color: var(--border) !important;
    box-shadow: none !important;
}

.stTextInput > div > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
.stNumberInput > div > div > div > input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(0,212,170,0.15) !important;
}

.stTextInput > div > div > div > input:focus-visible,
.stTextArea > div > div > textarea:focus-visible,
.stNumberInput > div > div > div > input:focus-visible {
    outline: none !important;
}

/* Selectbox dropdown */
.stSelectbox > div {
    background-color: var(--surface) !important;
    color: var(--text-primary) !important;
}

/* Radio buttons */
.stRadio > label {
    color: var(--text-primary) !important;
}

/* Slider */
.stSlider > div > div > div > div {
    background-color: var(--surface) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    box-shadow: 0 0 12px rgba(0,212,170,0.15) !important;
    transition: all 0.25s ease !important;
}

.stButton > button:hover {
    box-shadow: 0 0 24px rgba(0,212,170,0.25) !important;
    transform: translateY(-1px) !important;
    filter: brightness(1.1);
}

.stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 0 8px rgba(0,212,170,0.15) !important;
}

.stButton > button[kind="secondary"] {
    background-color: var(--surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}

.stButton > button:focus-visible {
    outline: 2px solid var(--primary) !important;
    outline-offset: 2px !important;
}

/* Tabs */
.stTabs [role="tab"] {
    color: var(--text-secondary) !important;
    background-color: transparent !important;
}

.stTabs [role="tab"]:focus-visible {
    outline: 2px solid var(--primary) !important;
    outline-offset: 2px !important;
}

.stTabs [role="tab"][aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom: 2px solid var(--primary) !important;
    background-color: rgba(0,212,170,0.05) !important;
}

/* Divider */
.stDivider hr {
    border-color: var(--border) !important;
}

/* Caption */
.stCaption {
    color: var(--text-muted) !important;
}

/* Progress bar */
[data-testid="stProgress"] > div > div > div {
    background-color: var(--surface-alt) !important;
}

/* Checkbox */
.stCheckbox > label {
    color: var(--text-primary) !important;
}

/* Download button */
.stDownloadButton > button {
    background-color: var(--surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}

/* Plotly chart containers */
.js-plotly-plot .plotly .modebar {
    top: 4px !important;
    right: 4px !important;
}

.js-plotly-plot .plotly .modebar-btn {
    background: rgba(21, 31, 50, 0.8) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* Streamlit sidebar header */
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
}

/* Streamlit sidebar elements */
section[data-testid="stSidebar"] .stTextInput > div > div > div > input,
section[data-testid="stSidebar"] .stTextArea > div > div > textarea,
section[data-testid="stSidebar"] .stSelectbox > div > div > div,
section[data-testid="stSidebar"] .stSlider > div > div > div > div,
section[data-testid="stSidebar"] .stNumberInput > div > div > div > input {
    background-color: var(--surface) !important;
    color: var(--text-primary) !important;
    border-color: var(--border) !important;
}

section[data-testid="stSidebar"] .stSlider > div > div > div > div > div {
    background-color: var(--primary) !important;
}

/* Streamlit sidebar selectbox */
section[data-testid="stSidebar"] .stSelectbox > div > div > div {
    background-color: var(--surface) !important;
    color: var(--text-primary) !important;
}

/* Sidebar radio */
section[data-testid="stSidebar"] .stRadio > label {
    color: var(--text-primary) !important;
}

/* Sidebar metrics */
section[data-testid="stSidebar"] .stMetric {
    background-color: var(--surface) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.5rem 0.75rem !important;
}

/* ===== Three Column Layout ===== */
.app-layout-three-col {
    display: flex;
    width: 100%;
    min-height: calc(100vh - var(--nav-height, 56px));
    margin-top: var(--nav-height, 56px);
}

.app-left-panel {
    width: 240px;
    min-width: 240px;
    background: #0d1220;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
    overflow-y: auto;
    position: sticky;
    top: var(--nav-height, 56px);
    height: calc(100vh - var(--nav-height, 56px));
}

.app-right-panel {
    width: 300px;
    min-width: 300px;
    background: #0d1220;
    border-left: 1px solid rgba(255, 255, 255, 0.06);
    overflow-y: auto;
    position: sticky;
    top: var(--nav-height, 56px);
    height: calc(100vh - var(--nav-height, 56px));
}

.app-center-content {
    flex: 1;
    min-width: 0;
    padding: 24px;
    overflow-y: auto;
}

/* ===== Streamlit Columns 3-Column Layout Adapter ===== */
/* When using st.columns([1, 4, 1]), style the columns to look like panels */
section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] {
    gap: 0 !important;
}

/* Left column (params panel) */
section[data-testid="stMain"] > div > div > div > div[data-testid="stColumn"]:first-child {
    background: #0d1220;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
    padding: 1rem !important;
    max-width: 280px;
    min-width: 240px;
    overflow-y: auto;
}

/* Right column (reference panel) */
section[data-testid="stMain"] > div > div > div > div[data-testid="stColumn"]:last-child {
    background: #0d1220;
    border-left: 1px solid rgba(255, 255, 255, 0.06);
    padding: 1rem !important;
    max-width: 320px;
    min-width: 260px;
    overflow-y: auto;
}

/* Center column (main content) */
section[data-testid="stMain"] > div > div > div > div[data-testid="stColumn"]:nth-child(2) {
    padding: 1.5rem !important;
    flex: 4;
    min-width: 0;
}

/* Left/Right panel inner styling */
.app-left-panel-inner h4,
.app-left-panel-inner h3,
.app-right-panel-inner h4,
.app-right-panel-inner h3 {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    font-weight: 600;
}

.app-left-panel-inner .stSelectbox,
.app-left-panel-inner .stSlider,
.app-left-panel-inner .stTextInput,
.app-left-panel-inner .stTextArea,
.app-left-panel-inner .stRadio {
    margin-bottom: 0.5rem;
}

/* ===== Terminal-style Log Panel ===== */
.log-panel {
    background: #060a14;
    border-radius: 8px;
    padding: 12px;
    font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace;
    font-size: 0.8rem;
    color: #64748b;
    max-height: 200px;
    overflow-y: auto;
    line-height: 1.8;
    border: 1px solid rgba(255, 255, 255, 0.06);
}

.log-entry-time {
    color: #00d4aa;
}

.log-entry-success {
    color: #22c55e;
}

.log-entry-warning {
    color: #f59e0b;
}

/* ===== Tip Card (Right Panel) ===== */
.tip-card {
    display: flex;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
}

.tip-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    margin-top: 6px;
    flex-shrink: 0;
}

.tip-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 2px;
}

.tip-body {
    font-size: 0.75rem;
    color: #64748b;
    line-height: 1.4;
}

.tip-time {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 4px;
}

/* ===== Responsive Breakpoints ===== */
@media (max-width: 768px) {
    .top-nav-bar { padding: 0 1rem; height: 3rem; }
    .top-nav-tabs { max-width: 70%; gap: 0.15rem; }
    .top-nav-tab { padding: 0.35rem 0.7rem; font-size: 0.8rem; }
    .block-container { padding-top: 4rem; }
    .welcome-banner { padding: 1.5rem; margin-bottom: 1.5rem; }
    .welcome-banner h1 { font-size: 1.4rem !important; }
    .metric-glow-row { gap: 0.75rem; }
    .metric-glow { min-width: calc(50% - 0.75rem); }
    .page-nav-card { width: calc(100% - 1rem); }
    .app-layout-three-col { flex-direction: column; }
    .app-left-panel,
    .app-right-panel {
        width: 100%;
        min-width: 100%;
        position: static;
        height: auto;
    }
    .top-nav-search { display: none; }
}

@media (max-width: 480px) {
    .top-nav-brand span:last-child { display: none; }
    .top-nav-tab { padding: 0.3rem 0.5rem; font-size: 0.75rem; }
    .metric-glow { min-width: 100%; }
}

"""
