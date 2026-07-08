"""
Custom CSS for Streamlit app — AI 全栈可视化教学平台.

Dark neon dashboard design system:
  Background  #0B0F19 (deep navy black)
  Surface     #151B2B (card surface)
  Primary     #00D4FF (neon cyan)
  Accent      #7B61FF (neon purple)
  Success     #00E676 (neon green)
  Warning     #FFAB00 (neon amber)
  Error       #FF5252 (neon red)
  Text        #E2E8F0 (primary), #94A3B8 (secondary), #64748B (muted)

CJK font stack: PingFang SC → Microsoft YaHei → Noto Sans SC → sans-serif
"""

STYLESHEET = """
/* ===== CSS Variables ===== */
:root {
    --bg: #0B0F19;
    --surface: #151B2B;
    --surface-alt: #1A2236;
    --border: rgba(255,255,255,0.08);
    --border-light: rgba(255,255,255,0.04);
    --border-glow: rgba(0,212,255,0.2);
    --primary: #00D4FF;
    --primary-light: #33DBFF;
    --primary-dark: #00A8CC;
    --accent: #7B61FF;
    --accent-light: #9D87FF;
    --success: #00E676;
    --warning: #FFAB00;
    --error: #FF5252;
    --text-primary: #E2E8F0;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.4);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.5);
    --shadow-lg: 0 8px 24px rgba(0,0,0,0.6);
    --glow-primary: 0 0 20px rgba(0,212,255,0.3);
    --glow-accent: 0 0 20px rgba(123,97,255,0.3);
    --glow-success: 0 0 20px rgba(0,230,118,0.3);
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --font-cjk: "PingFang SC", "Microsoft YaHei", "Noto Sans SC", sans-serif;
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
    color: #0B0F19;
}

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
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

/* ===== Page Background ===== */
.block-container {
    padding-top: 5rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

main {
    background-color: var(--bg) !important;
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
    height: 3.5rem;
    background: rgba(11,15,25,0.95);
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
}

.top-nav-tab:hover {
    color: var(--primary);
    background: rgba(0,212,255,0.08);
}

.top-nav-tab.nav-active {
    color: var(--primary);
    background: rgba(0,212,255,0.12);
    box-shadow: inset 0 -2px 0 var(--primary);
}

/* Push page content below fixed nav */
.stTitle {
    display: none !important;
}

/* ===== Metric Glow Cards ===== */
.metric-glow-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0 2rem 0;
    flex-wrap: wrap;
}

.metric-glow {
    flex: 1;
    min-width: 160px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
    transition: all 0.25s ease;
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
    opacity: 0.6;
}

.metric-glow:hover {
    border-color: var(--border-glow);
    box-shadow: var(--glow-primary);
    transform: translateY(-2px);
}

.metric-glow .metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: var(--primary);
    line-height: 1.2;
    margin-bottom: 0.25rem;
}

.metric-glow .metric-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ===== Cards ===== */
.stCard {
    background: var(--surface);
    border-radius: var(--radius-md);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
    transition: all 0.25s ease;
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

/* ===== Page Navigation Cards ===== */
.page-nav-card {
    display: inline-block;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.25rem 1.5rem;
    margin: 0.5rem;
    width: calc(33.333% - 1rem);
    min-width: 240px;
    text-decoration: none;
    color: var(--text-primary);
    box-shadow: var(--shadow-sm);
    transition: all 0.25s ease;
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
    transition: opacity 0.25s ease;
}

.page-nav-card:hover {
    border-color: var(--primary);
    box-shadow: var(--glow-primary);
    transform: translateY(-4px);
}

.page-nav-card:hover::before {
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

/* ===== Welcome Banner ===== */
.welcome-banner {
    background: linear-gradient(135deg, var(--primary-dark) 0%, var(--accent) 100%);
    border-radius: var(--radius-lg);
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    color: white;
    box-shadow: var(--shadow-lg), var(--glow-primary);
    position: relative;
    overflow: hidden;
}

.welcome-banner::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(circle at 80% 20%, rgba(255,255,255,0.1), transparent 60%);
    pointer-events: none;
}

.welcome-banner h1 {
    color: white !important;
    font-size: 1.8rem !important;
    font-weight: 800;
    margin-bottom: 0.5rem;
    text-shadow: 0 0 20px rgba(0,212,255,0.5);
    position: relative;
    z-index: 1;
}

.welcome-banner p {
    color: rgba(255,255,255,0.9) !important;
    font-size: 1.05rem;
    line-height: 1.6;
    position: relative;
    z-index: 1;
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
    background-color: rgba(0,212,255,0.05) !important;
    color: var(--primary-light) !important;
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
    background-color: #0D1117 !important;
    border-radius: var(--radius-sm) !important;
    padding: 1rem !important;
    box-shadow: var(--shadow-sm) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
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
    background-color: rgba(0,212,255,0.06) !important;
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background-color: #0D1220;
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

/* ===== Info / Alert Boxes ===== */
.stAlert {
    border-radius: var(--radius-md) !important;
    border: none !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ===== Section Dividers ===== */
.section-divider {
    border: none;
    border-top: 2px solid var(--border);
    margin: 2rem 0;
}

/* ===== Badge / Tag ===== */
.badge {
    display: inline-block;
    background: var(--primary);
    color: #0B0F19;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    margin-right: 0.5rem;
    vertical-align: middle;
    box-shadow: 0 0 8px rgba(0,212,255,0.3);
}

.badge-accent {
    background: var(--accent);
    color: white;
    box-shadow: 0 0 8px rgba(123,97,255,0.3);
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
    box-shadow: 0 0 10px rgba(0,212,255,0.2);
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
        animation: fadeIn 0.4s ease-out;
    }
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
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
    background: var(--surface);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
}

.config-card h4 {
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
    color: #0B0F19;
    font-size: 0.8rem;
    font-weight: 700;
    margin-right: 0.5rem;
    flex-shrink: 0;
    box-shadow: 0 0 10px rgba(0,212,255,0.3);
}

/* ===== Tech Stack Cards ===== */
.tech-card {
    background: var(--surface);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    text-align: center;
    transition: all 0.2s ease;
}

.tech-card:hover {
    border-color: var(--border-glow);
    box-shadow: var(--glow-primary);
    transform: translateY(-2px);
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
    background: var(--surface);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    transition: all 0.2s ease;
}

.route-card:hover {
    border-color: var(--border-glow);
    box-shadow: var(--glow-primary);
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
    box-shadow: 0 0 10px rgba(0,212,255,0.3);
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
    background: var(--surface);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    height: 100%;
    transition: all 0.2s ease;
}

.comparison-card:hover {
    border-color: var(--border-glow);
    box-shadow: var(--glow-primary);
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
    background: linear-gradient(135deg, rgba(0,212,255,0.05), rgba(123,97,255,0.05));
    border-radius: var(--radius-md);
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
    border-left: 4px solid var(--primary);
    box-shadow: var(--shadow-sm);
}

/* ===== Code Output Block ===== */
.code-output {
    background: #0D1117;
    border-radius: var(--radius-sm);
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace;
    font-size: 0.85rem;
    color: #E6EDF3;
    border: 1px solid rgba(255,255,255,0.1);
}

/* ===== Generated Text Block ===== */
.gen-text-box {
    background: var(--surface);
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
    box-shadow: 0 0 0 2px rgba(0,212,255,0.2) !important;
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
    background-color: var(--primary) !important;
    color: #0B0F19 !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    box-shadow: 0 0 12px rgba(0,212,255,0.3) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    box-shadow: 0 0 20px rgba(0,212,255,0.5) !important;
    transform: translateY(-1px);
}

.stButton > button[kind="secondary"] {
    background-color: var(--surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* Tabs */
.stTabs [role="tab"] {
    color: var(--text-secondary) !important;
    background-color: transparent !important;
}

.stTabs [role="tab"][aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom: 2px solid var(--primary) !important;
    background-color: rgba(0,212,255,0.05) !important;
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

</style>
"""
