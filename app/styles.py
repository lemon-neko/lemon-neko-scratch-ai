"""
Custom CSS for Streamlit app — AI 全栈可视化教学平台.

Design system:
  Primary   #4A90D9 (tech blue)
  Accent    #2EC4B6 (teal green)
  Background #FAFBFC (near-white)
  Card      #FFFFFF
  Border    #E2E8F0 (light gray)
  Text      #1A202C (near-black), #718096 (secondary)

CJK font stack: PingFang SC → Microsoft YaHei → Noto Sans SC → sans-serif
"""

STYLESHEET = """
/* ===== CSS Variables ===== */
:root {
    --primary: #4A90D9;
    --primary-light: #6BA3E0;
    --primary-dark: #3A7BC8;
    --accent: #2EC4B6;
    --accent-light: #5DD9CD;
    --bg: #FAFBFC;
    --card-bg: #FFFFFF;
    --border: #E2E8F0;
    --border-light: #EDF2F7;
    --text-primary: #1A202C;
    --text-secondary: #718096;
    --text-muted: #A0AEC0;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.08);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
    --shadow-lg: 0 8px 24px rgba(0,0,0,0.1);
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

/* Page background */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main content area */
main {
    background-color: var(--bg);
}

/* ===== Title & Headers ===== */
.stTitle {
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.5px;
}

div[data-testid="stHeader"] {
    background-color: rgba(255,255,255,0.85);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--border-light);
}

/* ===== Cards ===== */
.stCard {
    background: var(--card-bg);
    border-radius: var(--radius-md);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-light);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}

.stCard:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}

/* ===== Page Navigation Cards (main.py) ===== */
.page-nav-card {
    display: inline-block;
    background: var(--card-bg);
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
}

.page-nav-card:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-4px);
    border-color: var(--primary);
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
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
    border-radius: var(--radius-lg);
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    color: white;
    box-shadow: var(--shadow-lg);
}

.welcome-banner h1 {
    color: white !important;
    font-size: 1.8rem !important;
    font-weight: 800;
    margin-bottom: 0.5rem;
}

.welcome-banner p {
    color: rgba(255,255,255,0.9) !important;
    font-size: 1.05rem;
    line-height: 1.6;
}

/* ===== Expander Styling ===== */
.streamlit-expanderHeader {
    font-weight: 600;
    font-size: 1.05em;
    color: var(--text-primary);
    border-left: 3px solid var(--primary);
    padding-left: 0.75rem;
    border-radius: var(--radius-sm);
    transition: background-color 0.2s ease;
}

.streamlit-expanderHeader:hover {
    background-color: rgba(74, 144, 217, 0.05);
}

/* ===== Code Blocks ===== */
pre {
    font-size: 13px !important;
    background-color: #1E293B !important;
    border-radius: var(--radius-sm) !important;
    padding: 1rem !important;
    box-shadow: var(--shadow-sm) !important;
    border: 1px solid #334155 !important;
}

pre code {
    color: #E2E8F0 !important;
}

/* ===== DataFrames ===== */
.stDataFrame {
    font-size: 13px !important;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* Zebra striping for dataframes */
[data-testid="stDataFrame"] tbody tr:nth-child(even) {
    background-color: var(--border-light) !important;
}

[data-testid="stDataFrame"] tbody tr:hover {
    background-color: rgba(74, 144, 217, 0.06) !important;
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background-color: #F1F5F9;
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

/* ===== Metric Cards ===== */
.metric-value {
    font-size: 1.5em !important;
    font-weight: 700;
    color: var(--primary) !important;
}

/* ===== Info / Warning / Success Boxes ===== */
.stAlert {
    border-radius: var(--radius-md) !important;
    border: none !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ===== Section Dividers ===== */
.section-divider {
    border: none;
    border-top: 2px solid var(--border-light);
    margin: 2rem 0;
}

/* ===== Badge / Tag ===== */
.badge {
    display: inline-block;
    background: var(--primary);
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    margin-right: 0.5rem;
    vertical-align: middle;
}

.badge-accent {
    background: var(--accent);
}

.badge-outline {
    background: transparent;
    color: var(--primary);
    border: 1px solid var(--primary);
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
    background: var(--card-bg);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-light);
}

.config-card h4 {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-light);
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
    color: white;
    font-size: 0.8rem;
    font-weight: 700;
    margin-right: 0.5rem;
    flex-shrink: 0;
}

/* ===== Tech Stack Cards ===== */
.tech-card {
    background: var(--card-bg);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    text-align: center;
    transition: all 0.2s ease;
}

.tech-card:hover {
    box-shadow: var(--shadow-md);
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
    background: var(--card-bg);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    border: 1px solid var(--border-light);
    box-shadow: var(--shadow-sm);
    transition: all 0.2s ease;
}

.route-card:hover {
    box-shadow: var(--shadow-md);
    border-color: var(--primary);
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
    background: var(--card-bg);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    height: 100%;
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
    background: linear-gradient(135deg, rgba(74,144,217,0.05), rgba(46,196,182,0.05));
    border-radius: var(--radius-md);
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
    border-left: 4px solid var(--primary);
}

/* ===== Code Output Block ===== */
.code-output {
    background: #1E293B;
    border-radius: var(--radius-sm);
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace;
    font-size: 0.85rem;
    color: #E2E8F0;
    border: 1px solid #334155;
}

/* ===== Generated Text Block ===== */
.gen-text-box {
    background: var(--card-bg);
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
"""
