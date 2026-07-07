"""
Custom CSS for Streamlit app.
CJK font support, consistent styling, dark theme hints.
"""

STYLESHEET = """
/* CJK font support */
.stMarkdown, .stText, .stHeader, .stBody, .stSidebar {
    font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans SC", sans-serif;
}

/* Improve code blocks */
pre {
    font-size: 14px !important;
}

/* Make dataframes more readable */
.stDataFrame {
    font-size: 13px !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #f0f2f6;
}

/* Reduce spacing in expander headers */
.streamlit-expanderHeader {
    font-weight: bold;
    font-size: 1.1em;
}

/* Metric cards */
.metric-value {
    font-size: 1.5em !important;
    font-weight: bold;
}

/* Custom tooltip */
[data-testid="stTooltip"] {
    font-style: italic;
}
"""
