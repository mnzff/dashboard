import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

st.set_page_config(
    page_title="Actuarial Triangle Viewer",
    page_icon="△",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={},
)

# ── Session state ──────────────────────────────────────────────────────────────
if "show_proj" not in st.session_state:
    st.session_state.show_proj = True
if "file_content" not in st.session_state:
    st.session_state.file_content = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None

# ── Light-mode theme tokens ────────────────────────────────────────────────────
T = dict(
    bg="#ffffff", surface="#f8fafc", surface2="#f1f5f9",
    border="#e2e8f0", text="#1e293b", muted="#64748b",
    plot_bg="#ffffff", paper_bg="#ffffff",
    grid="#e5e7eb", axis_color="#1e293b",
    divider="#e2e8f0",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 0.5rem !important; padding-bottom: 2rem !important; }

    /* Hide sidebar entirely */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    button[kind="header"] { display: none !important; }

    /* Metrics */
    [data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: #64748b !important; }
    [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    [data-testid="metric-container"] {
        background-color: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 0.6rem 0.8rem !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-size: 0.92rem !important;
        background-color: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
    }
    .streamlit-expanderContent {
        background-color: #f1f5f9 !important;
        border: 1px solid #e2e8f0 !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
    }

    /* Buttons */
    [data-testid="stButton"] button,
    [data-testid="stBaseButton-secondary"],
    [data-testid="stBaseButton-primary"] {
        background-color: #f8fafc !important;
        color: #1e293b !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
    }
    [data-testid="stButton"] button:hover {
        background-color: #f1f5f9 !important;
        border-color: #94a3b8 !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploaderDropzone"] {
        background-color: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CHART_COLORS = [
    "#3b82f6", "#10b981", "#f59e0b", "#ef4444",
    "#8b5cf6", "#06b6d4", "#f97316", "#84cc16",
    "#ec4899", "#14b8a6",
]

BLUE_SCHEME = {"lo": (219, 234, 254), "hi": (37, 99, 235)}


# ── Data helpers ───────────────────────────────────────────────────────────────

# Parse raw CSV content into triangle labels and numeric values.
# Expects first column to contain origin period labels, and remaining
# columns to contain development period values.
def parse_triangle(content: str):
    df = pd.read_csv(io.StringIO(content), header=0)
    row_headers = df.iloc[:, 0].astype(str).tolist()
    col_headers = [str(c).strip() for c in df.columns[1:]]
    # Convert all remaining cells to numeric, invalid values become NaN.
    raw = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").values.astype(float)
    return row_headers, col_headers, raw


# Compute chain-ladder development factors, project missing cells,
# and derive ultimate, latest observed, and IBNR values.
def compute_chain_ladder(values: np.ndarray):
    n_rows, n_cols = values.shape

    # Calculate average development factors for each transition.
    dev_factors = []
    for j in range(n_cols - 1):
        num, den = 0.0, 0.0
        for i in range(n_rows):
            c, nx = values[i, j], values[i, j + 1]
            # Only use rows where both current and next period are observed
            # and current period is non-zero.
            if not (np.isnan(c) or np.isnan(nx)) and c != 0:
                num += nx
                den += c
        dev_factors.append(num / den if den > 0 else np.nan)

    # Copy the original triangle and fill missing cells with projections.
    projected = values.copy()
    is_projected = np.zeros((n_rows, n_cols), dtype=bool)
    for i in range(n_rows):
        for j in range(1, n_cols):
            if np.isnan(projected[i, j]):
                prev = projected[i, j - 1]
                f = dev_factors[j - 1]
                # Only project if the previous cell exists and the factor is valid.
                if not np.isnan(prev) and not np.isnan(f):
                    projected[i, j] = prev * f
                    is_projected[i, j] = True

    # Ultimates = last column of the projected triangle.
    ultimates = projected[:, -1].copy()

    # Find the latest observed value in each row by scanning backward.
    latest_obs = []
    for i in range(n_rows):
        for j in range(n_cols - 1, -1, -1):
            if not np.isnan(values[i, j]):
                latest_obs.append(values[i, j])
                break
        else:
            latest_obs.append(np.nan)

    # IBNR = ultimate minus latest observed reported value.
    ibnr = np.array([
        u - o if not (np.isnan(u) or np.isnan(o)) else np.nan
        for u, o in zip(ultimates, latest_obs)
    ])
    return dev_factors, projected, is_projected, ultimates, np.array(latest_obs), ibnr


# Return the last non-missing development column index for each row.
def latest_obs_col_idx(raw_values):
    result = []
    n_rows, n_cols = raw_values.shape
    for i in range(n_rows):
        last_j = -1
        for j in range(n_cols - 1, -1, -1):
            if not np.isnan(raw_values[i, j]):
                last_j = j
                break
        result.append(last_j)
    return result


# Format numbers for display, with dashes for missing values and
# compact formatting for thousands/millions.
def fmt(n):
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "—"
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{n:,.0f}"
    return f"{n:.2f}"


# Linearly interpolate between two RGB colors by a fraction t.
# Returns a CSS rgb string and the resulting integer RGB tuple.
def lerp_color(t, lo_rgb, hi_rgb):
    r = int(lo_rgb[0] + t * (hi_rgb[0] - lo_rgb[0]))
    g = int(lo_rgb[1] + t * (hi_rgb[1] - lo_rgb[1]))
    b = int(lo_rgb[2] + t * (hi_rgb[2] - lo_rgb[2]))
    return f"rgb({r},{g},{b})", (r, g, b)

# ── Chart helpers ──────────────────────────────────────────────────────────────

def _base_layout(**extra):
    base = dict(
        plot_bgcolor=T["plot_bg"],
        paper_bgcolor=T["paper_bg"],
        font=dict(color=T["text"], size=11),
        xaxis=dict(
            showgrid=True, gridcolor=T["grid"],
            tickfont=dict(color=T["axis_color"]),
            title_font=dict(color=T["axis_color"]),
            linecolor=T["border"], zerolinecolor=T["border"],
        ),
        yaxis=dict(
            showgrid=True, gridcolor=T["grid"],
            tickfont=dict(color=T["axis_color"]),
            title_font=dict(color=T["axis_color"]),
            linecolor=T["border"], zerolinecolor=T["border"],
        ),
        legend=dict(
            font=dict(color=T["text"]),
            bgcolor=T["surface"],
            bordercolor=T["border"],
            borderwidth=1,
        ),
    )
    base.update(extra)
    return base


def make_single_dev_curve(row_idx, row_headers, col_headers, values, projected, is_projected):
    label = row_headers[row_idx]
    color = CHART_COLORS[row_idx % len(CHART_COLORS)]

    # Find last observed column
    last_obs_j = -1
    for j in range(len(col_headers) - 1, -1, -1):
        if not np.isnan(values[row_idx, j]):
            last_obs_j = j; break

    obs_x, obs_y, proj_x, proj_y = [], [], [], []
    for j, col in enumerate(col_headers):
        v  = values[row_idx, j]
        pv = projected[row_idx, j]
        isp = is_projected[row_idx, j]
        if j <= last_obs_j and not np.isnan(v):
            obs_x.append(col); obs_y.append(v)
        if isp and not np.isnan(pv):
            # bridge from last observed point
            if j == last_obs_j + 1 and last_obs_j >= 0:
                proj_x.append(col_headers[last_obs_j])
                proj_y.append(values[row_idx, last_obs_j])
            proj_x.append(col); proj_y.append(pv)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=obs_x, y=obs_y, mode="lines+markers", name="Observed",
        line=dict(color=color, width=2.5), marker=dict(size=7, color=color),
    ))
    if proj_x:
        fig.add_trace(go.Scatter(
            x=proj_x, y=proj_y, mode="lines+markers", name="Projected",
            line=dict(color=color, width=2.5, dash="dash"),
            marker=dict(size=7, color=color, symbol="circle-open"),
        ))

    layout = _base_layout(
        height=220, margin=dict(l=55, r=10, t=30, b=50),
        title=dict(text="Cumulative Development", font=dict(size=12, color=T["text"])),
        showlegend=True,
    )
    layout["xaxis"]["title"] = dict(text="Development Period", font=dict(color=T["axis_color"], size=11))
    layout["yaxis"]["title"] = dict(text="Cumulative Losses", font=dict(color=T["axis_color"], size=11))
    layout["legend"].update(orientation="h", y=-0.3, x=0,
                            title=dict(text="", font=dict(color=T["text"], size=10)))
    fig.update_layout(**layout)
    return fig


def make_ldf_chart(row_idx, row_headers, col_headers, projected, is_projected, dev_factors):
    label = row_headers[row_idx]
    color = CHART_COLORS[row_idx % len(CHART_COLORS)]
    n_cols = len(col_headers)
    row_ldfs, transitions, bar_colors = [], [], []
    for j in range(n_cols - 1):
        c, nx = projected[row_idx, j], projected[row_idx, j + 1]
        row_ldfs.append(nx / c if not np.isnan(c) and not np.isnan(nx) and c != 0 else np.nan)
        transitions.append(f"{col_headers[j]}→{col_headers[j+1]}")
        bar_colors.append(color if not is_projected[row_idx, j + 1] else "#fbbf24")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=transitions, y=dev_factors, name="Chain-ladder avg",
        marker_color="#cbd5e1", marker_line_width=0,
    ))
    fig.add_trace(go.Bar(
        x=transitions, y=row_ldfs, name=label,
        marker_color=bar_colors,
        marker_line_color="#f59e0b",
        marker_line_width=[0 if not is_projected[row_idx, j + 1] else 1.5 for j in range(n_cols - 1)],
    ))
    layout = _base_layout(
        barmode="group", height=240,
        margin=dict(l=55, r=10, t=30, b=70),
        title=dict(text="LDFs vs Chain-Ladder Avg", font=dict(size=12, color=T["text"])),
    )
    layout["xaxis"].update(
        title=dict(text="Transition", font=dict(color=T["axis_color"], size=11)),
        tickangle=-30, showgrid=False,
    )
    layout["yaxis"].update(
        title=dict(text="LDF", font=dict(color=T["axis_color"], size=11)),
        tickformat=".3f",
    )
    layout["legend"].update(
        orientation="h", y=-0.42, x=0,
        title=dict(text="Series", font=dict(color=T["text"], size=10)),
    )
    fig.update_layout(**layout)
    return fig


def make_pct_chart(row_idx, row_headers, col_headers, projected, ultimates):
    label = row_headers[row_idx]
    color = CHART_COLORS[row_idx % len(CHART_COLORS)]
    n_rows, n_cols = len(row_headers), len(col_headers)
    ult = ultimates[row_idx]
    year_pct, avg_pct = [], []
    for j in range(n_cols):
        yv = projected[row_idx, j]
        year_pct.append(yv / ult * 100 if not np.isnan(yv) and not np.isnan(ult) and ult != 0 else np.nan)
        vals = [projected[ri, j] / ultimates[ri] * 100
                for ri in range(n_rows)
                if not np.isnan(projected[ri, j]) and not np.isnan(ultimates[ri]) and ultimates[ri] != 0]
        avg_pct.append(np.mean(vals) if vals else np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=col_headers, y=avg_pct, mode="lines", name="All-year avg",
        line=dict(color="#94a3b8", width=1.5, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=col_headers, y=year_pct, mode="lines+markers", name=label,
        line=dict(color=color, width=2.5), marker=dict(size=7, color=color),
    ))
    fig.add_hline(y=100, line_dash="dot", line_color=T["border"], line_width=1)
    layout = _base_layout(
        height=240, margin=dict(l=55, r=10, t=30, b=60),
        title=dict(text="% Developed vs All-Year Avg", font=dict(size=12, color=T["text"])),
    )
    layout["xaxis"]["title"] = dict(text="Development Period", font=dict(color=T["axis_color"], size=11))
    layout["yaxis"].update(
        title=dict(text="% of Ultimate", font=dict(color=T["axis_color"], size=11)),
        ticksuffix="%", range=[0, 105],
    )
    layout["legend"].update(
        orientation="h", y=-0.36, x=0,
        title=dict(text="Series", font=dict(color=T["text"], size=10)),
    )
    fig.update_layout(**layout)
    return fig


def make_triangle_table(row_headers, col_headers, values, projected, is_projected,
                        ultimates, ibnr, show_proj):
    n_rows, n_cols = len(row_headers), len(col_headers)
    vmin, vmax = float(np.nanmin(projected)), float(np.nanmax(projected))
    lo_rgb = BLUE_SCHEME["lo"]
    hi_rgb = BLUE_SCHEME["hi"]

    cells_by_col, fill_by_col, font_colors_by_col = [], [], []

    # Row label column
    cells_by_col.append(row_headers)
    fill_by_col.append(["#f8fafc"] * n_rows)
    font_colors_by_col.append(["#1e293b"] * n_rows)

    for j in range(n_cols):
        col_vals, col_fill, col_fc = [], [], []
        for i in range(n_rows):
            v, pv, isp = values[i, j], projected[i, j], is_projected[i, j]
            if not show_proj and isp:
                col_vals.append(""); col_fill.append("#e5e7eb"); col_fc.append("#9ca3af")
            elif np.isnan(v) and np.isnan(pv):
                col_vals.append(""); col_fill.append("#e5e7eb"); col_fc.append("#9ca3af")
            elif isp:
                col_vals.append(fmt(pv)); col_fill.append("#fef3c7"); col_fc.append("#92400e")
            else:
                t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5))
                bg, (r2, g2, b2) = lerp_color(t, lo_rgb, hi_rgb)
                lum = 0.2126 * r2 + 0.7152 * g2 + 0.0722 * b2
                col_vals.append(fmt(v))
                col_fill.append(bg)
                col_fc.append("#ffffff" if lum < 140 else "#1e293b")
        cells_by_col.append(col_vals)
        fill_by_col.append(col_fill)
        font_colors_by_col.append(col_fc)

    # Ultimate column
    cells_by_col.append([fmt(u) for u in ultimates])
    fill_by_col.append(["#eff6ff"] * n_rows)
    font_colors_by_col.append(["#1e40af"] * n_rows)

    # IBNR column
    ibnr_vals, ibnr_bg, ibnr_fg = [], [], []
    for ib in ibnr:
        ibnr_vals.append(fmt(ib))
        if np.isnan(ib) or ib <= 0:
            ibnr_bg.append("#f0fdf4"); ibnr_fg.append("#15803d")
        else:
            ibnr_bg.append("#fffbeb"); ibnr_fg.append("#b45309")
    cells_by_col.append(ibnr_vals)
    fill_by_col.append(ibnr_bg)
    font_colors_by_col.append(ibnr_fg)

    header_vals = ["Period"] + col_headers + ["Ultimate", "IBNR"]
    col_widths   = [80] + [80] * n_cols + [90, 90]

    fig = go.Figure(go.Table(
        columnwidth=col_widths,
        header=dict(
            values=[f"<b>{h}</b>" for h in header_vals],
            fill_color="#f1f5f9", font=dict(color="#475569", size=11),
            align=["left"] + ["right"] * (n_cols + 2), height=32,
        ),
        cells=dict(
            values=cells_by_col, fill_color=fill_by_col,
            font=dict(color=font_colors_by_col, size=11),
            align=["left"] + ["right"] * (n_cols + 2), height=30,
        ),
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=max(200, n_rows * 34 + 50),
        paper_bgcolor="#ffffff",
    )
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── Top bar ──────────────────────────────────────────────────────────────
    t1, t2, t3 = st.columns([3, 1.4, 2])
    with t1:
        st.markdown("### △ Actuarial Triangle Viewer")
    with t2:
        proj_label = "📊 Hide Projected" if st.session_state.show_proj else "📊 Show Projected"
        if st.button(proj_label, key="proj_toggle", use_container_width=True):
            st.session_state.show_proj = not st.session_state.show_proj
            st.rerun()
    with t3:
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    # Cache file content in session state so reruns don't lose it
    if uploaded is not None:
        st.session_state.file_content = uploaded.read().decode("utf-8")
        st.session_state.file_name = uploaded.name

    st.divider()

    if st.session_state.file_content is None:
        st.info("Upload a loss development triangle CSV to begin. "
                "First column = origin period labels, remaining columns = development periods.")
        st.code(
            "Period, 12mo, 24mo, 36mo\n"
            "2020, 12500, 21000, 27300\n"
            "2021, 13800, 22400,\n"
            "2022, 14600,,",
            language=None,
        )
        return

    # ── Parse & compute ───────────────────────────────────────────────────────
    try:
        content = st.session_state.file_content
        row_headers, col_headers, raw_values = parse_triangle(content)
        dev_factors, projected, is_projected, ultimates, latest_obs, ibnr = compute_chain_ladder(raw_values)
    except Exception as e:
        st.error(f"Could not parse file: {e}")
        return

    n_rows        = len(row_headers)
    n_cols        = len(col_headers)
    num_projected = int(is_projected.sum())
    total_ult     = float(np.nansum(ultimates))
    total_obs     = float(np.nansum(latest_obs))
    total_ibnr    = total_ult - total_obs
    obs_col_idx   = latest_obs_col_idx(raw_values)
    show_proj     = st.session_state.show_proj

    # ── Metric row ────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Ultimate",  fmt(total_ult),       help="Projected to full development")
    m2.metric("Total Observed",  fmt(total_obs),       help="Latest diagonal values")
    m3.metric("Total IBNR",      fmt(total_ibnr),      help="Incurred but not reported")
    m4.metric("Projected Cells", str(num_projected),   help=f"of {n_rows * n_cols} total cells")

    st.divider()

    # ── 1. LDF strip + Development triangle ───────────────────────────────────
    # Compact horizontal LDF chips
    chips_html = " ".join(
        f'<span style="display:inline-flex;flex-direction:column;align-items:center;'
        f'background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;'
        f'padding:4px 10px;margin:2px;font-size:0.75rem;white-space:nowrap;">'
        f'<span style="color:#64748b;font-size:0.68rem;">'
        f'{col_headers[j]}&nbsp;→&nbsp;{col_headers[j+1]}</span>'
        f'<span style="color:#1e293b;font-weight:600;">'
        f'{dev_factors[j]:.4f}</span></span>'
        for j in range(n_cols - 1)
        if not np.isnan(dev_factors[j])
    )
    st.markdown(
        f'<div style="margin-bottom:0.5rem;">'
        f'<span style="font-size:0.78rem;color:#64748b;font-weight:600;'
        f'margin-right:6px;">Chain-ladder LDFs</span>'
        f'{chips_html}</div>',
        unsafe_allow_html=True,
    )

    proj_note = f" · {num_projected} projected cells" if show_proj and num_projected else ""
    with st.expander(f"Development Triangle{proj_note}", expanded=True):
        vmin = float(np.nanmin(projected))
        vmax = float(np.nanmax(projected))
        st.caption(f"Range: **{fmt(vmin)}** → **{fmt(vmax)}**")
        st.plotly_chart(
            make_triangle_table(row_headers, col_headers, raw_values,
                                projected, is_projected, ultimates, ibnr, show_proj),
            use_container_width=True, config={"displayModeBar": False},
        )

    st.divider()

    # ── 2. Accordion — per origin year ────────────────────────────────────────
    st.markdown("**Origin Year Breakdown**")
    st.caption("Expand a year to see its development curve, LDFs and % developed")

    for i, label in enumerate(row_headers):
        rep     = latest_obs[i]
        ult     = ultimates[i]
        ib      = ibnr[i]
        pct     = rep / ult * 100 if not np.isnan(rep) and not np.isnan(ult) and ult != 0 else np.nan
        pct_str = f"{pct:.1f}%" if not np.isnan(pct) else "—"
        ib_str  = fmt(ib) if not np.isnan(ib) and ib > 0 else "Fully developed"

        header = (
            f"**{label}** &nbsp;&nbsp; "
            f"Reported: {fmt(rep)} &nbsp;·&nbsp; "
            f"Ultimate: {fmt(ult)} &nbsp;·&nbsp; "
            f"IBNR: {ib_str} &nbsp;·&nbsp; "
            f"% Dev: {pct_str}"
        )

        with st.expander(header, expanded=False):
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Reported",    fmt(rep))
            a2.metric("Ultimate",    fmt(ult))
            a3.metric("IBNR",        fmt(ib))
            a4.metric("% Developed", pct_str)

            st.plotly_chart(
                make_single_dev_curve(i, row_headers, col_headers, raw_values,
                                      projected, is_projected),
                use_container_width=True, config={"displayModeBar": False},
            )




if __name__ == "__main__":
    main()
