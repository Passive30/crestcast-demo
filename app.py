import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image

# === Page Configuration ===
st.set_page_config(
    page_title="Passive 3.0™ Overlay – Direct Indexing Demo",
    layout="wide"
)

# === Custom Intervallum Brand Styling ===
st.markdown("""
    <style>
    /* Fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'DM Sans', sans-serif;
        color: #F5F8FA;
        background-color: #0E1A2F;
    }

    /* Streamlit Table Headers */
    .stDataFrame th {
        background-color: #121D33 !important;
        color: #F5F8FA !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #3D8BFF !important;
    }

    /* Table cells */
    .stDataFrame td {
        background-color: #0E1A2F !important;
        color: #F5F8FA !important;
        border-bottom: 0.25px solid #1F2A44;
    }

    /* Alternating row background */
    .stDataFrame tbody tr:nth-child(even) td {
        background-color: #16233C !important;
    }

    /* Scrollbar (if needed) */
    ::-webkit-scrollbar-thumb {
        background: #3D8BFF;
    }
    ::-webkit-scrollbar-track {
        background: #121D33;
    }

    </style>
""", unsafe_allow_html=True)


# === Display Banner ===
banner = Image.open("banner.png")
st.image(banner, use_container_width=True)
st.warning(
    "🔧 This demo is currently in **DRAFT** status. Index data becomes live and finalized on August 1, 2025. "
    "Please treat all results as preliminary and subject to refinement during this period."
)

# === Load and Clean CSV ===
file_path = "timeseries_6-25-2025.csv"
returns_df = pd.read_csv(file_path)
returns_df['Date'] = pd.to_datetime(returns_df['Date'], errors='coerce')
returns_df = returns_df.set_index('Date')
returns_df = returns_df.apply(pd.to_numeric, errors='coerce')
returns_df = returns_df.dropna(how="all")

# === Metric Functions ===
def annualized_return(r):
    if r.empty:
        return np.nan
    cumulative_return = (1 + r).prod()
    n_years = len(r) / 12
    return cumulative_return ** (1 / n_years) - 1


def annualized_std(r):
    if r.empty:
        return np.nan
    return r.std() * np.sqrt(12)

def beta_alpha(port, bench):
    port = port.dropna()
    bench = bench.dropna()
    df = pd.concat([port.rename("CrestCast"), bench.rename("Benchmark")], axis=1).dropna()
    if df.shape[0] < 2:
        return np.nan, np.nan
    cov = np.cov(df["CrestCast"], df["Benchmark"])
    beta = cov[0, 1] / cov[1, 1]
    alpha = annualized_return(df["CrestCast"]) - beta * annualized_return(df["Benchmark"])
    return beta, alpha


def sharpe_ratio(r, rf=0.0): return ((r - rf/12).mean() / r.std()) * np.sqrt(12)
def max_drawdown(r):
    cumulative = (1 + r).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()
def ulcer_index(returns):
    if returns.empty:
        return np.nan
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    squared_dd = drawdown ** 2
    return np.sqrt(squared_dd.mean())

def ulcer_ratio(port, bench):
    ui = ulcer_index(port)
    ar = annualized_return(port)
    return ar / ui if ui != 0 else np.nan
    
def cumulative_return(series): return (1 + series).cumprod()
def tracking_error(port, bench):
    try:
        df = pd.concat([port, bench], axis=1).dropna()
        if df.shape[0] < 2:
            return np.nan
        excess_returns = df.iloc[:, 0] - df.iloc[:, 1]
        if not np.issubdtype(excess_returns.dtype, np.number):
            return np.nan
        return excess_returns.std() * np.sqrt(12)
    except Exception as e:
        print(f"Tracking Error Calculation Failed: {e}")
        return np.nan
def information_ratio(port, bench):
    try:
        df = pd.concat([port, bench], axis=1).dropna()
        if df.shape[0] < 2:
            return np.nan
        beta, alpha = beta_alpha(df.iloc[:, 0], df.iloc[:, 1])
        te = tracking_error(df.iloc[:, 0], df.iloc[:, 1])
        return alpha / te if te and te != 0 else np.nan
    except Exception as e:
        print(f"Information Ratio Calculation Failed: {e}")
        return np.nan


# === Intro and Branding ===
st.markdown("""
## Introducing CrestCast™ Macro-Aware US Factor Rotation Index
### Detailed Analytics Demonstration
This demo illustrates how the CrestCast™ index can dynamically enhance core equity exposure using regime-aware factor rotation.
""")

# === Section 1: Simulation Parameters ===
st.header("1. Simulation Setup")
st.info(
       "📌 *Disclaimer:* Intervallum Technologies is not a registered investment advisor and does not provide personalized investment advice. "
    "This demonstration is for informational and educational purposes only and is intended to illustrate the capabilities of the CrestCast™ "
    "Macro-Aware U.S. Factor Rotation Index. The model was built using walk-forward cross-validation on data from 1968 through 2013. "
    "From 2014 forward, index results reflect strict out-of-sample application with no retraining or parameter adjustments—"
    "providing a real-world view of model integrity and implementation discipline."

)



client_name = "CC Demo"
st.markdown(f"**Demo Label:** {client_name}")
account_type = "Individual"

# === Section 2: Select Base Index ===
st.header("2. Select Benchmark for Comparison")

index_options = {
    "Russell 3000 Index": "^RUATR",
    "R3000 ETF (IWV)": "IWV"  # Add this line
}

selected_label = st.selectbox("Preferred Index", list(index_options.keys()))
preferred_index = index_options[selected_label]


# === Section 3: Activate Overlay Logic ===
st.header("3. Activate Macro-Aware Index")
macro_aware = True


if macro_aware:
    st.markdown("The CrestCast™ index can serve as an overlay to empower dynamic shifts to style allocations in an underlying index.")

    # Overlay Fee Dropdown
    fee_bps = st.selectbox("Overlay Fee (basis points))", [0, 20, 35, 50], index=0)
    annual_fee = fee_bps / 10000
    monthly_fee = annual_fee / 12

    # Friendly Tracking Error Dropdown
    # Hard-coded tracking error preference
    tracking_error_label = "How closely should your portfolio follow the index?"
    tracking_error_label_choice = "Flexible"

    # Map investor-friendly terms to λ
    lambda_values = {
        "Very closely": 0.2,
        "Somewhat closely": 0.5,
        "Flexible": 1.0
    }
    lam = lambda_values[tracking_error_label_choice]


    # Optional Email Opt-In
#    email_opt_in = st.checkbox("📬 Please send me an email with monthly commentary on regime outlook and implications for my portfolio.")

else:
    # If macro overlay is off, set safe defaults
    fee_bps = 0
    monthly_fee = 0.0
    tracking_error = "Not applicable"
    lam = 0.0
    email_opt_in = False

# --- Section 4: Select Time Period for Analysis ---
st.header("4. Select Time Period for Analysis")

min_date = returns_df.index.min().to_pydatetime().date()
max_date = returns_df.index.max().to_pydatetime().date()
min_window = pd.DateOffset(years=5)

# Mode selector
analysis_mode = st.radio(
    "Choose Time Frame Mode:",
    ["Custom Range", "Rolling 5-Year Window"],
    index=0,  # make Custom Range the default
    horizontal=True
)


# Initialize variables
start_date, end_date = None, None

if analysis_mode == "Custom Range":
    # Calculate default to enforce minimum range
    default_start = min_date  # start at full history by default

    # Custom range slider
    date_range = st.slider(
        "Select Custom Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(default_start, max_date)
    )

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    # Enforce 3-year minimum
    if (end_date - start_date) < pd.Timedelta(days=365 * 3):
        st.error("Please select a date range of at least 3 years.")
        st.stop()

    st.caption(f"Showing performance from **{start_date.date()}** to **{end_date.date()}**")

elif analysis_mode == "Rolling 5-Year Window":
    # Calculate latest valid start
    latest_valid_start = (pd.to_datetime(max_date) - min_window).date()

    # Single-point slider
    rolling_start = st.slider(
        "Select Start Date (5-Year Window)",
        min_value=min_date,
        max_value=latest_valid_start,
        value=latest_valid_start
    )

    start_date = pd.to_datetime(rolling_start)
    end_date = start_date + min_window - pd.DateOffset(days=1)

    st.caption(f"Showing performance from **{start_date.date()}** to **{end_date.date()}**")

# Slice return data for selected time frame
cumulative_returns = returns_df.loc[start_date:end_date]


# --- Extract Data for Chart + Stats ---
benchmark = cumulative_returns[preferred_index]
gross_crestcast = cumulative_returns["CrestCast"]
net_crestcast = gross_crestcast - monthly_fee

# Create a working returns DataFrame with standardized column names
returns_df = pd.concat([
    benchmark.rename("Benchmark"),
    net_crestcast.rename("CrestCast")
], axis=1).dropna()

valid_data = pd.concat([benchmark, net_crestcast], axis=1).dropna()
benchmark = valid_data.iloc[:, 0]
net_crestcast = valid_data.iloc[:, 1]

# Blend based on tracking error (λ)
blended_crestcast = (1 - lam) * benchmark + lam * net_crestcast

# --- Section: Percent Return Over Selected Time Range ---
st.subheader("📈 Net Total Return Over Selected Period (%)")

# Calculate cumulative return as percent
cum_benchmark = (1 + benchmark).cumprod() - 1

if macro_aware:
    cum_crestcast = (1 + net_crestcast).cumprod() - 1
    cum_blended = (1 + blended_crestcast).cumprod() - 1

    if lam == 1.0:
        # Only show benchmark + full CrestCast line
        comparison_df = pd.DataFrame({
            f"{preferred_index} (Benchmark)": cum_benchmark,
            "CrestCast (100% Net of Fee)": cum_crestcast
        })
    else:
        # Show all three
        comparison_df = pd.DataFrame({
            f"{preferred_index} (Benchmark)": cum_benchmark,
            f"CrestCast™({tracking_error_label_choice})": cum_blended,
            "CrestCast (100% Net of Fee)": cum_crestcast
        })

else:
    comparison_df = pd.DataFrame({
        f"{preferred_index} (Benchmark)": cum_benchmark
    })

comparison_df = comparison_df.dropna()

# Plot full-width with matplotlib
if not comparison_df.empty:
    fig, ax = plt.subplots(figsize=(6, 3))
    for col in comparison_df.columns:
        ax.plot(comparison_df.index, comparison_df[col], label=col)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    title_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    ax.set_title(f"Net Total Return ({title_range})", fontsize=12)
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig)
else:
    st.warning("Not enough data to plot. Please try a different date range.")

# --- Performance Summary Table ---
st.subheader(f"📊 Performance Summary (Net of Fees) — {start_date.date()} to {end_date.date()}")


# New: Up/down capture & return delta
def up_capture(port, bench):
    mask = bench > 0
    return port[mask].mean() / bench[mask].mean() if bench[mask].mean() != 0 else np.nan

def down_capture(port, bench):
    mask = bench < 0
    return port[mask].mean() / bench[mask].mean() if bench[mask].mean() != 0 else np.nan

def return_diff(port, bench):
    return annualized_return(port) - annualized_return(bench)

# Rename series before using beta_alpha
named_crestcast = blended_crestcast.rename("CrestCast")
named_benchmark = benchmark.rename("Benchmark")

# Metrics to display
metrics = [
    "Annualized Return", "Annualized Std Dev", 
    "Beta vs Benchmark", "Alpha vs Benchmark", 
    "Sharpe Ratio", "Tracking Error", "Information Ratio",
    "Max Drawdown", "Ulcer Ratio", "Up Capture", "Down Capture"
]

# CrestCast metrics (labeled for beta_alpha to work)
crestcast_metrics = [
    annualized_return(named_crestcast),
    annualized_std(named_crestcast),
    *beta_alpha(named_crestcast, named_benchmark),
    sharpe_ratio(named_crestcast),
    tracking_error(named_crestcast, named_benchmark),
    information_ratio(named_crestcast, named_benchmark),
    max_drawdown(named_crestcast),
    ulcer_ratio(named_crestcast, named_benchmark),
    up_capture(named_crestcast, named_benchmark),
    down_capture(named_crestcast, named_benchmark)
]

# Benchmark metrics
benchmark_metrics = [
    annualized_return(named_benchmark),
    annualized_std(named_benchmark),
    None, None,
    sharpe_ratio(named_benchmark),
    None,  # Tracking Error placeholder
    None,  # Information Ratio
    max_drawdown(named_benchmark),
    ulcer_ratio(named_benchmark, named_benchmark),
    1.0, 1.0
]


# Format function with metric context
def fmt(x, metric=None):
    if x is None:
        return "-"
    if isinstance(x, (float, np.float64)):
        if metric in ["Sharpe Ratio", "Information Ratio", "Ulcer Ratio"]:
            return f"{x:.2f}"
        return f"{x:.2%}" if abs(x) < 10 else f"{x:.2f}"
    return str(x)

# Rebuild and format table from raw values
formatted_data = []

for i in range(len(metrics)):
    metric = metrics[i]
    cc_val = crestcast_metrics[i]
    bench_val = benchmark_metrics[i]
    
    formatted_data.append({
        "Metric": metric,
        f"CrestCast™ ({tracking_error_label_choice})": fmt(cc_val, metric),
        "Benchmark": fmt(bench_val, metric)
    })

summary_df = pd.DataFrame(formatted_data)
st.table(summary_df)

# === Full-Period Drawdown Comparison ===
st.subheader("📉 Full-Period Drawdown: CrestCast™ vs. Benchmark")

# Align data and calculate drawdowns
valid_data = pd.concat([blended_crestcast, benchmark], axis=1).dropna()
blended_crestcast = valid_data.iloc[:, 0]
benchmark = valid_data.iloc[:, 1]

cumulative_crest = (1 + blended_crestcast).cumprod()
cumulative_bench = (1 + benchmark).cumprod()

dd_crest = (cumulative_crest / cumulative_crest.cummax()) - 1
dd_bench = (cumulative_bench / cumulative_bench.cummax()) - 1

# Plot
fig, ax = plt.subplots(figsize=(6, 3))
ax.fill_between(dd_crest.index, dd_crest.values, 0, color='green', alpha=0.15, label="CrestCast™ Drawdown")
ax.plot(dd_bench.index, dd_bench.values, label="Benchmark Drawdown", color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7)
ax.set_ylabel("Drawdown")
ax.set_title("Full-Period Drawdown: CrestCast™ vs. Benchmark")
ax.set_ylim(-0.6, 0.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc="lower left")
st.pyplot(fig)

st.caption(
    "**Interpretation:** This chart shows the historical drawdowns of CrestCast™ index and the benchmark across the full sample period. "
    "It highlights the depth and duration of capital declines, offering a clear comparison of downside experience across time."
)

# Download button
drawdown_df = pd.DataFrame({
    "Date": dd_crest.index,
    "CrestCast™ Drawdown": dd_crest.values,
    "Benchmark Drawdown": dd_bench.values
})
csv_drawdown = drawdown_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Full-Period Drawdown Data",
    data=csv_drawdown,
    file_name="full_period_drawdowns.csv",
    mime="text/csv"
)

st.markdown("## 🔎 Advanced Analytics")


# === Metric-First Performance Table ===
if st.checkbox("Show 1yr, 5yr, 10yr, Since Inception Statistics"):
    st.subheader("📊 CrestCast™ vs. Benchmark: Metrics by Period")

    # Periods to evaluate
    today = blended_crestcast.index[-1]
    periods = {
        "1 Year": today - pd.DateOffset(years=1),
        "5 Year": today - pd.DateOffset(years=5),
        "10 Year": today - pd.DateOffset(years=10),
        "Since Inception": blended_crestcast.index[0]
    }

    # Metrics to compute
    metrics = {
        "Ann. Return": lambda p, b: (annualized_return(p), annualized_return(b)),
        "Ann. Std Dev": lambda p, b: (annualized_std(p), annualized_std(b)),
        "Beta": lambda p, b: beta_alpha(p, b)[0],
        "Alpha": lambda p, b: beta_alpha(p, b)[1],
        "Sharpe Ratio": lambda p, b: (sharpe_ratio(p), sharpe_ratio(b)),
        "Max Drawdown": lambda p, b: (max_drawdown(p), max_drawdown(b)),
        "Ulcer Ratio": lambda p, b: (ulcer_ratio(p, b), ulcer_ratio(b, b)),
        "Tracking Error": lambda p, b: tracking_error(p, b),
        "Information Ratio": lambda p, b: information_ratio(p, b),
        "Up Capture": lambda p, b: (up_capture(p, b), up_capture(b, b)),
        "Down Capture": lambda p, b: (down_capture(p, b), down_capture(b, b)),
    }

    # Storage: {(period, label) -> {metric: value}}
    from collections import defaultdict
    
    # Create nested structure: outer keys are periods, inner are CrestCast / Benchmark
    nested_data = defaultdict(dict)
    
    for label, start_date in periods.items():
        port = blended_crestcast.loc[start_date:].rename("CrestCast")
        bench = benchmark.loc[start_date:].rename("Benchmark")
    
        df = pd.concat([port, bench], axis=1).dropna()
        if df.empty:
            for metric_name in metrics.keys():
                nested_data[(label, "CrestCast™")][metric_name] = np.nan
                nested_data[(label, "Benchmark")][metric_name] = np.nan
            continue
    
        for metric_name, func in metrics.items():
            result = func(df["CrestCast"], df["Benchmark"])
            if isinstance(result, tuple):
                nested_data[(label, "CrestCast™")][metric_name] = result[0]
                nested_data[(label, "Benchmark")][metric_name] = result[1]
            else:
                nested_data[(label, "CrestCast™")][metric_name] = result
                nested_data[(label, "Benchmark")][metric_name] = np.nan
    
    # Build new DataFrame with MultiIndex columns
    multi_index_df = pd.DataFrame(nested_data)
    multi_index_df.columns = pd.MultiIndex.from_tuples(multi_index_df.columns)
    multi_index_df = multi_index_df[["1 Year", "5 Year", "10 Year", "Since Inception"]]  # Reorder
    
    # Format values as percents where applicable
    # Metrics that should be shown as decimal numbers (not percentages)
    decimal_metrics = ["Beta", "Sharpe Ratio", "Ulcer Ratio", "Information Ratio"]
    
    def smart_format(val, metric_name):
        if isinstance(val, (int, float)):
            return f"{val:.2f}" if any(dm in metric_name for dm in decimal_metrics) else f"{val:.2%}"
        return val
    
    formatted_df = multi_index_df.copy()
    for metric in formatted_df.index:
        formatted_df.loc[metric] = formatted_df.loc[metric].apply(lambda x: smart_format(x, metric))

    
    # Display
    st.dataframe(formatted_df, use_container_width=True)
    
    # Download CSV
    csv_df = multi_index_df.copy()
    csv_df.columns = [f"{p} - {s}" for p, s in csv_df.columns]
    csv_bytes = csv_df.reset_index().rename(columns={"index": "Metric"}).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Metric Summary by Period",
        data=csv_bytes,
        file_name="metrics_by_period.csv",
        mime="text/csv"
    )


if st.checkbox("Show Rolling 5-Year Alpha Summary and Distribution"):

    rolling_window = 60  # 5 years
    alpha_values = []
    alpha_dates = []

    for i in range(rolling_window, len(returns_df)):
        window = returns_df.iloc[i - rolling_window:i]

        if "CrestCast" not in window.columns or "Benchmark" not in window.columns:
            st.error("Missing required columns: 'CrestCast' and 'Benchmark'")
            st.stop()

        port = window["CrestCast"].rename("CrestCast")
        bench = window["Benchmark"].rename("Benchmark")

        if port.isnull().any() or bench.isnull().any():
            continue

        _, alpha = beta_alpha(port, bench)
        alpha_values.append(alpha)
        alpha_dates.append(window.index[-1])

    alpha_series = pd.Series(alpha_values, index=alpha_dates)

    if alpha_series.empty:
        st.warning("Not enough data to calculate rolling 5-year alpha.")
    else:
        percent_positive = (alpha_series > 0).mean()
        average_alpha = alpha_series.mean()

        st.markdown(f"- **Percent of 5-Year Windows with Positive Alpha**: **{percent_positive:.1%}**")
        st.markdown(f"- **Average Annualized Alpha (5-Year Windows)**: **{average_alpha:.2%}**")

        # Histogram
        fig1, ax1 = plt.subplots()
        alpha_series.hist(bins=30, edgecolor='black', ax=ax1)
        ax1.set_title("Distribution of 5-Year Rolling Alpha")
        ax1.set_xlabel("Annualized Alpha")
        ax1.set_ylabel("Frequency")
        st.pyplot(fig1)

        # Bar chart of rolling alpha with Intervallum branding
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        
        # Use default matplotlib blue
        alpha_series.plot(kind="bar", ax=ax2, color="#1f77b4", edgecolor="black")
        
        # Zero line
        ax2.axhline(0, linestyle='--', color='gray', linewidth=1)
        
        # Title and axes
        ax2.set_title("Rolling 5-Year Alpha Over Time")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Annualized Alpha")
        
        # Annual x-axis labels (only for January)
        tick_labels = []
        tick_positions = []
        
        for i, dt in enumerate(alpha_series.index):
            if dt.month == 1:
                tick_labels.append(dt.strftime('%Y'))
                tick_positions.append(i)
        
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels, rotation=45)
        
        # Final layout tweak
        plt.tight_layout()
        
        # Show in Streamlit
        st.pyplot(fig2)

if st.checkbox("Show Rolling 5-Year Sharpe Comparison"):
    # Sharpe stats and chart
    rolling_window = 60  # 5 years
    crest_sharpes = []
    bench_sharpes = []
    dates = []

    for i in range(rolling_window, len(returns_df)):
        window = returns_df.iloc[i - rolling_window:i]

        # Safe column access
        if "CrestCast" not in window.columns or "Benchmark" not in window.columns:
            continue

        port = window["CrestCast"]
        bench = window["Benchmark"]

        if port.isnull().any() or bench.isnull().any():
            continue

        crest_sharpes.append(sharpe_ratio(port))
        bench_sharpes.append(sharpe_ratio(bench))
        dates.append(window.index[-1])

    # Assemble results
    sharpe_df = pd.DataFrame({
        "Date": dates,
        "CrestCast Sharpe": crest_sharpes,
        "Benchmark Sharpe": bench_sharpes
    }).set_index("Date")

    # Summary stats
    percent_better_sharpe = (sharpe_df["CrestCast Sharpe"] > sharpe_df["Benchmark Sharpe"]).mean()
    avg_diff = (sharpe_df["CrestCast Sharpe"] - sharpe_df["Benchmark Sharpe"]).mean()

    st.markdown("### 📈 Rolling 5-Year Sharpe Ratio Comparison")
    st.markdown(f"- **% of 5-Year Windows Where CrestCast™ > Benchmark**: **{percent_better_sharpe:.1%}**")
    st.markdown(f"- **Average Sharpe Advantage (CrestCast™ minus Benchmark)**: **{avg_diff:.2f}**")

    # Optional chart
    fig, ax = plt.subplots(figsize=(6, 3))  # Smaller footprint
    sharpe_df.plot(ax=ax)
    ax.set_title("Rolling 5-Year Sharpe Ratio")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig)

    # Compute Sharpe improvement series
    sharpe_diff = sharpe_df["CrestCast Sharpe"] - sharpe_df["Benchmark Sharpe"]
    
    # Plot histogram of Sharpe improvements
    fig, ax = plt.subplots(figsize=(6, 3))
    sharpe_diff.hist(bins=30, edgecolor="blue", ax=ax)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Distribution of Sharpe Ratio Improvement (CrestCast™ - Benchmark)")
    ax.set_xlabel("Sharpe Advantage")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# === Final Summary Stat Row (Always Visible) ===
st.markdown("---")
#st.markdown("### 📊 Performance Consistency: Alpha + Sharpe Advantage")

cols = st.columns(3)

with cols[0]:
    st.metric(
        label="📈 10-Year Windows",
        value="100% Alpha ⬆",
        delta="Sharpe outperformance: 99% of periods"
    )

with cols[1]:
    st.metric(
        label="📊 5-Year Windows",
        value="91% Alpha ⬆",
        delta="Sharpe outperformance: 75% of periods"
    )

with cols[2]:
    st.metric(
        label="📉 3-Year Windows",
        value="74% Alpha ⬆",
        delta="Sharpe outperformance: 60% of periods"
    )


st.markdown("---")

st.markdown("### ☎️ Let's Talk")
st.markdown(
    "#### _Whether you're building ETFs, models, or SMAs & personalized portfolios, CrestCast™ can add power to your solution. "
    "[Schedule a quick call](https://meetings-na2.hubspot.com/alan-thomson) to explore fit. Macro-Aware Factor Rotation is here._"
)
st.markdown("— *Powered by CrestCast™*")
