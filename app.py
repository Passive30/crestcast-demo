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
# === Display Banner ===
banner = Image.open("banner.png")
st.image(banner, use_container_width=True)
st.warning(
    "🔧 This demo is currently in **DRAFT** status. CrestCast Index goes live on August 1, 2025. "
)

# === Load and Clean CSV ===
file_path = "timeseries_6-25-2025.csv"
returns_df = pd.read_csv(file_path)
returns_df['Date'] = pd.to_datetime(returns_df['Date'], errors='coerce')
returns_df = returns_df.set_index('Date')
returns_df = returns_df.apply(pd.to_numeric, errors='coerce')
returns_df = returns_df.dropna(how="all")

# === Risk-Free Rate Extraction ===
risk_free_series = returns_df['Risk_Free'].dropna()

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

def beta_alpha(port, bench, rf=None):
    port = port.dropna()
    bench = bench.dropna()
    df = pd.concat([port.rename("CrestCast"), bench.rename("Benchmark")], axis=1).dropna()
    if rf is not None:
        rf = rf.reindex(df.index).fillna(method='ffill')
        df["CrestCast"] -= rf
        df["Benchmark"] -= rf
    if df.shape[0] < 2:
        return np.nan, np.nan
    cov = np.cov(df["CrestCast"], df["Benchmark"])
    beta = cov[0, 1] / cov[1, 1]
    alpha = annualized_return(df["CrestCast"]) - beta * annualized_return(df["Benchmark"])
    return beta, alpha

def sharpe_ratio(r, rf=None):
    if r.empty:
        return np.nan
    if rf is None:
        rf = 0.0
    elif isinstance(rf, pd.Series):
        rf = rf.reindex(r.index).fillna(method='ffill')
        excess = r - rf
    else:
        excess = r - rf / 12
    return (excess.mean() / r.std()) * np.sqrt(12)

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

def cumulative_return(series):
    return (1 + series).cumprod()

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

def information_ratio(port, bench, rf=None):
    try:
        df = pd.concat([port, bench], axis=1).dropna()
        if df.shape[0] < 2:
            return np.nan
        beta, alpha = beta_alpha(df.iloc[:, 0], df.iloc[:, 1], rf=rf)
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
#st.markdown(f"**Demo Label:** {client_name}")
account_type = "Individual"

# === Section 2: Select Base Index ===
st.header("2. Select CrestCast Index")

# Select CrestCast index to analyze
available_indexes = [col for col in returns_df.columns if col.startswith("CrestCast")]
selected_index = st.selectbox("Choose a CrestCast Index to Analyze", available_indexes)

# Automatically select the benchmark
if "Bond" in selected_index:
    preferred_index = "AGG"
else:
    preferred_index = "Russell_3000"


# === Section 3: Activate Overlay Logic ===
st.header("3. Apply Management Fee")
macro_aware = True

if macro_aware:
    st.markdown("Apply expense ratio to simulate a live product comparison to the benchmark.")

    # Overlay Fee Dropdown
    fee_bps = st.selectbox("Fee Stress Test (basis points))", [0, 20, 35, 50, 75, 100], index=0)
    annual_fee = fee_bps / 10000
    monthly_fee = annual_fee / 12

    # Friendly Tracking Error Dropdown
    tracking_error_label = "How closely should your portfolio follow the index?"
    tracking_error_label_choice = "Flexible"

    # Map investor-friendly terms to λ
    lambda_values = {
        "Very closely": 0.2,
        "Somewhat closely": 0.5,
        "Flexible": 1.0
    }
    lam = lambda_values[tracking_error_label_choice]

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
    index=0,
    horizontal=True
)

# Initialize variables
start_date, end_date = None, None

if analysis_mode == "Custom Range":
    default_start = min_date
    date_range = st.slider(
        "Select Custom Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(default_start, max_date)
    )
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    if (end_date - start_date) < pd.Timedelta(days=365 * 3):
        st.error("Please select a date range of at least 3 years.")
        st.stop()

    st.caption(f"Showing performance from **{start_date.date()}** to **{end_date.date()}**")

elif analysis_mode == "Rolling 5-Year Window":
    latest_valid_start = (pd.to_datetime(max_date) - min_window).date()
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
# Ensure selected index and benchmark exist
if selected_index not in returns_df.columns or preferred_index not in returns_df.columns:
    st.error(f"Missing required column(s): {selected_index} or {preferred_index}")
    st.stop()

gross_crestcast = cumulative_returns[selected_index]
net_crestcast = gross_crestcast - monthly_fee
benchmark = cumulative_returns[preferred_index]


# Clean slice of working return data
returns_subset = pd.concat([
    benchmark.rename("Benchmark"),
    net_crestcast.rename("CrestCast")
], axis=1).dropna()

# Optional: use these for downstream stats/plots
benchmark = returns_subset["Benchmark"]
net_crestcast = returns_subset["CrestCast"]
blended_crestcast = (1 - lam) * benchmark + lam * net_crestcast


# --- Section: Percent Return Over Selected Time Range ---
st.subheader("📈 Net Total Return Over Selected Period (%)")

# Calculate cumulative returns
cum_benchmark = (1 + benchmark).cumprod() - 1
cum_crestcast = (1 + net_crestcast).cumprod() - 1
cum_blended = (1 + blended_crestcast).cumprod() - 1

# Build comparison DataFrame based on lambda
if macro_aware:
    if lam == 1.0:
        comparison_df = pd.DataFrame({
            f"{preferred_index} (Benchmark)": cum_benchmark,
            f"{selected_index} (Net of Fee)": cum_crestcast
        })
    else:
        comparison_df = pd.DataFrame({
            f"{preferred_index} (Benchmark)": cum_benchmark,
            f"{selected_index} ({tracking_error_label_choice})": cum_blended,
            f"{selected_index} (100% Net of Fee)": cum_crestcast
        })
else:
    comparison_df = pd.DataFrame({
        f"{preferred_index} (Benchmark)": cum_benchmark
    })

comparison_df = comparison_df.dropna()

# Plot with matplotlib
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

def safe_beta_alpha(port, bench, rf_series):
    """
    Aligns and cleans portfolio, benchmark, and risk-free series before computing beta and alpha.
    Returns (beta, alpha) or (np.nan, np.nan) if insufficient data.
    """
    if port is None or bench is None:
        return np.nan, np.nan

    df = pd.concat([
        port.rename("CrestCast"),
        bench.rename("Benchmark")
    ], axis=1).dropna()

    if df.empty or df.shape[0] < 2:
        return np.nan, np.nan

    if rf_series is not None:
        rf = rf_series.reindex(df.index).dropna()
        if rf.shape[0] < 2:
            rf = None
    else:
        rf = None

    return beta_alpha(df["CrestCast"], df["Benchmark"], rf=rf)


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
    *beta_alpha(named_crestcast, named_benchmark, rf=risk_free_series),
    sharpe_ratio(named_crestcast, rf=risk_free_series),
    tracking_error(named_crestcast, named_benchmark),
    information_ratio(named_crestcast, named_benchmark, rf=risk_free_series),
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
    sharpe_ratio(named_benchmark, rf=risk_free_series),
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
valid_data = returns_subset.dropna()
blended_crestcast = (1 - lam) * valid_data["Benchmark"] + lam * valid_data["CrestCast"]
benchmark = valid_data["Benchmark"]


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

    def safe_beta_alpha(port, bench, rf_series):
    """
    Aligns and cleans portfolio, benchmark, and risk-free series before computing beta and alpha.
    Returns (beta, alpha) or (np.nan, np.nan) if insufficient data.
    """
    if port is None or bench is None:
        return np.nan, np.nan

    df = pd.concat([
        port.rename("CrestCast"),
        bench.rename("Benchmark")
    ], axis=1).dropna()

    if df.empty or df.shape[0] < 2:
        return np.nan, np.nan

    if rf_series is not None:
        rf = rf_series.reindex(df.index).dropna()
        if rf.shape[0] < 2:
            rf = None
    else:
        rf = None

    return beta_alpha(df["CrestCast"], df["Benchmark"], rf=rf)


    # Periods to evaluate
    today = returns_subset.index[-1]
    periods = {
        "1 Year": today - pd.DateOffset(years=1),
        "5 Year": today - pd.DateOffset(years=5),
        "10 Year": today - pd.DateOffset(years=10),
        "Since Inception": returns_subset.index[0]
    }

    # Metrics to compute
    metrics = {
        "Ann. Return": lambda p, b: (annualized_return(p), annualized_return(b)),
        "Ann. Std Dev": lambda p, b: (annualized_std(p), annualized_std(b)),
        "Beta": lambda p, b: beta_alpha(p, b, rf=risk_free_series)[0],
        "Alpha": lambda p, b: beta_alpha(p, b, rf=risk_free_series)[1],
        "Sharpe Ratio": lambda p, b: (sharpe_ratio(p, rf=risk_free_series), sharpe_ratio(b, rf=risk_free_series)),
        "Information Ratio": lambda p, b: information_ratio(p, b, rf=risk_free_series),
        "Max Drawdown": lambda p, b: (max_drawdown(p), max_drawdown(b)),
        "Ulcer Ratio": lambda p, b: (ulcer_ratio(p, b), ulcer_ratio(b, b)),
        "Tracking Error": lambda p, b: tracking_error(p, b),
        "Up Capture": lambda p, b: (up_capture(p, b), up_capture(b, b)),
        "Down Capture": lambda p, b: (down_capture(p, b), down_capture(b, b)),
    }

    from collections import defaultdict
    nested_data = defaultdict(dict)

    for label, start_date in periods.items():
        port = returns_subset["CrestCast"].loc[start_date:]
        bench = returns_subset["Benchmark"].loc[start_date:]
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

    multi_index_df = pd.DataFrame(nested_data)
    multi_index_df.columns = pd.MultiIndex.from_tuples(multi_index_df.columns)
    multi_index_df = multi_index_df[["1 Year", "5 Year", "10 Year", "Since Inception"]]

    decimal_metrics = ["Beta", "Sharpe Ratio", "Ulcer Ratio", "Information Ratio"]

    def smart_format(val, metric_name):
        if isinstance(val, (int, float)):
            return f"{val:.2f}" if any(dm in metric_name for dm in decimal_metrics) else f"{val:.2%}"
        return val

    formatted_df = multi_index_df.copy()
    for metric in formatted_df.index:
        formatted_df.loc[metric] = formatted_df.loc[metric].apply(lambda x: smart_format(x, metric))

    st.dataframe(formatted_df, use_container_width=True)

    csv_df = multi_index_df.copy()
    csv_df.columns = [f"{p} - {s}" for p, s in csv_df.columns]
    csv_bytes = csv_df.reset_index().rename(columns={"index": "Metric"}).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Metric Summary by Period",
        data=csv_bytes,
        file_name="metrics_by_period.csv",
        mime="text/csv"
    )
# === Rolling 5-Year Alpha Summary ===
if st.checkbox("Show Rolling 5-Year Alpha Summary and Distribution"):
    rolling_window = 36  # 5 years
    alpha_values = []
    alpha_dates = []

    for i in range(rolling_window, len(returns_subset)):
        window = returns_subset.iloc[i - rolling_window:i]
        port = window["CrestCast"]
        bench = window["Benchmark"]

        if port.isnull().any() or bench.isnull().any():
            continue

        _, alpha = beta_alpha(port, bench, rf=risk_free_series.loc[window.index])
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

        # Bar chart of rolling alpha
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        alpha_series.plot(
            kind="bar",
            ax=ax2,
            color="#4A90E2",
            edgecolor="white",
            width=0.9
        )
        ax2.axhline(0, linestyle='--', color='gray', linewidth=1)
        ax2.set_title("Rolling 5-Year Alpha Over Time")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Annualized Alpha")

        tick_labels = []
        tick_positions = []
        for i, dt in enumerate(alpha_series.index):
            if dt.month == 1:
                tick_labels.append(dt.strftime('%Y'))
                tick_positions.append(i)

        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels, rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)

        # Download
        csv_bytes = alpha_series.reset_index().rename(
            columns={alpha_series.name: "Rolling 5-Year Alpha", "index": "Date"}
        ).to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Rolling Alpha Data (CSV)",
            data=csv_bytes,
            file_name="rolling_5y_alpha.csv",
            mime="text/csv"
        )
# === Rolling 5-Year Sharpe Comparison ===
if st.checkbox("Show Rolling 5-Year Sharpe Comparison"):
    rolling_window = 36
    crest_sharpes = []
    bench_sharpes = []
    dates = []

    for i in range(rolling_window, len(returns_subset)):
        window = returns_subset.iloc[i - rolling_window:i]

        port = window["CrestCast"]
        bench = window["Benchmark"]

        if port.isnull().any() or bench.isnull().any():
            continue

        crest_sharpes.append(sharpe_ratio(port, rf=risk_free_series.loc[window.index]))
        bench_sharpes.append(sharpe_ratio(bench, rf=risk_free_series.loc[window.index]))
        dates.append(window.index[-1])

    sharpe_df = pd.DataFrame({
        "Date": dates,
        "CrestCast Sharpe": crest_sharpes,
        "Benchmark Sharpe": bench_sharpes
    }).set_index("Date")

    percent_better_sharpe = (sharpe_df["CrestCast Sharpe"] > sharpe_df["Benchmark Sharpe"]).mean()
    avg_diff = (sharpe_df["CrestCast Sharpe"] - sharpe_df["Benchmark Sharpe"]).mean()

    st.markdown("### 📈 Rolling 5-Year Sharpe Ratio Comparison")
    st.markdown(f"- **% of 5-Year Windows Where CrestCast™ > Benchmark**: **{percent_better_sharpe:.1%}**")
    st.markdown(f"- **Average Sharpe Advantage (CrestCast™ minus Benchmark)**: **{avg_diff:.2f}**")

    # Line chart
    fig, ax = plt.subplots(figsize=(6, 3))
    sharpe_df.plot(ax=ax)
    ax.set_title("Rolling 5-Year Sharpe Ratio")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig)

    # Histogram of Sharpe Advantage
    sharpe_diff = sharpe_df["CrestCast Sharpe"] - sharpe_df["Benchmark Sharpe"]
    fig, ax = plt.subplots(figsize=(6, 3))
    sharpe_diff.hist(bins=30, edgecolor="blue", ax=ax)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Distribution of Sharpe Ratio Improvement (CrestCast™ - Benchmark)")
    ax.set_xlabel("Sharpe Advantage")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# === Final Summary Stat Row (Always Visible) ===
st.markdown("---")

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
        value="89% Alpha ⬆",
        delta="Sharpe outperformance: 76% of periods"
    )

with cols[2]:
    st.metric(
        label="📉 3-Year Windows",
        value="73% Alpha ⬆",
        delta="Sharpe outperformance: 60% of periods"
    )

st.markdown("---")

st.markdown("### ☎️ Let's Talk")
st.markdown(
    "#### _Whether you're building ETFs, models, or SMAs & personalized portfolios, a CrestCast™ license can add power to your solution. "
    "[Schedule a quick call](https://intervallumtech.com/meeting) to explore fit. Macro-Aware Factor Rotation is here._"
)
st.markdown("— *Powered by CrestCast™*")
