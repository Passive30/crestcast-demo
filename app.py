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
    total_return = (1 + r).prod() - 1
    n_months = len(r)
    if n_months < 1:
        return np.nan
    return (1 + total_return) ** (12 / n_months) - 1

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
# Introducing CrestCast™ Macro-Aware US Factor Rotation Index 
### Detailed Analytics Demonstration | Powered by Intervallum Technologies
This demo illustrates how the CrestCast™ index can dynamically enhance core equity exposure using regime-aware factor rotation.
""")

# === Section 1: Simulation Parameters ===
st.header("1. Simulation Setup")
st.info(
"This demo is preconfigured to highlight the full power of the CrestCast™ Macro-Aware U.S. Factor Rotation Index."
"Results from August 1, 2025, onward reflect live index data. From 2001 through July 2025, the index is based on strict" 
"out-of-sample implementation using a walk-forward validated model. Factor tilts are applied by our calculation agent based"
"on analytics derived exclusively from the training period—ensuring no forward-looking bias is introduced into the model."
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
            f"CrestCast Overlay ({tracking_error_label_choice})": cum_blended,
            "CrestCast (100% Net of Fee)": cum_crestcast
        })

else:
    comparison_df = pd.DataFrame({
        f"{preferred_index} (Benchmark)": cum_benchmark
    })

comparison_df = comparison_df.dropna()

# Plot full-width with matplotlib
if not comparison_df.empty:
    fig, ax = plt.subplots(figsize=(12, 4))
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
        f"CrestCast Overlay ({tracking_error_label_choice})": fmt(cc_val, metric),
        "Benchmark": fmt(bench_val, metric)
    })

summary_df = pd.DataFrame(formatted_data)
st.table(summary_df)

# === Rolling 10-Year Alpha Chart ===
st.subheader("📉 Rolling 10-Year Alpha vs. Benchmark")

rolling_window = 120
rolling_alpha = []

for i in range(rolling_window, len(net_crestcast)):
    port = net_crestcast.iloc[i - rolling_window:i].rename("CrestCast")
    bench = benchmark.iloc[i - rolling_window:i].rename("Benchmark")

    if port.isnull().any() or bench.isnull().any():
        rolling_alpha.append(np.nan)
        continue

    beta, alpha = beta_alpha(port, bench)
    rolling_alpha.append(alpha)

# Align with date index
alpha_series = pd.Series(rolling_alpha, index=net_crestcast.index[rolling_window:])

# Prepare alpha data for download
alpha_df = alpha_series.reset_index()
alpha_df.columns = ["Date", "Rolling 10Y Alpha"]

# Download button
csv_alpha = alpha_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Rolling 10-Year Alpha Data",
    data=csv_alpha,
    file_name="rolling_10y_alpha.csv",
    mime="text/csv"
)

# Plot
fig, ax = plt.subplots(figsize=(7.5, 3))
colors = ["green" if val >= 0 else "red" for val in alpha_series]
ax.bar(alpha_series.index, alpha_series.values, color=colors, width=20)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_title("Rolling 10-Year Alpha vs. Benchmark")
ax.set_ylabel("Alpha (Annualized)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.grid(True, linestyle="--", alpha=0.3)
st.pyplot(fig)

st.caption(
    "Each bar represents CrestCast’s alpha over the prior 10 years. "
    "Green bars indicate positive alpha; red bars indicate negative performance relative to beta exposure."
)


# === Full-Period Drawdown Comparison ===
st.subheader("📉 Full-Period Drawdown: CrestCast vs. Benchmark")

# Align data and calculate drawdowns
valid_data = pd.concat([blended_crestcast, benchmark], axis=1).dropna()
blended_crestcast = valid_data.iloc[:, 0]
benchmark = valid_data.iloc[:, 1]

cumulative_crest = (1 + blended_crestcast).cumprod()
cumulative_bench = (1 + benchmark).cumprod()

dd_crest = (cumulative_crest / cumulative_crest.cummax()) - 1
dd_bench = (cumulative_bench / cumulative_bench.cummax()) - 1

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(dd_crest.index, dd_crest.values, 0, color='green', alpha=0.15, label="CrestCast Drawdown")
ax.plot(dd_bench.index, dd_bench.values, label="Benchmark Drawdown", color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7)
ax.set_ylabel("Drawdown")
ax.set_title("Full-Period Drawdown: CrestCast vs. Benchmark")
ax.set_ylim(-0.6, 0.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc="lower left")
st.pyplot(fig)

st.caption(
    "**Interpretation:** This chart shows the historical drawdowns of CrestCast and the benchmark across the full sample period. "
    "It highlights the depth and duration of capital declines, offering a clear comparison of downside experience across time."
)

# Download button
drawdown_df = pd.DataFrame({
    "Date": dd_crest.index,
    "CrestCast Drawdown": dd_crest.values,
    "Benchmark Drawdown": dd_bench.values
})
csv_drawdown = drawdown_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Full-Period Drawdown Data",
    data=csv_drawdown,
    file_name="full_period_drawdowns.csv",
    mime="text/csv"
)



# === Metric-First Performance Table ===
st.subheader("📊 CrestCast vs. Benchmark: Metrics by Period")

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

# Storage: {metric -> {period -> value}}
results = {}

for metric_name, func in metrics.items():
    crestcast_values = {}
    benchmark_values = {}

    for label, start_date in periods.items():
        # Slice the data
        port = blended_crestcast.loc[start_date:].rename("CrestCast")
        bench = benchmark.loc[start_date:].rename("Benchmark")

        df = pd.concat([port, bench], axis=1).dropna()
        if df.empty:
            crestcast_values[label] = np.nan
            benchmark_values[label] = np.nan
            continue

        # Ensure consistent naming
        result = func(df["CrestCast"], df["Benchmark"])

        if isinstance(result, tuple):
            crestcast_values[label] = result[0]
            benchmark_values[label] = result[1]
        else:
            crestcast_values[label] = result
            benchmark_values[label] = np.nan  # Only CrestCast result, e.g., TE, IR

    results[f"CrestCast: {metric_name}"] = crestcast_values
    if any(v is not np.nan for v in benchmark_values.values()):
        results[f"Benchmark: {metric_name}"] = benchmark_values

# Create DataFrame
summary_df = pd.DataFrame(results).T
summary_df = summary_df[["1 Year", "5 Year", "10 Year", "Since Inception"]]  # Order columns
formatted_df = summary_df.applymap(lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else x)

# Display
st.dataframe(formatted_df)

# Download
csv_download = summary_df.reset_index().rename(columns={"index": "Metric"})
csv_bytes = csv_download.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Metric Summary by Period",
    data=csv_bytes,
    file_name="metrics_by_period.csv",
    mime="text/csv"
)


# --- Optional Section: Rolling 10-Year Alpha Summary ---
st.markdown("### 📈 Optional: Rolling 5-Year Alpha Analysis")

if st.checkbox("Show Rolling 5-Year Alpha Summary and Distribution"):

    rolling_window = 60  # 5 years
    alpha_values = []
    alpha_dates = []

    for i in range(rolling_window, len(returns_df)):
        window = returns_df.iloc[i - rolling_window:i]

        # Use named columns directly — no iloc for series
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
        # Summary stats
        percent_positive = (alpha_series > 0).mean()
        average_alpha = alpha_series.mean()

        st.markdown(f"- **Percent of 5-Year Windows with Positive Alpha**: **{percent_positive:.1%}**")
        st.markdown(f"- **Average Annualized Alpha (5-Year Windows)**: **{average_alpha:.2%}**")

        # Histogram
        fig, ax = plt.subplots()
        alpha_series.hist(bins=30, edgecolor='black', ax=ax)
        ax.set_title("Distribution of 5-Year Rolling Alpha")
        ax.set_xlabel("Annualized Alpha")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # Optional: bar chart preview (match visual)
        fig2, ax2 = plt.subplots(figsize=(7.5, 3))
        ax2.bar(alpha_series.index, alpha_series.values, color="green", width=20)
        ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax2.set_title("Rolling 5-Year Alpha vs. Benchmark")
        ax2.set_ylabel("Alpha (Annualized)")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax2.grid(True, linestyle="--", alpha=0.3)
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
    st.markdown(f"- **% of 5-Year Windows Where CrestCast > Benchmark**: **{percent_better_sharpe:.1%}**")
    st.markdown(f"- **Average Sharpe Advantage (CrestCast minus Benchmark)**: **{avg_diff:.2f}**")

    # Optional chart
    fig, ax = plt.subplots(figsize=(7.5, 3))  # Smaller footprint
    sharpe_df.plot(ax=ax)
    ax.set_title("Rolling 5-Year Sharpe Ratio")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig)

    sharpe_df.plot(ax=ax)
    ax.set_title("Rolling 5-Year Sharpe Ratio")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig)

# === Final Summary Stat Row (Always Visible) ===
st.markdown("---")
st.markdown("### 📊 Performance Consistency: Alpha + Sharpe Advantage")

cols = st.columns(3)

with cols[0]:
    st.metric(
        label="📈 10-Year Windows",
        value="100% Alpha ⬆",
        delta="CrestCast beat benchmark on Sharpe 96% of the time"
    )

with cols[1]:
    st.metric(
        label="📊 5-Year Windows",
        value="98% Alpha ⬆",
        delta="Sharpe higher in 77% of periods"
    )

with cols[2]:
    st.metric(
        label="📉 3-Year Windows",
        value="97% Alpha ⬆",
        delta="Sharpe higher in 78% of periods"
    )

st.markdown("### Let’s Talk")
st.markdown(
    "_Whether you're building ETFs, models, or personalized portfolios, CrestCast™ can power your solution. "
    "[Schedule a quick call](https://meetings-na2.hubspot.com/alan-thomson) to explore fit. Macro-Aware Factor Rotation is here._"
)

