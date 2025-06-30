import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

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
def annualized_return(r): return (1 + r.mean()) ** 12 - 1
def annualized_std(r): return r.std() * np.sqrt(12)
def beta_alpha(port, bench):
    df = pd.concat([port, bench], axis=1).dropna()
    if df.shape[0] < 2: return np.nan, np.nan
    cov = np.cov(df.iloc[:, 0], df.iloc[:, 1])
    beta = cov[0, 1] / cov[1, 1]
    alpha = annualized_return(df.iloc[:, 0]) - beta * annualized_return(df.iloc[:, 1])
    return beta, alpha
def sharpe_ratio(r, rf=0.0): return ((r - rf/12).mean() / r.std()) * np.sqrt(12)
def max_drawdown(r):
    cumulative = (1 + r).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()
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
# Passive 3.0™ Macro Overlay  
### Direct Indexing Demonstration | Powered by Intervallum Technologies
This demo illustrates how the Passive 3.0™ macro overlay can dynamically enhance any core index using regime-aware factor rotation.
""")

# === Section 1: Simulation Parameters ===
st.header("1. Simulation Setup")
client_name = st.text_input("Client (or Demo Label)", value="")
account_type = st.selectbox("Account Type", ["Individual", "Joint", "Trust", "IRA", "Corporate", "Other"])

# === Section 2: Select Base Index ===
st.header("2. Select Core Index for Overlay")
index_options = {
    "Russell 3000 (IWV)": "IWV",
    "S&P 500 (SPY)": "SPY"
    
}
selected_label = st.selectbox("Preferred Index", list(index_options.keys()))
preferred_index = index_options[selected_label]

# === Section 3: Activate Overlay Logic ===
st.header("3. Activate Macro-Aware Overlay")
macro_aware = st.checkbox("Enable Macro-Aware Overlay?")


if macro_aware:
    st.markdown("This overlay helps your portfolio respond to changing economic conditions using advanced analytics. Learn More.")

    # Overlay Fee Dropdown
    fee_bps = st.selectbox("Overlay Fee (basis points))", [0, 20, 35, 50], index=0)
    annual_fee = fee_bps / 10000
    monthly_fee = annual_fee / 12

    # Friendly Tracking Error Dropdown
    tracking_error_label = "How closely should your portfolio follow the index?"
    tracking_error_label_choice = st.selectbox(
        tracking_error_label,
        options=["Flexible", "Somewhat closely", "Very closely"],
        index=0
    )


    # Map investor-friendly terms to λ
    lambda_values = {
        "Very closely": 0.2,
        "Somewhat closely": 0.5,
        "Flexible": 1.0
    }
    lam = lambda_values[tracking_error_label_choice]


    # Optional Email Opt-In
    email_opt_in = st.checkbox("📬 Please send me an email with monthly commentary on regime outlook and implications for my portfolio.")

else:
    # If macro overlay is off, set safe defaults
    fee_bps = 0
    monthly_fee = 0.0
    tracking_error = "Not applicable"
    lam = 0.0
    email_opt_in = False
# --- Section 4: Select Time Period ---
st.header("4. Select Time Period for Analysis")

min_date = returns_df.index.min().to_pydatetime().date()
max_date = returns_df.index.max().to_pydatetime().date()

# Custom date range slider only (stress scenarios removed)
date_range = st.slider(
    "Select Custom Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date)
)

start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])
st.caption(f"Showing performance from **{start_date.date()}** to **{end_date.date()}**")

# Slice the return data
filtered_returns = returns_df.loc[start_date:end_date]


# --- Extract Data for Chart + Stats ---
benchmark = filtered_returns[preferred_index]
gross_crestcast = filtered_returns['P:1529191']
net_crestcast = gross_crestcast - monthly_fee

valid_data = pd.concat([benchmark, net_crestcast], axis=1).dropna()
benchmark = valid_data.iloc[:, 0]
net_crestcast = valid_data.iloc[:, 1]

# Blend based on tracking error (λ)
blended_crestcast = (1 - lam) * benchmark + lam * net_crestcast

# Plot the chart

# --- Cumulative Return Chart ---
st.subheader("📈 Growth of $1,000 (net of fees)")

# Precompute cumulative return paths starting at $1,000
cum_benchmark = 1000 * cumulative_return(benchmark)

if macro_aware:
    cum_blended = 1000 * cumulative_return(blended_crestcast)
    cum_crestcast = 1000 * cumulative_return(net_crestcast)

    # Combine into DataFrame with macro overlay
    comparison_df = pd.DataFrame({
        f"{preferred_index} (Benchmark)": cum_benchmark,
        f"CrestCast Overlay ({tracking_error})": cum_blended,
        "CrestCast (100% Net of Fee)": cum_crestcast
    })
else:
    # Show benchmark only
    comparison_df = pd.DataFrame({
        f"{preferred_index} (Benchmark)": cum_benchmark
    })

# Drop NaNs to avoid blank charts
comparison_df = comparison_df.dropna()

# Plot the chart
if not comparison_df.empty:
    st.line_chart(comparison_df)
else:
    st.warning("Not enough data to plot. Please try a different date range.")

# --- Optional Toggle for Rolling Outperformance ---
st.markdown("#### 📊 Performance View Options")

show_relative_perf = st.checkbox("🔁 View Rolling Relative Performance", value=False)

if show_relative_perf:
    st.subheader("📉 Rolling 10-Year Relative Performance")

    # Calculate 3-year rolling performance difference
    # Calculate rolling 3-year annualized return
    crest_rolling_ann = net_crestcast.rolling(window=120).apply(lambda r: (1 + r).prod()**(1/3) - 1)
    bench_rolling_ann = benchmark.rolling(window=120).apply(lambda r: (1 + r).prod()**(1/3) - 1)
    
    # Compute the annualized spread
    rel_perf = crest_rolling_ann - bench_rolling_ann
    rel_perf = rel_perf.dropna()

    # Plot as bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["green" if val >= 0 else "red" for val in rel_perf]
    ax.bar(rel_perf.index, rel_perf.values, color=colors, width=20)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Rolling 10-Year Outperformance vs. Benchmark")
    ax.set_ylabel("CrestCast – Benchmark (Annualized Return)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig)

    st.caption("Each bar represents CrestCast’s outperformance or underperformance over the prior 3 years. Green bars indicate periods of relative outperformance; red bars indicate relative lag.")
# --- Performance Summary Table ---
st.subheader("📊 Performance Summary (net of fees)")



# New: Up/down capture & return delta
def up_capture(port, bench):
    mask = bench > 0
    return port[mask].mean() / bench[mask].mean() if bench[mask].mean() != 0 else np.nan

def down_capture(port, bench):
    mask = bench < 0
    return port[mask].mean() / bench[mask].mean() if bench[mask].mean() != 0 else np.nan

def return_diff(port, bench):
    return annualized_return(port) - annualized_return(bench)

# Metrics to display
metrics = [
    "Annualized Return", "Annualized Std Dev", 
    "Beta vs Benchmark", "Alpha vs Benchmark", 
    "Sharpe Ratio", "Information Ratio",  # <-- new
    "Max Drawdown", "Up Capture", "Down Capture", 
    "Return Outperformance"
]


# CrestCast metrics
crestcast_metrics = [
    annualized_return(blended_crestcast),
    annualized_std(blended_crestcast),
    *beta_alpha(blended_crestcast, benchmark),
    sharpe_ratio(blended_crestcast),
    information_ratio(blended_crestcast, benchmark),  # <-- new
    max_drawdown(blended_crestcast),
    up_capture(blended_crestcast, benchmark),
    down_capture(blended_crestcast, benchmark),
    return_diff(blended_crestcast, benchmark)
]


benchmark_metrics = [
    annualized_return(benchmark),
    annualized_std(benchmark),
    None, None,
    sharpe_ratio(benchmark),
    None,  # Information Ratio doesn't apply
    max_drawdown(benchmark),
    1.0, 1.0, 0.0
]


# Format function with metric context
def fmt(x, metric=None):
    if x is None:
        return "-"
    if isinstance(x, (float, np.float64)):
        if metric in ["Sharpe Ratio", "Information Ratio"]:
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
        f"CrestCast Overlay ({tracking_error})": fmt(cc_val, metric),
        "Benchmark": fmt(bench_val, metric)
    })

summary_df = pd.DataFrame(formatted_data)
st.table(summary_df)

import matplotlib.pyplot as plt

# --- Rolling 3-Year IR with Drawdown Overlay ---
import matplotlib.ticker as ticker

# --- Enhanced Chart Section ---
if macro_aware:
    st.markdown("### Rolling 3-Year Information Ratio vs. Drawdown Context")

    # Ensure clean data
    valid_data = pd.concat([blended_crestcast, benchmark], axis=1).dropna()
    blended_crestcast = valid_data.iloc[:, 0]
    benchmark = valid_data.iloc[:, 1]

    # --- Compute Rolling IR ---
    rolling_window = 36
    ir_values = []
    dates = []

    for i in range(rolling_window, len(blended_crestcast)):
        port = blended_crestcast.iloc[i - rolling_window:i]
        bench = benchmark.iloc[i - rolling_window:i]

        if port.isnull().any() or bench.isnull().any():
            ir_values.append(np.nan)
            continue

        x = bench.values
        y = port.values
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        beta = np.cov(x, y)[0, 1] / np.var(x)
        alpha = y_mean - beta * x_mean
        residuals = y - (alpha + beta * x)
        tracking_err = np.std(residuals) * np.sqrt(12)
        annual_alpha = alpha * 12
        ir = annual_alpha / tracking_err if tracking_err != 0 else np.nan

        ir_values.append(ir)
        dates.append(port.index[-1])

    ir_series = pd.Series(ir_values, index=dates).dropna()

    if not ir_series.empty:
        # --- Drawdown Calculation ---
        cumulative_crest = (1 + blended_crestcast).cumprod()
        cumulative_bench = (1 + benchmark).cumprod()
        dd_crest = (cumulative_crest / cumulative_crest.cummax()) - 1
        dd_bench = (cumulative_bench / cumulative_bench.cummax()) - 1
        dd_crest_aligned = dd_crest.loc[ir_series.index]
        dd_bench_aligned = dd_bench.loc[ir_series.index]

        # --- Plot ---
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # IR line (primary axis)
        ax1.plot(ir_series.index, ir_series.values, label="Rolling 3-Year IR", color="#1f77b4", linewidth=2)
        ax1.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="IR = 0.5 threshold")
        ax1.set_ylabel("Information Ratio", fontsize=10, color="#1f77b4")
        ax1.set_ylim(-0.5, 1.75)
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(6))
        ax1.tick_params(axis='y', labelcolor="#1f77b4", labelsize=9)
        ax1.grid(True, linestyle="--", alpha=0.3)

        # Drawdown area (secondary axis)
        ax2 = ax1.twinx()
        ax2.fill_between(dd_crest_aligned.index, dd_crest_aligned.values, 0, color='green', alpha=0.15, label="CrestCast Drawdown")
        ax2.plot(dd_bench_aligned.index, dd_bench_aligned.values, label="Benchmark Drawdown", color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7)
        ax2.set_ylabel("Drawdown", fontsize=10, color="gray")
        ax2.set_ylim(-0.6, 0.05)
        ax2.tick_params(axis='y', labelcolor="gray", labelsize=9)

        # Legend
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2,
                   loc="upper center", bbox_to_anchor=(0.5, -0.15),
                   ncol=2, frameon=False)

        fig.tight_layout()
        st.pyplot(fig)

        # --- Institutional Caption ---
        st.caption(
            "**Interpretation:** This chart reveals how CrestCast's rolling 3-year Information Ratio moves in relation to market drawdowns. "
            "During periods of major macro stress (e.g., 2008, 2020, 2022), the IR often compresses — not due to strategy failure, "
            "but because CrestCast’s lower-beta profile reduces alpha in falling markets even as it mitigates losses significantly. "
            "The drawdown overlays make this visible: IR may dip, but capital is preserved. "
            "Over the full cycle, this is what sustains long-term, risk-adjusted alpha."
        )
    else:
        st.warning("Not enough clean data to calculate rolling IR or drawdowns.")


# --- Section 6: Implementation Add-Ons (Non-Performance Adjusted) ---
st.header("6. Implementation Add-Ons (Non-Performance Adjusted)")

tax_aware = st.checkbox("Enable Tax-Aware Overlay")

value_screens = st.multiselect(
    "Apply Value Screens (optional)",
    options=[
        "Exclude Tobacco",
        "Exclude Fossil Fuels",
        "Exclude Defense Contractors",
        "ESG Tilt",
        "Faith-Based Screen",
        "Climate Alignment"
    ]
)


# --- Section 7: Summary Recap ---
st.header("7. Summary")
st.write(f"**Client:** {client_name if client_name else '—'}")
st.write(f"**Account Type:** {account_type}")
st.write(f"**Preferred Index:** {preferred_index}")
st.write(f"**Overlay Fee:** {fee_bps} bps")
st.write(f"**Macro-Aware Overlay:** {'Enabled' if macro_aware else 'Disabled'}")
if macro_aware:
    st.write(f"**Tracking Error Target:** {tracking_error}")
st.write(f"**Tax-Aware Overlay:** {'Enabled' if tax_aware else 'Disabled'}")
st.write(f"**Value Screens:** {', '.join(value_screens) if value_screens else 'None'}")

