import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Simulated demo setup
dates = pd.date_range(start="2010-01-01", periods=200, freq="M")
np.random.seed(0)
benchmark = pd.Series(np.random.normal(0.01, 0.04, size=200), index=dates)
net_crestcast = pd.Series(np.random.normal(0.012, 0.035, size=200), index=dates)

# Create summary metrics (fake for example)
summary_df = pd.DataFrame({
    "Metric": ["Annualized Return", "Volatility", "Max Drawdown"],
    "CrestCast": ["9.2%", "12.1%", "-15.6%"],
    "Benchmark": ["7.8%", "13.3%", "-21.4%"]
})

# === Section: Toggle ===
st.markdown("#### 📊 Performance View Options")
show_relative_perf = st.checkbox("🔁 View Rolling Relative Performance", value=False)

# === Section: Cumulative Chart (Always Shows When Toggle is Off) ===
if not show_relative_perf:
    st.subheader("📈 Growth of $1,000 (net of fees)")
    crest_cum = (1 + net_crestcast).cumprod()
    bench_cum = (1 + benchmark).cumprod()
    st.line_chart(pd.DataFrame({"CrestCast": crest_cum, "Benchmark": bench_cum}))

    # === Performance Summary Table (Always visible here) ===
    st.subheader("📋 Performance Metrics (Net of Fees)")
    st.table(summary_df)

# === Section: Rolling Performance Charts (Only When Toggle is On) ===
if show_relative_perf:
    st.subheader("📉 Rolling 10-Year Annualized Outperformance vs. Benchmark")
    crest_rolling_ann = net_crestcast.rolling(window=120).apply(lambda r: (1 + r).prod()**(1/10) - 1)
    bench_rolling_ann = benchmark.rolling(window=120).apply(lambda r: (1 + r).prod()**(1/10) - 1)
    rel_perf = crest_rolling_ann - bench_rolling_ann
    rel_perf = rel_perf.dropna()

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["green" if val >= 0 else "red" for val in rel_perf]
    ax.bar(rel_perf.index, rel_perf.values, color=colors, width=20)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Rolling 10-Year Outperformance vs. Benchmark")
    ax.set_ylabel("CrestCast – Benchmark (Annualized Return)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig)

    st.caption("Each bar represents CrestCast’s outperformance or underperformance over the prior 10 years. Green bars indicate periods of relative outperformance; red bars indicate relative lag.")

    st.subheader("📈 Rolling 10-Year Information Ratio vs. Drawdown")
    rolling_window = 120
    ir_values = []
    dates = []

    for i in range(rolling_window, len(net_crestcast)):
        port = net_crestcast.iloc[i - rolling_window:i]
        bench = benchmark.iloc[i - rolling_window:i]
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
    crest_cum = (1 + net_crestcast).cumprod()
    bench_cum = (1 + benchmark).cumprod()
    dd_crest = (crest_cum / crest_cum.cummax()) - 1
    dd_bench = (bench_cum / bench_cum.cummax()) - 1
    dd_crest_aligned = dd_crest.loc[ir_series.index]
    dd_bench_aligned = dd_bench.loc[ir_series.index]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(ir_series.index, ir_series.values, label="Rolling 10-Year IR", color="#1f77b4", linewidth=2)
    ax1.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="IR = 0.5 threshold")
    ax1.set_ylabel("Information Ratio", fontsize=10, color="#1f77b4")
    ax1.set_ylim(-0.5, 1.75)
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(6))
    ax1.tick_params(axis='y', labelcolor="#1f77b4", labelsize=9)
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.fill_between(dd_crest_aligned.index, dd_crest_aligned.values, 0, color='green', alpha=0.15, label="CrestCast Drawdown")
    ax2.plot(dd_bench_aligned.index, dd_bench_aligned.values, label="Benchmark Drawdown", color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.set_ylabel("Drawdown", fontsize=10, color="gray")
    ax2.set_ylim(-0.6, 0.05)
    ax2.tick_params(axis='y', labelcolor="gray", labelsize=9)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

    fig.tight_layout()
    st.pyplot(fig)

    st.caption("This chart shows CrestCast’s rolling 10-year Information Ratio with drawdowns overlaid. It illustrates how downside mitigation impacts alpha delivery over a full cycle, even when risk-adjusted returns temporarily compress.")

