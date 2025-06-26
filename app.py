import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from PIL import Image
import streamlit as st

# Optional: set page metadata
st.set_page_config(
    page_title="Passive 3.0 Portfolio Builder",
    layout="wide"
)

# Load and display banner
banner = Image.open("banner.png")
st.image(banner, use_column_width=True)


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

# === Streamlit App ===
st.set_page_config(page_title="CrestCast Overlay Builder", layout="centered")
st.title("Passive3.0™ Overlay Builder")

# --- Section 1: Client Info ---
st.header("1. Client Information")
client_name = st.text_input("Client Name")
account_type = st.selectbox("Account Type", ["Individual", "Joint", "Trust", "IRA", "Corporate", "Other"]
)
# --- Section 2: Index Selection ---
st.header("2. Select Core Index")
index_options = {
    "S&P 500 (SPY)": "SPY",
    "Russell 3000 (IWV)": "IWV"
}
selected_label = st.selectbox("Preferred Index", list(index_options.keys()))
preferred_index = index_options[selected_label]

# --- Section 3: Add Macro-Aware Overlay ---
st.header("3. Add Macro-Aware Overlay")

macro_aware = st.checkbox("Enable Macro-Aware Overlay")

if macro_aware:
    st.markdown("This overlay helps your portfolio respond to changing economic conditions using a data-driven model.")

    # Overlay Fee Dropdown
    fee_bps = st.selectbox("Overlay Fee (MSRP)", [50, 35, 20], index=0)
    annual_fee = fee_bps / 10000
    monthly_fee = annual_fee / 12

    # Friendly Tracking Error Dropdown
    tracking_error_label = "How closely should your portfolio follow the index?"
    tracking_error = st.selectbox(
        tracking_error_label,
        options=["Very closely", "Somewhat closely", "Flexible"],
        index=0
    )

    # Map investor-friendly terms to λ
    lambda_values = {
        "Very closely": 0.2,
        "Somewhat closely": 0.5,
        "Flexible": 1.0
    }
    lam = lambda_values[tracking_error]

    # Optional Email Opt-In
    email_opt_in = st.checkbox("📬 Please send me an email with monthly commentary on regime outlook and implications for my portfolio.")

else:
    # If macro overlay is off, set safe defaults
    fee_bps = 0
    monthly_fee = 0.0
    tracking_error = "Not applicable"
    lam = 0.0
    email_opt_in = False
# --- Section 5: Select Time Period or Stress Scenario ---
st.header("4. Select Time Period for Analysis")

min_date = returns_df.index.min().to_pydatetime().date()
max_date = returns_df.index.max().to_pydatetime().date()

stress_periods = {
    "Custom Date Range": (None, None),
    "Financial Crisis (2007–2009)": ("2007-10-01", "2009-03-31"),
    "Post-Financial Crisis Bull Run (2009–2014)": ("2009-04-01", "2014-05-31"),
    "Oil Crash (2014–2016)": ("2014-06-01", "2016-02-29"),
    "Pandemic Crash (2020)": ("2020-02-01", "2020-04-30"),
    "Post-COVID Bull Run (2020–2021)": ("2020-05-01", "2021-12-31"),
    "Inflation Regime (2022–2023)": ("2022-01-01", "2023-10-31"),
    "Recent Volatility (2024–Present)": ("2024-01-01", returns_df.index.max().strftime("%Y-%m-%d"))
}


# Dropdown to choose stress period
selected_period = st.selectbox("Select Predefined Stress Period", list(stress_periods.keys()))

# Determine dates based on selection
start_str, end_str = stress_periods[selected_period]
if start_str and end_str:
    start_date = pd.to_datetime(start_str)
    end_date = pd.to_datetime(end_str)
    st.caption(f"Showing performance from **{start_date.date()}** to **{end_date.date()}**")
else:
    # Default to custom range slider
    min_date = returns_df.index.min().to_pydatetime().date()
    max_date = returns_df.index.max().to_pydatetime().date()
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
st.subheader("📈 Cumulative Return Comparison")

# Precompute cumulative returns
cum_benchmark = cumulative_return(benchmark)
cum_blended = cumulative_return(blended_crestcast)
cum_crestcast = cumulative_return(net_crestcast)

# Combine into DataFrame
comparison_df = pd.DataFrame({
    f"{preferred_index} (Benchmark)": cum_benchmark,
    f"CrestCast Overlay ({tracking_error})": cum_blended,
    "CrestCast (100% Net of Fee)": cum_crestcast
})

# Drop NaNs to avoid blank charts
comparison_df = comparison_df.dropna()

# Plot the chart
if not comparison_df.empty:
    st.line_chart(comparison_df)
else:
    st.warning("Not enough data to plot. Please try a different date range.")


# --- Performance Summary Table ---
st.subheader("📊 Performance Summary")

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
    "Sharpe Ratio", "Max Drawdown",
    "Up Capture", "Down Capture", "Return Outperformance"
]

# CrestCast metrics
crestcast_metrics = [
    annualized_return(blended_crestcast),
    annualized_std(blended_crestcast),
    *beta_alpha(blended_crestcast, benchmark),
    sharpe_ratio(blended_crestcast),
    max_drawdown(blended_crestcast),
    up_capture(blended_crestcast, benchmark),
    down_capture(blended_crestcast, benchmark),
    return_diff(blended_crestcast, benchmark)
]

# Benchmark metrics
benchmark_metrics = [
    annualized_return(benchmark),
    annualized_std(benchmark),
    None, None,
    sharpe_ratio(benchmark),
    max_drawdown(benchmark),
    1.0, 1.0, 0.0
]

# Format results
def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, (float, np.float64)):
        return f"{x:.2%}" if abs(x) < 10 else f"{x:.2f}"
    return str(x)

# Create and format table
summary_df = pd.DataFrame({
    "Metric": metrics,
    f"CrestCast Overlay ({tracking_error})": crestcast_metrics,
    "Benchmark": benchmark_metrics
})

summary_df = summary_df.applymap(fmt)
st.table(summary_df)

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

