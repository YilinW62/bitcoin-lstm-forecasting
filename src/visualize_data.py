"""
Bitcoin data visualization script
Show historical data and forecasting illustrations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Read data
print("Load data...")
df = pd.read_excel('Bitcoin_Historical_Data_10years.xlsx')

# Data preprocessing
if 'Price' in df.columns:
    df = df.rename(columns={'Price': 'Close'})

df['Date'] = pd.to_datetime(df['Date'])

if df['Close'].dtype == 'object':
    df['Close'] = df['Close'].str.replace(',', '').astype(float)

# Ensure chronological order
if df['Date'].iloc[0] > df['Date'].iloc[-1]:
    df = df.sort_values('Date', ascending=True).reset_index(drop=True)

df = df.dropna(subset=['Close'])
df = df[df['Close'] > 0].reset_index(drop=True)

print(f"Date range: {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}")
print(f"Observations: {len(df)}")
print(f"Latest price: ${df['Close'].iloc[-1]:,.2f}")

# ============================================================================
# 1. Full historical price chart
# ============================================================================
print("\nGenerate plot 1: Full historical price chart...")

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(df.index, df['Close'], linewidth=1.5, color='blue', alpha=0.8)
ax.fill_between(df.index, df['Close'], alpha=0.3, color='lightblue')

# Add reference price lines
max_price = df['Close'].max()
min_price = df['Close'].min()
current_price = df['Close'].iloc[-1]

ax.axhline(y=max_price, color='red', linestyle='--', linewidth=1, alpha=0.5, label=f'Historical maximum: ${max_price:,.0f}')
ax.axhline(y=current_price, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Latest price: ${current_price:,.0f}')

ax.set_title('Bitcoin historical prices (2015-2025)', fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Price (USD)', fontsize=14)
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)

# Set x-axis date labels
n_ticks = 10
tick_indices = np.linspace(0, len(df)-1, n_ticks, dtype=int)
ax.set_xticks(tick_indices)
ax.set_xticklabels([df['Date'].iloc[i].strftime('%Y-%m') for i in tick_indices], rotation=45)

plt.tight_layout()
plt.savefig('bitcoin_full_history.png', dpi=300, bbox_inches='tight')
print("Saved: bitcoin_full_history.png")
plt.close()

# ============================================================================
# 2. Price series over the last year
# ============================================================================
print("\nGenerate plot 2: Price series over the last year...")

df_recent = df.tail(365).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df_recent.index, df_recent['Close'], linewidth=2, color='blue', marker='o', 
        markersize=2, alpha=0.8, label='Actual price')

# Add moving averages
ma_7 = df_recent['Close'].rolling(window=7).mean()
ma_30 = df_recent['Close'].rolling(window=30).mean()

ax.plot(df_recent.index, ma_7, linewidth=1.5, color='orange', alpha=0.7, label='7-day moving average')
ax.plot(df_recent.index, ma_30, linewidth=1.5, color='red', alpha=0.7, label='30-day moving average')

ax.set_title('Bitcoin prices over the last year', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Days', fontsize=12)
ax.set_ylabel('Price (USD)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bitcoin_recent_year.png', dpi=300, bbox_inches='tight')
print("Saved: bitcoin_recent_year.png")
plt.close()

# ============================================================================
# 3. Forecast illustration (simulated)
# ============================================================================
print("\nGenerate plot 3: Forecast illustration...")

# Use the last 60 days of observed data
TIME_STEP = 60
PRED_DAYS = 3

last_60_days = df['Close'].tail(TIME_STEP).values
last_60_dates = df['Date'].tail(TIME_STEP).values

# Simple forecast: Apply small changes to the last observed price
current_price = last_60_days[-1]
# Simulated forecasts (using simple linear extrapolation + random noise)
trend = (last_60_days[-1] - last_60_days[-10]) / 10
future_predictions = []
for i in range(PRED_DAYS):
    noise = np.random.normal(0, current_price * 0.01)  # 1% noise
    pred = current_price + trend * (i + 1) + noise
    future_predictions.append(pred)

# Generate future dates
last_date = df['Date'].iloc[-1]
future_dates = [last_date + timedelta(days=i+1) for i in range(PRED_DAYS)]

fig, ax = plt.subplots(figsize=(14, 7))

# Plot historical data
days = np.arange(1, TIME_STEP + 1)
ax.plot(days, last_60_days, label=f'Last {TIME_STEP} days of observed prices',
        marker='o', linewidth=2.5, markersize=5, color='blue')

# Plot forecast data
pred_days = np.arange(TIME_STEP + 1, TIME_STEP + PRED_DAYS + 1)
ax.plot(pred_days, future_predictions, label=f'Next {PRED_DAYS} day forecast prices',
        marker='s', linewidth=3, markersize=8, color='red', linestyle='--')

# Annotate predicted values
for i, (day, price) in enumerate(zip(pred_days, future_predictions)):
    ax.annotate(f'${price:,.0f}',
                xy=(day, price),
                xytext=(5, 10),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                color='red')

# Mark the forecast origin
ax.axvline(x=TIME_STEP, color='green', linestyle=':', linewidth=2, label='Forecast origin')

ax.set_title(f'Bitcoin price: Last {TIME_STEP} days vs Next {PRED_DAYS} day forecast (illustration)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Days', fontsize=13)
ax.set_ylabel('Price (USD)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('future_prediction_concept.png', dpi=300, bbox_inches='tight')
print("Saved: future_prediction_concept.png")
plt.close()

# ============================================================================
# 4. Price distributions and summary statistics
# ============================================================================
print("\nGenerate plot 4: Price distributions and summary statistics...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 4.1 Price histogram
axes[0, 0].hist(df['Close'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(df['Close'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${df["Close"].mean():,.0f}')
axes[0, 0].axvline(df['Close'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${df["Close"].median():,.0f}')
axes[0, 0].set_title('Price distribution histogram', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Price (USD)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 4.2 Percentage price change
df_temp = df.copy()
df_temp['Daily_Change'] = df_temp['Close'].pct_change() * 100
axes[0, 1].hist(df_temp['Daily_Change'].dropna(), bins=100, color='lightcoral', edgecolor='black', alpha=0.7)
axes[0, 1].axvline(0, color='black', linestyle='-', linewidth=2)
axes[0, 1].set_title('Daily return distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Return (%)', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim(-15, 15)

# 4.3 Annual mean price
df_temp['Year'] = df_temp['Date'].dt.year
yearly_avg = df_temp.groupby('Year')['Close'].mean()
axes[1, 0].bar(yearly_avg.index, yearly_avg.values, color='steelblue', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Annual mean price', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Year', fontsize=12)
axes[1, 0].set_ylabel('Mean price (USD)', fontsize=12)
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, (year, price) in enumerate(yearly_avg.items()):
    axes[1, 0].text(year, price, f'${price:,.0f}', ha='center', va='bottom', fontsize=9)

# 4.4 Volatility (30-day rolling standard deviation)
df_temp['Volatility'] = df_temp['Close'].rolling(window=30).std()
axes[1, 1].plot(df_temp.index, df_temp['Volatility'], linewidth=1.5, color='purple', alpha=0.7)
axes[1, 1].fill_between(df_temp.index, df_temp['Volatility'], alpha=0.3, color='purple')
axes[1, 1].set_title('Price volatility (30-day rolling standard deviation)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Time', fontsize=12)
axes[1, 1].set_ylabel('Standard deviation', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bitcoin_statistics.png', dpi=300, bbox_inches='tight')
print("Saved: bitcoin_statistics.png")
plt.close()

# ============================================================================
# 5. many-to-one vs multi-step Conceptual comparison
# ============================================================================
print("\nGenerate plot 5: many-to-one vs Multi-step model comparison...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 5.1 Many-to-one forecasting (conceptual illustration)
ax = axes[0]
input_days = np.arange(1, 61)
input_prices = last_60_days

ax.plot(input_days, input_prices, 'o-', linewidth=2, markersize=4, color='blue', label='Input: 60 days of historical data')

# First prediction
ax.plot([61], [input_prices[-1] * 1.01], 'rs', markersize=12, label='Forecast: Day 1')
ax.arrow(60, input_prices[-1], 1, input_prices[-1] * 0.01, head_width=1.5, head_length=500, fc='red', ec='red', alpha=0.5)

# Second prediction (using the first prediction)
ax.plot([62], [input_prices[-1] * 1.02], 'rs', markersize=12, label='Forecast: Day 2')
ax.arrow(61, input_prices[-1] * 1.01, 1, input_prices[-1] * 0.01, head_width=1.5, head_length=500, fc='orange', ec='orange', alpha=0.5)

# Third prediction
ax.plot([63], [input_prices[-1] * 1.03], 'rs', markersize=12, label='Forecast: Day 3')
ax.arrow(62, input_prices[-1] * 1.02, 1, input_prices[-1] * 0.01, head_width=1.5, head_length=500, fc='yellow', ec='orange', alpha=0.5)

ax.axvline(x=60, color='green', linestyle='--', linewidth=2, label='Forecast origin')
ax.set_title('Many-to-one forecasting (Many-to-One)\nPredict one day at a time, repeated three times\nErrors can propagate recursively', fontsize=13, fontweight='bold')
ax.set_xlabel('Days', fontsize=11)
ax.set_ylabel('Price (USD)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 5.2 Multi-step forecasting (direct-output illustration)
ax = axes[1]
ax.plot(input_days, input_prices, 'o-', linewidth=2, markersize=4, color='blue', label='Input: 60 days of historical data')

# Predict 3 days in one pass
pred_days_multi = [61, 62, 63]
pred_prices_multi = [input_prices[-1] * 1.01, input_prices[-1] * 1.02, input_prices[-1] * 1.03]
ax.plot(pred_days_multi, pred_prices_multi, 's-', linewidth=3, markersize=12, color='red', label='output in one pass: 3-day forecast')

# A large arrow indicates output in one pass
ax.annotate('', xy=(62, input_prices[-1] * 1.02), xytext=(60, input_prices[-1]),
            arrowprops=dict(arrowstyle='->', lw=3, color='red', alpha=0.6))
ax.text(61, input_prices[-1] * 0.95, 'in one pass\nForecast 3 days', fontsize=11, ha='center', 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax.axvline(x=60, color='green', linestyle='--', linewidth=2, label='Forecast origin')
ax.set_title('Multi-step forecasting (Many-to-Many)\nPredict 3 days in one pass\nno recursive forecast feedback', fontsize=13, fontweight='bold')
ax.set_xlabel('Days', fontsize=11)
ax.set_ylabel('Price (USD)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('many_to_one_vs_many_to_many.png', dpi=300, bbox_inches='tight')
print("Saved: many_to_one_vs_many_to_many.png")
plt.close()

# ============================================================================
# Print summary statistics
# ============================================================================
print("\n" + "="*70)
print("Data statistics")
print("="*70)
print(f"Total days: {len(df)}")
print(f"Date range: {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}")
print(f"Maximum price: ${df['Close'].max():,.2f}")
print(f"Minimum price: ${df['Close'].min():,.2f}")
print(f"Mean price: ${df['Close'].mean():,.2f}")
print(f"Median: ${df['Close'].median():,.2f}")
print(f"Latest price: ${df['Close'].iloc[-1]:,.2f}")
print(f"Standard deviation: ${df['Close'].std():,.2f}")

print("\n" + "="*70)
print("All plots generated!")
print("="*70)
print("\nGenerated plots:")
print("  1. bitcoin_full_history.png        - Full historical price series")
print("  2. bitcoin_recent_year.png         - Price series over the last year")
print("  3. future_prediction_concept.png   - Conceptual forecast illustration")
print("  4. bitcoin_statistics.png          - Price statistics plot")
print("  5. many_to_one_vs_many_to_many.png - Comparison of forecasting approaches")
print("\n" + "="*70)
