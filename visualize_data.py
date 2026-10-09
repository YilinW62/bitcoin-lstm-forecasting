"""
比特币数据可视化脚本
展示历史数据和预测概念
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
print("加载数据...")
df = pd.read_excel('Bitcoin_Historical_Data_10年.xlsx')

# 数据预处理
if 'Price' in df.columns:
    df = df.rename(columns={'Price': 'Close'})

df['Date'] = pd.to_datetime(df['Date'])

if df['Close'].dtype == 'object':
    df['Close'] = df['Close'].str.replace(',', '').astype(float)

# 确保从旧到新排序
if df['Date'].iloc[0] > df['Date'].iloc[-1]:
    df = df.sort_values('Date', ascending=True).reset_index(drop=True)

df = df.dropna(subset=['Close'])
df = df[df['Close'] > 0].reset_index(drop=True)

print(f"数据范围: {df['Date'].iloc[0].date()} 到 {df['Date'].iloc[-1].date()}")
print(f"数据点数: {len(df)}")
print(f"当前价格: ${df['Close'].iloc[-1]:,.2f}")

# ============================================================================
# 1. 完整历史价格走势图
# ============================================================================
print("\n生成图1: 完整历史价格走势图...")

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(df.index, df['Close'], linewidth=1.5, color='blue', alpha=0.8)
ax.fill_between(df.index, df['Close'], alpha=0.3, color='lightblue')

# 添加关键价格线
max_price = df['Close'].max()
min_price = df['Close'].min()
current_price = df['Close'].iloc[-1]

ax.axhline(y=max_price, color='red', linestyle='--', linewidth=1, alpha=0.5, label=f'历史最高: ${max_price:,.0f}')
ax.axhline(y=current_price, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'当前价格: ${current_price:,.0f}')

ax.set_title('比特币完整历史价格走势 (2015-2025)', fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('时间', fontsize=14)
ax.set_ylabel('价格 (USD)', fontsize=14)
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)

# 设置x轴日期标签
n_ticks = 10
tick_indices = np.linspace(0, len(df)-1, n_ticks, dtype=int)
ax.set_xticks(tick_indices)
ax.set_xticklabels([df['Date'].iloc[i].strftime('%Y-%m') for i in tick_indices], rotation=45)

plt.tight_layout()
plt.savefig('bitcoin_full_history.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: bitcoin_full_history.png")
plt.close()

# ============================================================================
# 2. 最近一年的价格走势
# ============================================================================
print("\n生成图2: 最近一年价格走势...")

df_recent = df.tail(365).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df_recent.index, df_recent['Close'], linewidth=2, color='blue', marker='o', 
        markersize=2, alpha=0.8, label='实际价格')

# 添加移动平均线
ma_7 = df_recent['Close'].rolling(window=7).mean()
ma_30 = df_recent['Close'].rolling(window=30).mean()

ax.plot(df_recent.index, ma_7, linewidth=1.5, color='orange', alpha=0.7, label='7日均线')
ax.plot(df_recent.index, ma_30, linewidth=1.5, color='red', alpha=0.7, label='30日均线')

ax.set_title('比特币最近一年价格走势', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('天数', fontsize=12)
ax.set_ylabel('价格 (USD)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bitcoin_recent_year.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: bitcoin_recent_year.png")
plt.close()

# ============================================================================
# 3. 预测概念图（模拟）
# ============================================================================
print("\n生成图3: 预测概念图...")

# 使用最后60天的真实数据
TIME_STEP = 60
PRED_DAYS = 3

last_60_days = df['Close'].tail(TIME_STEP).values
last_60_dates = df['Date'].tail(TIME_STEP).values

# 简单预测：使用最后一天价格的小幅波动
current_price = last_60_days[-1]
# 模拟预测（这里用简单的线性外推 + 随机波动）
trend = (last_60_days[-1] - last_60_days[-10]) / 10
future_predictions = []
for i in range(PRED_DAYS):
    noise = np.random.normal(0, current_price * 0.01)  # 1%的波动
    pred = current_price + trend * (i + 1) + noise
    future_predictions.append(pred)

# 生成未来日期
last_date = df['Date'].iloc[-1]
future_dates = [last_date + timedelta(days=i+1) for i in range(PRED_DAYS)]

fig, ax = plt.subplots(figsize=(14, 7))

# 绘制历史数据
days = np.arange(1, TIME_STEP + 1)
ax.plot(days, last_60_days, label=f'最近 {TIME_STEP} 天实际价格',
        marker='o', linewidth=2.5, markersize=5, color='blue')

# 绘制预测数据
pred_days = np.arange(TIME_STEP + 1, TIME_STEP + PRED_DAYS + 1)
ax.plot(pred_days, future_predictions, label=f'未来 {PRED_DAYS} 天预测价格',
        marker='s', linewidth=3, markersize=8, color='red', linestyle='--')

# 标注预测值
for i, (day, price) in enumerate(zip(pred_days, future_predictions)):
    ax.annotate(f'${price:,.0f}',
                xy=(day, price),
                xytext=(5, 10),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                color='red')

# 添加预测起点线
ax.axvline(x=TIME_STEP, color='green', linestyle=':', linewidth=2, label='预测起点')

ax.set_title(f'比特币价格：最近 {TIME_STEP} 天 vs 未来 {PRED_DAYS} 天预测（示例）',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('天数', fontsize=13)
ax.set_ylabel('价格 (USD)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('future_prediction_concept.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: future_prediction_concept.png")
plt.close()

# ============================================================================
# 4. 价格分布和统计图
# ============================================================================
print("\n生成图4: 价格分布和统计图...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 4.1 价格直方图
axes[0, 0].hist(df['Close'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(df['Close'].mean(), color='red', linestyle='--', linewidth=2, label=f'平均: ${df["Close"].mean():,.0f}')
axes[0, 0].axvline(df['Close'].median(), color='green', linestyle='--', linewidth=2, label=f'中位数: ${df["Close"].median():,.0f}')
axes[0, 0].set_title('价格分布直方图', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('价格 (USD)', fontsize=12)
axes[0, 0].set_ylabel('频数', fontsize=12)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 4.2 价格变化百分比
df_temp = df.copy()
df_temp['Daily_Change'] = df_temp['Close'].pct_change() * 100
axes[0, 1].hist(df_temp['Daily_Change'].dropna(), bins=100, color='lightcoral', edgecolor='black', alpha=0.7)
axes[0, 1].axvline(0, color='black', linestyle='-', linewidth=2)
axes[0, 1].set_title('日涨跌幅分布', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('涨跌幅 (%)', fontsize=12)
axes[0, 1].set_ylabel('频数', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim(-15, 15)

# 4.3 年度平均价格
df_temp['Year'] = df_temp['Date'].dt.year
yearly_avg = df_temp.groupby('Year')['Close'].mean()
axes[1, 0].bar(yearly_avg.index, yearly_avg.values, color='steelblue', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('年度平均价格', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('年份', fontsize=12)
axes[1, 0].set_ylabel('平均价格 (USD)', fontsize=12)
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, (year, price) in enumerate(yearly_avg.items()):
    axes[1, 0].text(year, price, f'${price:,.0f}', ha='center', va='bottom', fontsize=9)

# 4.4 波动率（30日滚动标准差）
df_temp['Volatility'] = df_temp['Close'].rolling(window=30).std()
axes[1, 1].plot(df_temp.index, df_temp['Volatility'], linewidth=1.5, color='purple', alpha=0.7)
axes[1, 1].fill_between(df_temp.index, df_temp['Volatility'], alpha=0.3, color='purple')
axes[1, 1].set_title('价格波动率 (30日滚动标准差)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('时间', fontsize=12)
axes[1, 1].set_ylabel('标准差', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bitcoin_statistics.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: bitcoin_statistics.png")
plt.close()

# ============================================================================
# 5. 多对一 vs 多对多 概念对比图
# ============================================================================
print("\n生成图5: 多对一 vs 多对多模型对比图...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 5.1 多对一模式（当前train.py）
ax = axes[0]
input_days = np.arange(1, 61)
input_prices = last_60_days

ax.plot(input_days, input_prices, 'o-', linewidth=2, markersize=4, color='blue', label='输入: 60天历史数据')

# 第一次预测
ax.plot([61], [input_prices[-1] * 1.01], 'rs', markersize=12, label='预测: 第1天')
ax.arrow(60, input_prices[-1], 1, input_prices[-1] * 0.01, head_width=1.5, head_length=500, fc='red', ec='red', alpha=0.5)

# 第二次预测（使用第一次的预测值）
ax.plot([62], [input_prices[-1] * 1.02], 'rs', markersize=12, label='预测: 第2天')
ax.arrow(61, input_prices[-1] * 1.01, 1, input_prices[-1] * 0.01, head_width=1.5, head_length=500, fc='orange', ec='orange', alpha=0.5)

# 第三次预测
ax.plot([63], [input_prices[-1] * 1.03], 'rs', markersize=12, label='预测: 第3天')
ax.arrow(62, input_prices[-1] * 1.02, 1, input_prices[-1] * 0.01, head_width=1.5, head_length=500, fc='yellow', ec='orange', alpha=0.5)

ax.axvline(x=60, color='green', linestyle='--', linewidth=2, label='预测起点')
ax.set_title('多对一模式 (Many-to-One)\n每次预测1天，循环3次\n❌ 误差会累积', fontsize=13, fontweight='bold')
ax.set_xlabel('天数', fontsize=11)
ax.set_ylabel('价格 (USD)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 5.2 多对多模式（train_seq2seq.py）
ax = axes[1]
ax.plot(input_days, input_prices, 'o-', linewidth=2, markersize=4, color='blue', label='输入: 60天历史数据')

# 一次性预测3天
pred_days_multi = [61, 62, 63]
pred_prices_multi = [input_prices[-1] * 1.01, input_prices[-1] * 1.02, input_prices[-1] * 1.03]
ax.plot(pred_days_multi, pred_prices_multi, 's-', linewidth=3, markersize=12, color='red', label='一次性输出: 3天预测')

# 大箭头表示一次性输出
ax.annotate('', xy=(62, input_prices[-1] * 1.02), xytext=(60, input_prices[-1]),
            arrowprops=dict(arrowstyle='->', lw=3, color='red', alpha=0.6))
ax.text(61, input_prices[-1] * 0.95, '一次性\n预测3天', fontsize=11, ha='center', 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax.axvline(x=60, color='green', linestyle='--', linewidth=2, label='预测起点')
ax.set_title('多对多模式 (Many-to-Many)\n一次性预测3天\n✅ 无误差累积', fontsize=13, fontweight='bold')
ax.set_xlabel('天数', fontsize=11)
ax.set_ylabel('价格 (USD)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('many_to_one_vs_many_to_many.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: many_to_one_vs_many_to_many.png")
plt.close()

# ============================================================================
# 打印统计信息
# ============================================================================
print("\n" + "="*70)
print("📊 数据统计")
print("="*70)
print(f"总天数: {len(df)}")
print(f"日期范围: {df['Date'].iloc[0].date()} 到 {df['Date'].iloc[-1].date()}")
print(f"最高价: ${df['Close'].max():,.2f}")
print(f"最低价: ${df['Close'].min():,.2f}")
print(f"平均价: ${df['Close'].mean():,.2f}")
print(f"中位数: ${df['Close'].median():,.2f}")
print(f"当前价: ${df['Close'].iloc[-1]:,.2f}")
print(f"标准差: ${df['Close'].std():,.2f}")

print("\n" + "="*70)
print("✅ 所有图片已生成！")
print("="*70)
print("\n生成的图片:")
print("  1. bitcoin_full_history.png        - 完整历史价格走势")
print("  2. bitcoin_recent_year.png         - 最近一年价格走势")
print("  3. future_prediction_concept.png   - 预测概念示意图")
print("  4. bitcoin_statistics.png          - 价格统计分析图")
print("  5. many_to_one_vs_many_to_many.png - 两种预测模式对比")
print("\n" + "="*70)
