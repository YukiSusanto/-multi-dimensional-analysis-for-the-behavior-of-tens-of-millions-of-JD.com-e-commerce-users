"""
第四阶段：A/B Test 实验设计
==========================
- 识别实验目标群体（加购未购买用户）
- 功效分析：最小样本量计算
- 分流方案与随机化策略
- 评估指标体系
- 使用 scipy.stats 进行统计推断
"""
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import os
import json

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(DATA_DIR, 'data', 'processed', 'user_behavior_clean.csv')
RFM_PATH = os.path.join(DATA_DIR, 'data', 'processed', 'rfm_clusters.csv')

print("=" * 60)
print("第四阶段：A/B Test 实验设计")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ============================================================
# Step 1: 识别实验目标群体
# ============================================================
print("\n[1/6] 识别目标群体...")
df = pd.read_csv(CSV_PATH, parse_dates=['event_date'])

# 定义"加购未购买"用户：
# 1. 有过 cart 行为
# 2. 从未有过 buy 行为（或近7天有 cart 但无 buy）
cart_users = set(df[df['behavior'] == 'cart']['user_id'].unique())
buy_users  = set(df[df['behavior'] == 'buy']['user_id'].unique())

# 群体 A: 有加购但从未购买
cart_no_buy_ever = cart_users - buy_users

# 群体 B: 近7天有加购但无购买（更精准的目标）
last_date = df['event_date'].max()
cutoff = last_date - timedelta(days=7)
recent_cart = df[(df['behavior'] == 'cart') & (df['event_date'] >= cutoff)]['user_id'].unique()
recent_buy  = df[(df['behavior'] == 'buy')  & (df['event_date'] >= cutoff)]['user_id'].unique()
cart_no_buy_recent = set(recent_cart) - set(recent_buy)

# 群体 C: 从聚类中筛选"潜在转化"+"流失风险"中有 cart 行为的
rfm = pd.read_csv(RFM_PATH)
target_clusters = rfm[rfm['cluster'].isin([2, 3])]  # 潜在转化 + 流失风险
eligible_ids = target_clusters['user_id'].isin(cart_users)
target_from_cluster = target_clusters[eligible_ids]['user_id'].tolist()

target_population = list(cart_no_buy_recent & set(target_from_cluster))

print(f"  有加购从未购买 (ever):      {len(cart_no_buy_ever):,}")
print(f"  近7天加购但无购买 (recent):  {len(cart_no_buy_recent):,}")
print(f"  交叉筛选后目标群体:          {len(target_population):,}")

# 获取目标群体的 baseline 行为统计
target_df = df[df['user_id'].isin(target_population)]
target_cart_count = len(target_df[target_df['behavior'] == 'cart'])
target_pv_count   = len(target_df[target_df['behavior'] == 'pv'])
print(f"  目标群体 cart 行为数: {target_cart_count:,}")
print(f"  目标群体 pv 行为数:   {target_pv_count:,}")

# ============================================================
# Step 2: 功效分析 — 计算最小样本量
# ============================================================
print("\n[2/6] 功效分析 (Power Analysis)...")

# 参数设定
alpha = 0.05          # 显著性水平
power_desired = 0.80  # 统计功效
mde_absolute = 0.05   # 绝对提升 5 个百分点

# baseline: 当前加购→购买的估计转化率
# 使用全体有 cart 行为用户的购买转化率作为 baseline
all_cart_users = df[df['behavior'] == 'cart']['user_id'].unique()
all_cart_buyers = set(all_cart_users) & buy_users
baseline_rate = len(all_cart_buyers) / len(all_cart_users) if len(all_cart_users) > 0 else 0
print(f"  baseline 转化率 (cart→buy): {baseline_rate*100:.1f}%")

# 样本量计算公式 (two-proportion z-test, two-sided)
z_alpha = stats.norm.ppf(1 - alpha / 2)  # 1.96
z_beta  = stats.norm.ppf(power_desired)   # 0.842

p1 = baseline_rate
p2 = baseline_rate + mde_absolute

# n per group
numerator = (z_alpha * np.sqrt(2 * p1 * (1 - p1)) + z_beta * np.sqrt(p1*(1-p1) + p2*(1-p2)))**2
denominator = (p2 - p1)**2
n_per_group = int(np.ceil(numerator / denominator))

print(f"\n  参数设定:")
print(f"    α (显著性水平):      {alpha}")
print(f"    1-β (统计功效):      {power_desired}")
print(f"    MDE (最小可检测效应): {mde_absolute*100:.0f}% (绝对)")
print(f"    baseline p1:         {p1*100:.1f}%")
print(f"    expected p2:         {p2*100:.1f}%")
print(f"\n  计算结果:")
print(f"    每组最小样本量: {n_per_group:,}")
print(f"    总样本量需求:   {n_per_group * 2:,}")

# ============================================================
# Step 3: 不同 MDE 下的灵敏度分析
# ============================================================
print("\n[3/6] 灵敏度分析 (不同 MDE/样本量)...")
print(f"\n  {'MDE':>8} | {'每组样本量':>12} | {'总样本量':>12} | {'评估'}")
print("  " + "-" * 58)

mdes = [0.01, 0.03, 0.05, 0.07, 0.10]
for mde in mdes:
    p2_mde = p1 + mde
    num = (z_alpha * np.sqrt(2 * p1 * (1 - p1)) + z_beta * np.sqrt(p1*(1-p1) + p2_mde*(1-p2_mde)))**2
    den = (p2_mde - p1)**2
    n = int(np.ceil(num / den))
    feasible = "✓ 可行" if n <= len(target_population) else "✗ 样本不足"
    print(f"  {mde*100:>7.1f}% | {n:>12,} | {n*2:>12,} | {feasible}")

# ============================================================
# Step 4: 实验分流方案
# ============================================================
print("\n[4/6] 分流方案设计...")

# 分层随机化：按聚类标签分层，确保 experiment/control 在各分层内均衡
np.random.seed(42)
target_rfm = rfm[rfm['user_id'].isin(target_population)].copy()

# 如果目标群体大于所需样本量2倍，随机抽样
sample_size = min(len(target_population), n_per_group * 2)
sampled_users = np.random.choice(target_population, size=sample_size, replace=False)

# 分层随机分配
sampled_rfm = target_rfm[target_rfm['user_id'].isin(sampled_users)]
sampled_rfm['group'] = 'control'
for cluster_id in sampled_rfm['cluster'].unique():
    cluster_mask = sampled_rfm['cluster'] == cluster_id
    cluster_users = sampled_rfm.loc[cluster_mask, 'user_id'].values.copy()
    np.random.shuffle(cluster_users)
    split_idx = len(cluster_users) // 2
    experiment_users = cluster_users[:split_idx]
    sampled_rfm.loc[sampled_rfm['user_id'].isin(experiment_users), 'group'] = 'experiment'

exp_count = (sampled_rfm['group'] == 'experiment').sum()
ctrl_count = (sampled_rfm['group'] == 'control').sum()

print(f"  总样本: {sample_size:,}")
print(f"  实验组: {exp_count:,} (接收优惠券推送)")
print(f"  对照组: {ctrl_count:,} (不推送)")
print(f"  分层策略: 按 K-Means 聚类标签分层随机化")
print(f"  分流比例: {exp_count/sample_size*100:.1f}% / {ctrl_count/sample_size*100:.1f}%")

# ============================================================
# Step 5: 评估指标体系
# ============================================================
print("\n[5/6] 评估指标体系...")

print("""
  ┌──────────────────────────────────────────────────────────────────┐
  │  指标层级           │ 指标名称              │ 统计方法             │
  ├──────────────────────────────────────────────────────────────────┤
  │  一级（核心）        │ cart→buy 转化率       │ Two-proportion z-test│
  │  一级（核心）        │ 人均购买频次           │ Welch's t-test       │
  │  二级（辅助）        │ 客单价变化             │ Welch's t-test       │
  │  二级（辅助）        │ 次日留存率             │ Chi-square test      │
  │  三级（观测）        │ 优惠券核销率           │ 描述性统计           │
  │  三级（观测）        │ 品类偏好偏移           │ 描述性统计           │
  └──────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# Step 6: 模拟实验结果
# ============================================================
print("[6/6] 实验模拟 (Monte Carlo)...")

np.random.seed(42)
n_sim = exp_count

# 模拟：对照组按 baseline 转化，实验组提升 MDE
ctrl_conversions = np.random.binomial(1, baseline_rate, n_sim)
exp_conversions  = np.random.binomial(1, baseline_rate + mde_absolute, n_sim)

ctrl_rate = ctrl_conversions.mean()
exp_rate  = exp_conversions.mean()

# Two-proportion z-test
n_ctrl = len(ctrl_conversions)
n_exp  = len(exp_conversions)
p_pool = (ctrl_conversions.sum() + exp_conversions.sum()) / (n_ctrl + n_exp)
se_pool = np.sqrt(p_pool * (1 - p_pool) * (1/n_ctrl + 1/n_exp))
z_stat = (exp_rate - ctrl_rate) / se_pool
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

# 置信区间
se_diff = np.sqrt(exp_rate*(1-exp_rate)/n_exp + ctrl_rate*(1-ctrl_rate)/n_ctrl)
ci_lower = (exp_rate - ctrl_rate) - z_alpha * se_diff
ci_upper = (exp_rate - ctrl_rate) + z_alpha * se_diff

print(f"\n  模拟实验结果:")
print(f"    对照组转化率:  {ctrl_rate*100:.2f}%")
print(f"    实验组转化率:  {exp_rate*100:.2f}%")
print(f"    提升幅度:      {(exp_rate-ctrl_rate)*100:.2f} 个百分点")
print(f"    相对提升:      {((exp_rate-ctrl_rate)/ctrl_rate)*100:.1f}%")
print(f"    z-statistic:   {z_stat:.3f}")
print(f"    p-value:       {p_value:.4f}")
print(f"    95% CI:        [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")

if p_value < alpha:
    print(f"    结论: 拒绝 H0，实验组显著优于对照组 ✅ (p < {alpha})")
else:
    print(f"    结论: 未拒绝 H0，差异不显著 (p >= {alpha})")

# ============================================================
# 保存实验设计方案
# ============================================================
design = {
    'experiment_name': '加购挽单优惠券 A/B Test',
    'target_population': len(target_population),
    'sample_size_per_group': n_per_group,
    'alpha': alpha,
    'power': power_desired,
    'mde_absolute': mde_absolute,
    'baseline_conversion_rate': round(baseline_rate, 4),
    'stratification': 'K-Means cluster labels',
    'metrics': {
        'primary': 'cart-to-buy conversion rate',
        'secondary': ['avg purchase frequency', 'avg order value', 'day1 retention'],
    },
    'duration_days': 14,
    'push_time': '20:00 (下单高峰前1小时)',
}

out_path = os.path.join(DATA_DIR, 'data', 'processed', 'ab_test_design.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(design, f, indent=2, ensure_ascii=False)

print(f"\n  实验设计方案已保存: {out_path}")
print(f"\n第四阶段完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
