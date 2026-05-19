"""
第三阶段：RFM 特征工程 + K-Means 用户聚类
========================================
- R (Recency): 距基准日天数
- F (Frequency): 累计购买次数
- M (Monetary): 估算消费总额（原始数据无金额，按品类分配模拟价格）
- K-Means 聚类 + 肘部法 + 轮廓系数
- 可视化输出
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from datetime import datetime, date
import os
import warnings
warnings.filterwarnings('ignore')

# 中文配置 — 直接注册 SimHei（避免 matplotlib 缓存问题）
import matplotlib.font_manager as fm
fm.fontManager.addfont(os.path.expanduser('~/.fonts/SimHei.ttf'))
fm._load_fontmanager(try_read_cache=False)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(DATA_DIR, 'data', 'processed', 'user_behavior_clean.csv')
IMG_DIR  = os.path.join(DATA_DIR, 'images')
os.makedirs(IMG_DIR, exist_ok=True)

REFERENCE_DATE = date(2014, 12, 19)  # 数据集最后一天的后一天

print("=" * 60)
print("第三阶段：RFM 用户价值建模 + K-Means 聚类")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ============================================================
# Step 1: 加载数据 & RFM 特征工程
# ============================================================
print("\n[1/5] RFM 特征工程...")
df = pd.read_csv(CSV_PATH, parse_dates=['event_date'])

# --- R: Recency (距基准日天数) ---
recency = df.groupby('user_id')['event_date'].max().reset_index()
recency['recency'] = recency['event_date'].apply(lambda x: (REFERENCE_DATE - x.date()).days)
recency = recency[['user_id', 'recency']]

# --- F: Frequency (购买次数) ---
buy_df = df[df['behavior'] == 'buy']
frequency = buy_df.groupby('user_id').size().reset_index(name='frequency')

# --- M: Monetary (模拟消费金额) ---
# 数据无金额字段，按品类随机分配模拟价格（10~500元，正态分布）
np.random.seed(42)
categories = df['item_category'].unique()
cat_prices = {c: max(10, np.random.normal(100, 80)) for c in categories}
cat_prices_series = pd.Series(cat_prices, name='avg_price')

buy_with_price = buy_df[['user_id', 'item_category']].drop_duplicates()
buy_with_price['price'] = buy_with_price['item_category'].map(cat_prices)

# 按用户实际购买品类匹配价格（简化：按品类去重后求和）
# 更准确的做法是每次购买都算一遍，但此处品类级去重避免重复计数
monetary = buy_with_price.groupby('user_id')['price'].sum().reset_index(name='monetary')

# --- 合并 RFM ---
rfm = pd.DataFrame({'user_id': df['user_id'].unique()})
rfm = rfm.merge(recency, on='user_id', how='left')
rfm = rfm.merge(frequency, on='user_id', how='left')
rfm = rfm.merge(monetary, on='user_id', how='left')

# 填充：无购买行为的用户 F=0, M=0
rfm['frequency'] = rfm['frequency'].fillna(0).astype(int)
rfm['monetary']  = rfm['monetary'].fillna(0).round(2)
rfm['recency']   = rfm['recency'].fillna(rfm['recency'].max()).astype(int)

# 补充行为统计
action_counts = df.groupby('user_id').size().reset_index(name='total_actions')
rfm = rfm.merge(action_counts, on='user_id', how='left')

print(f"  用户总数: {len(rfm):,}")
print(f"  R (Recency):  均值 {rfm['recency'].mean():.1f} 天, 范围 [{rfm['recency'].min()}, {rfm['recency'].max()}]")
print(f"  F (Frequency): 均值 {rfm['frequency'].mean():.1f} 次, 范围 [{rfm['frequency'].min()}, {rfm['frequency'].max()}]")
print(f"  M (Monetary):  均值 ¥{rfm['monetary'].mean():.0f}, 范围 [¥{rfm['monetary'].min():.0f}, ¥{rfm['monetary'].max():.0f}]")
print(f"  ⚠️ M 为基于品类的模拟价格（原始数据无金额字段）")

# ============================================================
# Step 2: 数据标准化
# ============================================================
print("\n[2/5] StandardScaler 标准化...")
features = ['recency', 'frequency', 'monetary']
X = rfm[features].copy()

# RFM 核心逻辑：R 越小越好，F/M 越大越好
# 对 R 取反使方向一致
X['recency_inv'] = 1 / (X['recency'] + 1)  # 标准化到 (0, 1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['recency_inv', 'frequency', 'monetary']])
print(f"  标准化完成: {X_scaled.shape}")

# ============================================================
# Step 3: 肘部法 + 轮廓系数确定最优 K
# ============================================================
print("\n[3/5] 肘部法 + 轮廓系数...")
K_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    print(f"  K={k:>2}: Inertia={km.inertia_:>10.1f}, Silhouette={silhouette_scores[-1]:.4f}")

# 找出最佳 K（轮廓系数最高的，但排除 K=2 过于简单的情况）
best_k_sil = K_range[np.argmax(silhouette_scores[1:]) + 1]  # 从 K=3 开始
best_k = best_k_sil if silhouette_scores[best_k_sil - 2] > 0.25 else 5  # fallback
print(f"\n  推荐 K = {best_k} (轮廓系数: {silhouette_scores[best_k-2]:.4f})")

# ============================================================
# Step 4: K-Means 聚类
# ============================================================
print(f"\n[4/5] K-Means 聚类 (K={best_k})...")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
rfm['cluster'] = kmeans.fit_predict(X_scaled)

# 聚类中心（反标准化查看实际含义）
centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=['recency_inv', 'frequency', 'monetary']
)
# 还原 recency
centers['recency'] = (1 / centers['recency_inv']) - 1

# 计算每个聚类的 RFM 均值
cluster_profile = rfm.groupby('cluster').agg(
    user_count=('user_id', 'count'),
    avg_recency=('recency', 'mean'),
    avg_frequency=('frequency', 'mean'),
    avg_monetary=('monetary', 'mean'),
    avg_actions=('total_actions', 'mean')
).round(2)

# 智能标签
def label_cluster(row):
    r, f, m = row['avg_recency'], row['avg_frequency'], row['avg_monetary']
    r_high = r > rfm['recency'].median()
    f_high = f > rfm['frequency'].median()
    m_high = m > rfm['monetary'].median()
    
    if not r_high and f_high and m_high:
        return '高价值核心用户'
    elif not r_high and f_high:
        return '重点保持用户'
    elif not r_high and not f_high and m_high:
        return '重点发展用户'
    elif r_high and not f_high:
        return '流失用户'
    elif not r_high and not f_high:
        return '潜在转化用户'
    else:
        return '一般用户'

cluster_profile['label'] = cluster_profile.apply(label_cluster, axis=1)

print("\n  聚类画像:")
print(f"  {'标签':<16} | {'人数':>8} | {'R(天)':>7} | {'F(次)':>7} | {'M(¥)':>9} | {'行为量':>8}")
print("  " + "-" * 72)
for idx, row in cluster_profile.iterrows():
    print(f"  {row['label']:<16} | {int(row['user_count']):>8,} | "
          f"{row['avg_recency']:>7.1f} | {row['avg_frequency']:>7.1f} | "
          f"¥{row['avg_monetary']:>8.0f} | {row['avg_actions']:>8.0f}")

# ============================================================
# Step 5: 可视化
# ============================================================
print("\n[5/5] 生成可视化图表...")

colors = plt.cm.tab10(np.linspace(0, 1, best_k))

# --- 5a: 肘部法 + 轮廓系数 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(K_range, inertias, 'bo-', markersize=6)
ax1.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
ax1.set_xlabel('聚类数量 (K)')
ax1.set_ylabel('簇内平方和 (SSE)')
ax1.set_title('肘部法')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(K_range, silhouette_scores, 'go-', markersize=6)
ax2.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'最佳 K={best_k}')
ax2.set_xlabel('聚类数量 (K)')
ax2.set_ylabel('轮廓系数')
ax2.set_title('轮廓系数分析')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'elbow_silhouette.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  已保存: images/elbow_silhouette.png")

# --- 5b: 3D 散点图 ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 降采样以便可视化
sample_rfm = rfm.sample(min(3000, len(rfm)), random_state=42)
sample_scaled = scaler.transform(X.loc[sample_rfm.index, ['recency_inv', 'frequency', 'monetary']])
sample_labels = kmeans.predict(sample_scaled)

for ci in range(best_k):
    mask = sample_labels == ci
    label = cluster_profile.loc[ci, 'label']
    ax.scatter(
        sample_rfm.loc[mask, 'recency'].values,
        sample_rfm.loc[mask, 'frequency'].values,
        sample_rfm.loc[mask, 'monetary'].values,
        c=[colors[ci]], label=f'{label} ({int(cluster_profile.loc[ci, "user_count"]):,})',
        s=15, alpha=0.7
    )

ax.set_xlabel('Recency (距上次活跃天数)')
ax.set_ylabel('Frequency (购买频次)')
ax.set_zlabel('Monetary (模拟消费额 ¥)')
ax.set_title(f'RFM 用户分群 (K={best_k})')
ax.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'rfm_3d_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  已保存: images/rfm_3d_scatter.png")

# --- 5c: 聚类分布饼图 ---
fig, ax = plt.subplots(figsize=(8, 8))
sizes = cluster_profile['user_count'].values
labels_pie = [f"{cluster_profile.loc[i, 'label']}\n({int(s):,})" for i, s in enumerate(sizes)]
wedges, texts = ax.pie(sizes, labels=None, colors=colors, startangle=90,
                        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax.legend(wedges, labels_pie, title='用户分群', loc='center left',
          bbox_to_anchor=(1, 0.5), fontsize=10)
ax.set_title(f'用户聚类分布 (K={best_k})')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'cluster_pie.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  已保存: images/cluster_pie.png")

# --- 5d: 聚类雷达图 ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
categories_radar = ['活跃度\n(越高越好)', '购买频次\n(越高越好)', '消费力\n(越高越好)']
N = len(categories_radar)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# 归一化到 0-1（R 取反）
radar_data = rfm[['recency', 'frequency', 'monetary']].copy()
radar_data['recency'] = radar_data['recency'].max() - radar_data['recency']  # 反转
for col in radar_data.columns:
    vmin, vmax = radar_data[col].min(), radar_data[col].max()
    if vmax > vmin:
        radar_data[f'{col}_norm'] = (radar_data[col] - vmin) / (vmax - vmin)

radar_profile = radar_data.groupby(rfm['cluster'])[
    ['recency_norm', 'frequency_norm', 'monetary_norm']].mean()

for ci in range(best_k):
    values = radar_profile.loc[ci].values.flatten().tolist()
    values += values[:1]
    ax.fill(angles, values, alpha=0.15, color=colors[ci])
    ax.plot(angles, values, 'o-', linewidth=2, color=colors[ci],
            label=cluster_profile.loc[ci, 'label'])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories_radar, fontsize=10)
ax.set_ylim(0, 1)
ax.set_title('聚类特征雷达图', pad=25, fontsize=14)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'cluster_radar.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  已保存: images/cluster_radar.png")

# ============================================================
# 保存结果
# ============================================================
rfm_output = rfm[['user_id', 'recency', 'frequency', 'monetary', 'total_actions', 'cluster']]
rfm_output['cluster_label'] = rfm_output['cluster'].map(cluster_profile['label'])
rfm_output.to_csv(os.path.join(DATA_DIR, 'data', 'processed', 'rfm_clusters.csv'), index=False)
cluster_profile.to_csv(os.path.join(DATA_DIR, 'data', 'processed', 'cluster_profile.csv'))

print(f"\n  结果已保存:")
print(f"    data/processed/rfm_clusters.csv — 全量用户 RFM + 聚类标签")
print(f"    data/processed/cluster_profile.csv — 聚类画像汇总")
print(f"\n第三阶段完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
