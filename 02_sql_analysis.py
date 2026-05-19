"""
第二阶段：SQL 深度分析
=====================
通过 DuckDB 执行漏斗分析、留存率计算、24h 时间分布
"""
import duckdb
import os
import json
from datetime import datetime

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQL_DIR  = os.path.join(DATA_DIR, 'sql')
CSV_PATH = os.path.join(DATA_DIR, 'data', 'processed', 'user_behavior_clean.csv')
OUT_DIR  = os.path.join(DATA_DIR, 'data', 'processed')

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("第二阶段：SQL 深度分析")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

con = duckdb.connect()

# ============================================================
# 2.1 整体漏斗
# ============================================================
print("\n[2.1] 整体转化漏斗...")
overall_funnel = con.execute(f"""
    SELECT
        COUNT(DISTINCT CASE WHEN behavior = 'pv'   THEN user_id END) AS pv_users,
        COUNT(DISTINCT CASE WHEN behavior IN ('cart', 'fav') THEN user_id END) AS interested_users,
        COUNT(DISTINCT CASE WHEN behavior = 'buy'  THEN user_id END) AS buy_users,
        COUNT(DISTINCT CASE WHEN behavior = 'cart' THEN user_id END) AS cart_users,
        COUNT(DISTINCT CASE WHEN behavior = 'fav'  THEN user_id END) AS fav_users
    FROM '{CSV_PATH}'
""").fetchone()

pv_u, int_u, buy_u, cart_u, fav_u = overall_funnel
print(f"""
  ┌─────────────────────────────────────────┐
  │  pv (点击)          {pv_u:>10,}         │
  │  cart (加购)        {cart_u:>10,}  ({cart_u/pv_u*100:.1f}%)│
  │  fav  (收藏)        {fav_u:>10,}  ({fav_u/pv_u*100:.1f}%)│
  │  interest (加购+收藏) {int_u:>10,}  ({int_u/pv_u*100:.1f}%)│
  │  buy (购买)          {buy_u:>10,}  ({buy_u/pv_u*100:.1f}%)│
  └─────────────────────────────────────────┘
""")

# ============================================================
# 2.2 品类漏斗 (TOP 20)
# ============================================================
print("\n[2.2] 品类漏斗 TOP 20...")
with open(os.path.join(SQL_DIR, '01_funnel.sql'), 'r') as f:
    sql = f.read().replace("data/processed/user_behavior_clean.csv", CSV_PATH)
category_funnel = con.execute(sql).fetchdf()

# 显示 TOP 10
print(f"\n{'品类':>10} | {'点击用户':>10} | {'意向用户':>10} | {'购买用户':>10} | {'点击→购买%':>10} | {'意向→购买%':>10}")
print("-" * 85)
for _, row in category_funnel.head(10).iterrows():
    print(f"{int(row['item_category']):>10} | {int(row['pv_users']):>10,} | "
          f"{int(row['interested_users']):>10,} | {int(row['buy_users']):>10,} | "
          f"{row['pv_to_buy_pct']:>9.1f}% | {row['interest_to_buy_pct']:>9.1f}%")

category_funnel.to_csv(os.path.join(OUT_DIR, 'funnel_by_category.csv'), index=False)

# ============================================================
# 2.3 留存率
# ============================================================
print("\n[2.3] 用户留存率分析...")
with open(os.path.join(SQL_DIR, '02_retention.sql'), 'r') as f:
    sql = f.read().replace("data/processed/user_behavior_clean.csv", CSV_PATH)
retention = con.execute(sql).fetchdf()

print(f"\n{'首次日期':>12} | {'新增用户':>10} | {'次日留存':>10} | {'次日率%':>8} | {'7日留存':>10} | {'7日率%':>8}")
print("-" * 82)
for _, row in retention.iterrows():
    fd = row['first_date']
    print(f"  {str(fd):>10} | {int(row['new_users']):>10,} | "
          f"{int(row['day1_retained']):>10,} | {row['day1_rate_pct']:>7.1f}% | "
          f"{int(row['day7_retained']):>10,} | {row['day7_rate_pct']:>7.1f}%")

avg_d1 = retention['day1_rate_pct'].mean()
avg_d7 = retention['day7_rate_pct'].mean()
print(f"\n  平均次日留存率: {avg_d1:.2f}%")
print(f"  平均7日留存率:  {avg_d7:.2f}%")

retention.to_csv(os.path.join(OUT_DIR, 'retention_daily.csv'), index=False)

# ============================================================
# 2.4 时间分析
# ============================================================
print("\n[2.4] 24h 活跃热力图...")
with open(os.path.join(SQL_DIR, '03_time_analysis.sql'), 'r') as f:
    sql = f.read().replace("data/processed/user_behavior_clean.csv", CSV_PATH)
hourly = con.execute(sql).fetchdf()

# 汇总到小时维度
hourly_agg = hourly.groupby('event_hour').agg({
    'total_actions': 'sum', 'pv_count': 'sum', 'cart_count': 'sum',
    'buy_count': 'sum', 'fav_count': 'sum'
}).reset_index()
hourly_agg['buy_rate_pct'] = hourly_agg['buy_count'] * 100.0 / hourly_agg['total_actions']

print(f"\n  {'小时':>5} | {'总行为':>12} | {'购买量':>10} | {'购买率%':>8}")
print("  " + "-" * 50)
for _, row in hourly_agg.iterrows():
    bar = '█' * int(row['buy_count'] / hourly_agg['buy_count'].max() * 30)
    print(f"  {int(row['event_hour']):>5} | {int(row['total_actions']):>12,} | "
          f"{int(row['buy_count']):>10,} | {row['buy_rate_pct']:>7.2f}%  {bar}")

peak_hour = hourly_agg.loc[hourly_agg['buy_count'].idxmax(), 'event_hour']
print(f"\n  下单高峰: {int(peak_hour)}:00")

hourly.to_csv(os.path.join(OUT_DIR, 'hourly_activity.csv'), index=False)

# ============================================================
# 2.5 用户行为统计摘要
# ============================================================
print("\n[2.5] 用户行为统计...")
user_stats = con.execute(f"""
    SELECT
        COUNT(DISTINCT user_id) AS total_users,
        AVG(action_count)       AS avg_actions_per_user,
        MEDIAN(action_count)    AS median_actions_per_user,
        MAX(action_count)       AS max_actions
    FROM (
        SELECT user_id, COUNT(*) AS action_count
        FROM '{CSV_PATH}'
        GROUP BY user_id
    )
""").fetchone()

print(f"  总用户数:             {user_stats[0]:,}")
print(f"  人均行为数 (均值):     {user_stats[1]:.0f}")
print(f"  人均行为数 (中位数):   {user_stats[2]:.0f}")
print(f"  单人最大行为数:       {user_stats[3]:,}")

con.close()
print(f"\n第二阶段完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
