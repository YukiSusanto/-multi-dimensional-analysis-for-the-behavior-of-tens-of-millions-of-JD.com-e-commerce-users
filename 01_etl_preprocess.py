"""
第一阶段：数据预处理 ETL
======================
- 时间字段解析（event_date + event_hour）
- behavior_type 映射为可读标签
- 异常值检测与清洗
- 内存优化（类型压缩）
- 输出清洗后 CSV → data/processed/
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# === 配置 ===
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(DATA_DIR, 'data', 'raw', '京东电商数据.csv')
OUT_PATH = os.path.join(DATA_DIR, 'data', 'processed', 'user_behavior_clean.csv')

BEHAVIOR_MAP = {
    1: 'pv',
    2: 'buy',
    3: 'cart',
    4: 'fav'
}

print("=" * 60)
print("第一阶段：数据预处理 ETL")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ============================================================
# Step 1: 加载数据
# ============================================================
print("\n[1/5] 加载原始数据...")
dtypes_raw = {
    'user_id': 'int32',
    'item_id': 'int32',
    'behavior_type': 'int8',
    'user_geohash': 'str',     # 68% 缺失，加载后丢弃
    'item_category': 'int32',
    'time': 'str'
}

df = pd.read_csv(RAW_PATH, dtype=dtypes_raw, usecols=['user_id', 'item_id', 'behavior_type', 'item_category', 'time'])
mem_before = df.memory_usage(deep=True).sum() / 1024**2
print(f"  加载完成: {len(df):,} 行, 内存 {mem_before:.1f} MB")
print(f"  列: {list(df.columns)}")

# ============================================================
# Step 2: 时间字段解析
# ============================================================
print("\n[2/5] 时间字段解析...")
df['event_datetime'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H')
df['event_date'] = df['event_datetime'].dt.date
df['event_hour'] = df['event_datetime'].dt.hour.astype('int8')
df['event_weekday'] = df['event_datetime'].dt.dayofweek.astype('int8')
df.drop(columns=['time', 'event_datetime'], inplace=True)

date_min = df['event_date'].min()
date_max = df['event_date'].max()
print(f"  时间范围: {date_min} ~ {date_max} ({(pd.Timestamp(date_max) - pd.Timestamp(date_min)).days + 1} 天)")

# ============================================================
# Step 3: behavior_type 映射与验证
# ============================================================
print("\n[3/5] behavior_type 映射与异常检测...")

# 映射
df['behavior'] = df['behavior_type'].map(BEHAVIOR_MAP)
invalid_bt = df['behavior'].isna().sum()
if invalid_bt > 0:
    print(f"  [WARNING] 发现 {invalid_bt} 条无效行为类型，已剔除")
    df = df[df['behavior'].notna()].copy()
df.drop(columns=['behavior_type'], inplace=True)

# 分布
bt_counts = df['behavior'].value_counts()
for k, v in bt_counts.items():
    print(f"  {k}: {v:,} ({v/len(df)*100:.1f}%)")

# ============================================================
# Step 4: 逻辑异常检测
# ============================================================
print("\n[4/5] 逻辑异常检测...")

anomaly_report = []

# 4a) 缺失值统计
null_count = df.isnull().sum()
null_fields = null_count[null_count > 0]
if len(null_fields) > 0:
    anomaly_report.append(f"缺失值: {dict(null_fields)}")
if len(null_fields) == 0:
    print("  缺失值检查: 通过")
else:
    print(f"  缺失值检查: 发现缺失\n    {dict(null_fields)}")

# 4b) 每个用户每天每种行为去重（同一用户对同一商品同一天的同种行为只保留一条）
before_dedup = len(df)
df = df.drop_duplicates(subset=['user_id', 'item_id', 'event_date', 'behavior'])
after_dedup = len(df)
if before_dedup > after_dedup:
    anomaly_report.append(f"去重: 移除 {before_dedup - after_dedup:,} 条重复记录")
    print(f"  去重: 移除 {before_dedup - after_dedup:,} 条重复记录 ({(before_dedup - after_dedup)/before_dedup*100:.2f}%)")

# 4c) 检测购买行为逻辑异常：同一用户对同一商品的购买是否在点击之前
buy_df = df[df['behavior'] == 'buy'][['user_id', 'item_id', 'event_date']].drop_duplicates()
pv_df = df[df['behavior'] == 'pv'][['user_id', 'item_id', 'event_date']].drop_duplicates()

# Merge: 找到有购买但没有当天或之前点击的
buy_with_pv = buy_df.merge(pv_df, on=['user_id', 'item_id'], suffixes=('_buy', '_pv'))
anomalous_buys = buy_with_pv[buy_with_pv['event_date_buy'] < buy_with_pv['event_date_pv']]
if len(anomalous_buys) > 0:
    anomaly_report.append(f"购买早于点击: {len(anomalous_buys)} 条")
    print(f"  购买早于点击异常: {len(anomalous_buys)} 条")

# ============================================================
# Step 5: 内存优化
# ============================================================
print("\n[5/5] 内存优化...")
mem_before_opt = df.memory_usage(deep=True).sum() / 1024**2

# 类型压缩
df['user_id'] = df['user_id'].astype('int32')
df['item_id'] = df['item_id'].astype('int32')
df['item_category'] = df['item_category'].astype('int32')
df['event_hour'] = df['event_hour'].astype('int8')
df['event_weekday'] = df['event_weekday'].astype('int8')
df['behavior'] = df['behavior'].astype('category')
df['event_date'] = pd.to_datetime(df['event_date'])

mem_after_opt = df.memory_usage(deep=True).sum() / 1024**2

print(f"  优化前内存: {mem_before:.1f} MB")
print(f"  优化后内存: {mem_after_opt:.1f} MB")
print(f"  内存降幅:   {(1 - mem_after_opt/mem_before)*100:.1f}%")

# ============================================================
# 输出
# ============================================================
print(f"\n保存清洗后数据至: {OUT_PATH}")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df.to_csv(OUT_PATH, index=False)

# 最终统计
print("\n" + "=" * 60)
print("ETL 完成摘要")
print("=" * 60)
print(f"  最终行数:       {len(df):,}")
print(f"  唯一用户数:     {df['user_id'].nunique():,}")
print(f"  唯一商品数:     {df['item_id'].nunique():,}")
print(f"  唯一品类数:     {df['item_category'].nunique():,}")
print(f"  时间跨度:       {date_min} ~ {date_max}")
print(f"  最终文件大小:   {os.path.getsize(OUT_PATH)/1024**2:.1f} MB")
print(f"  异常报告:       {'; '.join(anomaly_report) if anomaly_report else '无严重异常'}")
print(f"  结束时间:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
