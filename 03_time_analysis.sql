-- ============================================================
-- 03_time_analysis.sql — 24小时活跃热力图
-- 统计每小时活跃量与购买量，识别高峰期
-- ============================================================

WITH hourly_stats AS (
    SELECT
        event_hour,
        event_weekday,
        COUNT(*)                        AS total_actions,
        SUM(CASE WHEN behavior = 'pv'   THEN 1 ELSE 0 END) AS pv_count,
        SUM(CASE WHEN behavior = 'cart' THEN 1 ELSE 0 END) AS cart_count,
        SUM(CASE WHEN behavior = 'buy'  THEN 1 ELSE 0 END) AS buy_count,
        SUM(CASE WHEN behavior = 'fav'  THEN 1 ELSE 0 END) AS fav_count
    FROM 'data/processed/user_behavior_clean.csv'
    GROUP BY event_hour, event_weekday
)
SELECT
    event_hour,
    event_weekday,
    total_actions,
    pv_count,
    cart_count,
    buy_count,
    fav_count,
    -- 购买转化率（买 / 总行为）
    ROUND(buy_count * 100.0 / NULLIF(total_actions, 0), 2) AS buy_rate_pct
FROM hourly_stats
ORDER BY event_weekday, event_hour;
