-- ============================================================
-- 02_retention.sql — 用户留存率分析
-- 计算次日/7日/30日留存率（Self-Join + CTE）
-- ============================================================

WITH first_purchase AS (
    -- 每个用户的首次购买日期
    SELECT
        user_id,
        MIN(event_date) AS first_date
    FROM 'data/processed/user_behavior_clean.csv'
    WHERE behavior = 'buy'
    GROUP BY user_id
),
daily_cohort AS (
    -- 每日新增购买用户数
    SELECT
        first_date,
        COUNT(DISTINCT user_id) AS new_users
    FROM first_purchase
    GROUP BY first_date
),
-- 次日留存（自连接）
day1_retained AS (
    SELECT
        f.first_date,
        COUNT(DISTINCT b.user_id) AS retained_users
    FROM first_purchase f
    INNER JOIN 'data/processed/user_behavior_clean.csv' b
        ON f.user_id = b.user_id
       AND b.event_date = f.first_date + INTERVAL '1 day'
       AND b.behavior = 'buy'
    GROUP BY f.first_date
),
-- 7日留存
day7_retained AS (
    SELECT
        f.first_date,
        COUNT(DISTINCT b.user_id) AS retained_users
    FROM first_purchase f
    INNER JOIN 'data/processed/user_behavior_clean.csv' b
        ON f.user_id = b.user_id
       AND b.event_date >= f.first_date + INTERVAL '1 day'
       AND b.event_date <= f.first_date + INTERVAL '7 day'
       AND b.behavior = 'buy'
    GROUP BY f.first_date
)
SELECT
    c.first_date,
    c.new_users,
    -- 次日留存
    COALESCE(d1.retained_users, 0)                 AS day1_retained,
    ROUND(COALESCE(d1.retained_users, 0) * 100.0 / c.new_users, 2) AS day1_rate_pct,
    -- 7日留存
    COALESCE(d7.retained_users, 0)                 AS day7_retained,
    ROUND(COALESCE(d7.retained_users, 0) * 100.0 / c.new_users, 2) AS day7_rate_pct
FROM daily_cohort c
LEFT JOIN day1_retained d1 ON c.first_date = d1.first_date
LEFT JOIN day7_retained d7 ON c.first_date = d7.first_date
ORDER BY c.first_date;
