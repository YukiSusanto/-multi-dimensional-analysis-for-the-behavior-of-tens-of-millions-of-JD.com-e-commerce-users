-- ============================================================
-- 01_funnel.sql — 电商转化漏斗分析
-- 计算 pv → (cart+fav) → buy 三层转化率
-- ============================================================

WITH funnel_raw AS (
    SELECT
        item_category,
        COUNT(DISTINCT CASE WHEN behavior = 'pv'   THEN user_id END) AS pv_users,
        COUNT(DISTINCT CASE WHEN behavior IN ('cart', 'fav') THEN user_id END) AS interested_users,
        COUNT(DISTINCT CASE WHEN behavior = 'buy'  THEN user_id END) AS buy_users
    FROM 'data/processed/user_behavior_clean.csv'
    GROUP BY item_category
)
SELECT
    item_category,
    pv_users,
    interested_users,
    buy_users,
    ROUND(interested_users * 100.0 / NULLIF(pv_users, 0), 2)   AS pv_to_interest_pct,
    ROUND(buy_users * 100.0 / NULLIF(interested_users, 0), 2)   AS interest_to_buy_pct,
    ROUND(buy_users * 100.0 / NULLIF(pv_users, 0), 2)           AS pv_to_buy_pct,
    pv_users - interested_users   AS lost_at_interest,
    interested_users - buy_users   AS lost_at_buy
FROM funnel_raw
WHERE pv_users >= 100
ORDER BY pv_users DESC
LIMIT 30;
