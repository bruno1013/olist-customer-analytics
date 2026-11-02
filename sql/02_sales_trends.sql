-- =============================================================================
-- 02_sales_trends.sql — Monthly trends and category performance
-- Only complete months are used (Jan 2017 – Aug 2018): 2016 is the platform's
-- ramp-up and Sep/Oct 2018 were cut off by the data extraction.
-- =============================================================================

-- 2.1 Monthly orders, revenue, AOV and month-over-month growth
WITH monthly AS (
    SELECT
        strftime('%Y-%m', o.order_purchase_timestamp) AS month,
        COUNT(DISTINCT o.order_id)                     AS orders,
        SUM(oi.price)                                  AS revenue
    FROM orders o
    JOIN order_items oi ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp >= '2017-01-01'
      AND o.order_purchase_timestamp <  '2018-09-01'
    GROUP BY month
)
SELECT
    month,
    orders,
    ROUND(revenue, 0)                    AS revenue_brl,
    ROUND(revenue / orders, 2)           AS aov_brl,
    -- LAG() looks at the previous row (previous month) in the given order
    ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
          / LAG(revenue) OVER (ORDER BY month), 1) AS revenue_mom_pct
FROM monthly
ORDER BY month;

-- 2.2 Top 10 product categories by revenue (English names)
SELECT
    COALESCE(t.product_category_name_english, p.product_category_name, 'unknown') AS category,
    COUNT(DISTINCT oi.order_id)   AS orders,
    ROUND(SUM(oi.price), 0)       AS revenue_brl,
    ROUND(AVG(oi.price), 2)       AS avg_item_price_brl
FROM order_items oi
JOIN orders o            ON o.order_id = oi.order_id
LEFT JOIN products p     ON p.product_id = oi.product_id
LEFT JOIN category_translation t
                         ON t.product_category_name = p.product_category_name
WHERE o.order_status = 'delivered'
GROUP BY category
ORDER BY revenue_brl DESC
LIMIT 10;

-- 2.3 Pareto: how many categories make up 80% of revenue?
WITH category_revenue AS (
    SELECT
        COALESCE(t.product_category_name_english, p.product_category_name, 'unknown') AS category,
        SUM(oi.price) AS revenue
    FROM order_items oi
    JOIN orders o        ON o.order_id = oi.order_id
    LEFT JOIN products p ON p.product_id = oi.product_id
    LEFT JOIN category_translation t
                         ON t.product_category_name = p.product_category_name
    WHERE o.order_status = 'delivered'
    GROUP BY category
),
ranked AS (
    SELECT
        category,
        revenue,
        ROW_NUMBER() OVER (ORDER BY revenue DESC)                     AS rank,
        -- running total: sum of this row and every row ranked above it
        SUM(revenue) OVER (ORDER BY revenue DESC) / SUM(revenue) OVER () AS cum_share
    FROM category_revenue
)
SELECT
    COUNT(*)                                           AS total_categories,
    SUM(cum_share <= 0.80) + 1                         AS categories_for_80pct_revenue
FROM ranked;
