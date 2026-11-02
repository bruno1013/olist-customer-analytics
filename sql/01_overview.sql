-- =============================================================================
-- 01_overview.sql — Dataset overview and headline KPIs
-- Database: data/olist.db (SQLite)
-- =============================================================================

-- 1.1 Size of the business: orders, customers (people), sellers, products
SELECT
    (SELECT COUNT(*) FROM orders)                                 AS total_orders,
    (SELECT COUNT(DISTINCT customer_unique_id) FROM customers)    AS unique_customers,
    (SELECT COUNT(*) FROM sellers)                                AS sellers,
    (SELECT COUNT(*) FROM products)                               AS products;

-- 1.2 Orders by status (most orders should be 'delivered')
SELECT
    order_status,
    COUNT(*)                                            AS total_orders,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)  AS pct_of_orders
FROM orders
GROUP BY order_status
ORDER BY total_orders DESC;

-- 1.3 Top 10 customer states by number of unique customers
SELECT
    customer_state,
    COUNT(DISTINCT customer_unique_id) AS customers
FROM customers
GROUP BY customer_state
ORDER BY customers DESC
LIMIT 10;

-- 1.4 Review score distribution (one review per order: the most recent one)
WITH reviews_clean AS (
    SELECT
        order_id,
        review_score,
        ROW_NUMBER() OVER (
            PARTITION BY order_id
            ORDER BY review_answer_timestamp DESC
        ) AS rn
    FROM reviews
)
SELECT
    review_score,
    COUNT(*)                                            AS reviews,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)  AS pct
FROM reviews_clean
WHERE rn = 1
GROUP BY review_score
ORDER BY review_score;

-- 1.5 Headline KPIs for delivered orders
-- Revenue = product price (excludes freight). AOV = average order value.
WITH delivered_orders AS (
    SELECT o.order_id, SUM(oi.price) AS order_revenue
    FROM orders o
    JOIN order_items oi ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY o.order_id
)
SELECT
    COUNT(*)                         AS delivered_orders,
    ROUND(SUM(order_revenue), 0)     AS revenue_brl,
    ROUND(AVG(order_revenue), 2)     AS avg_order_value_brl
FROM delivered_orders;

-- 1.6 Repeat-purchase rate: share of customers (people) with 2+ orders
WITH orders_per_customer AS (
    SELECT c.customer_unique_id, COUNT(*) AS n_orders
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id
)
SELECT
    COUNT(*)                                          AS customers,
    SUM(n_orders >= 2)                                AS repeat_customers,
    ROUND(100.0 * SUM(n_orders >= 2) / COUNT(*), 2)   AS repeat_rate_pct
FROM orders_per_customer;
