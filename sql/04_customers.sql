-- =============================================================================
-- 04_customers.sql — Customer behaviour and the RFM base table
-- A "customer" is a person: customer_unique_id (customer_id changes per order).
-- =============================================================================

-- 4.1 How many orders does each customer place?
WITH orders_per_customer AS (
    SELECT c.customer_unique_id, COUNT(*) AS n_orders
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id
)
SELECT
    CASE WHEN n_orders >= 4 THEN '4+' ELSE CAST(n_orders AS TEXT) END AS orders_placed,
    COUNT(*)                                                           AS customers,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)                 AS pct
FROM orders_per_customer
GROUP BY orders_placed
ORDER BY orders_placed;

-- 4.2 For repeat customers: how many days until the second purchase?
-- LEAD() reads the NEXT row in each customer's ordered list of purchases.
WITH customer_orders AS (
    SELECT
        c.customer_unique_id,
        o.order_purchase_timestamp,
        LEAD(o.order_purchase_timestamp) OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY o.order_purchase_timestamp
        ) AS next_purchase,
        ROW_NUMBER() OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY o.order_purchase_timestamp
        ) AS purchase_number
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.order_status = 'delivered'
),
gaps AS (
    SELECT julianday(next_purchase) - julianday(order_purchase_timestamp) AS days_to_second
    FROM customer_orders
    WHERE purchase_number = 1 AND next_purchase IS NOT NULL
)
SELECT
    COUNT(*)                                   AS repeat_customers,
    ROUND(AVG(days_to_second), 0)              AS avg_days_to_second_purchase,
    ROUND(100.0 * AVG(days_to_second < 1), 1)  AS same_day_pct,
    ROUND(100.0 * AVG(days_to_second <= 90), 1) AS within_90_days_pct
FROM gaps;

-- 4.3 RFM base table (one row per customer)
-- Recency   = days since last purchase, measured from the day after the last
--             order in the dataset (the "snapshot date").
-- Frequency = number of distinct delivered orders.
-- Monetary  = total amount paid (payments include freight and installments).
WITH snapshot AS (
    SELECT date(MAX(order_purchase_timestamp), '+1 day') AS snapshot_date
    FROM orders
    WHERE order_status = 'delivered'
),
order_payments AS (
    SELECT order_id, SUM(payment_value) AS paid
    FROM payments
    GROUP BY order_id
),
rfm AS (
    SELECT
        c.customer_unique_id,
        CAST(julianday((SELECT snapshot_date FROM snapshot))
             - julianday(MAX(o.order_purchase_timestamp)) AS INTEGER) AS recency_days,
        COUNT(DISTINCT o.order_id)                                    AS frequency,
        ROUND(SUM(p.paid), 2)                                         AS monetary_brl
    FROM orders o
    JOIN customers c       ON c.customer_id = o.customer_id
    JOIN order_payments p  ON p.order_id = o.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id
)
SELECT
    COUNT(*)                        AS customers,
    ROUND(AVG(recency_days), 0)     AS avg_recency_days,
    ROUND(AVG(frequency), 3)        AS avg_frequency,
    ROUND(AVG(monetary_brl), 2)     AS avg_monetary_brl,
    MAX(monetary_brl)               AS max_monetary_brl
FROM rfm;
