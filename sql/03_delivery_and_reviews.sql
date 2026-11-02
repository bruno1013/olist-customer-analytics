-- =============================================================================
-- 03_delivery_and_reviews.sql — Does late delivery hurt customer satisfaction?
-- Scope: delivered orders with a delivery date and a review.
-- julianday() turns a date into a number of days, so subtracting two of them
-- gives the difference in days.
-- =============================================================================

-- 3.1 Delivery performance: how long do orders take, and how many are late?
SELECT
    COUNT(*) AS delivered_orders,
    ROUND(AVG(julianday(order_delivered_customer_date)
              - julianday(order_purchase_timestamp)), 1)        AS avg_days_to_deliver,
    ROUND(AVG(julianday(order_estimated_delivery_date)
              - julianday(order_purchase_timestamp)), 1)        AS avg_days_promised,
    ROUND(100.0 * AVG(date(order_delivered_customer_date)
                      > date(order_estimated_delivery_date)), 1) AS late_pct
FROM orders
WHERE order_status = 'delivered'
  AND order_delivered_customer_date IS NOT NULL;

-- 3.2 Average review score by delivery delay bucket
-- delay_days < 0 means the order arrived BEFORE the promised date.
WITH reviews_clean AS (
    SELECT order_id, review_score,
           ROW_NUMBER() OVER (PARTITION BY order_id
                              ORDER BY review_answer_timestamp DESC) AS rn
    FROM reviews
),
order_delay AS (
    SELECT
        o.order_id,
        r.review_score,
        julianday(date(o.order_delivered_customer_date))
          - julianday(date(o.order_estimated_delivery_date)) AS delay_days
    FROM orders o
    JOIN reviews_clean r ON r.order_id = o.order_id AND r.rn = 1
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
)
SELECT
    CASE
        WHEN delay_days <= -7 THEN '1. 7+ days early'
        WHEN delay_days <   0 THEN '2. 1-6 days early'
        WHEN delay_days =   0 THEN '3. on the promised day'
        WHEN delay_days <=  3 THEN '4. 1-3 days late'
        WHEN delay_days <=  7 THEN '5. 4-7 days late'
        ELSE                       '6. 8+ days late'
    END                                               AS delivery_bucket,
    COUNT(*)                                          AS orders,
    ROUND(AVG(review_score), 2)                       AS avg_review,
    ROUND(100.0 * AVG(review_score <= 2), 1)          AS bad_review_pct
FROM order_delay
GROUP BY delivery_bucket
ORDER BY delivery_bucket;

-- 3.3 Delivery and satisfaction by customer state (states with 500+ orders)
WITH reviews_clean AS (
    SELECT order_id, review_score,
           ROW_NUMBER() OVER (PARTITION BY order_id
                              ORDER BY review_answer_timestamp DESC) AS rn
    FROM reviews
)
SELECT
    c.customer_state,
    COUNT(*)                                                         AS orders,
    ROUND(AVG(julianday(o.order_delivered_customer_date)
              - julianday(o.order_purchase_timestamp)), 1)           AS avg_days_to_deliver,
    ROUND(100.0 * AVG(date(o.order_delivered_customer_date)
                      > date(o.order_estimated_delivery_date)), 1)   AS late_pct,
    ROUND(AVG(r.review_score), 2)                                    AS avg_review
FROM orders o
JOIN customers c     ON c.customer_id = o.customer_id
JOIN reviews_clean r ON r.order_id = o.order_id AND r.rn = 1
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
GROUP BY c.customer_state
HAVING COUNT(*) >= 500
ORDER BY late_pct DESC;
