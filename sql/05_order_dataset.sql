-- =============================================================================
-- 05_order_dataset.sql — One row per delivered order, ready for analysis/ML
-- Used by src/build_datasets.py → data/processed/orders.parquet
--
-- Performance note: CTEs that keep "rn = 1" are marked MATERIALIZED so SQLite
-- computes them once and joins them on order_id. Without the hint the planner
-- inlines them and searches by rn only, which makes the query run for minutes.
-- =============================================================================

WITH reviews_clean AS MATERIALIZED (
    -- most recent review per order (some orders have several).
    -- rn = 1 is filtered INSIDE the CTE so later joins only need order_id.
    SELECT * FROM (
        SELECT
            order_id,
            review_score,
            review_creation_date,
            review_comment_message,
            ROW_NUMBER() OVER (PARTITION BY order_id
                               ORDER BY review_answer_timestamp DESC) AS rn
        FROM reviews
    )
    WHERE rn = 1
),
items AS (
    -- order-level totals from the item lines
    SELECT
        order_id,
        COUNT(*)                  AS n_items,
        COUNT(DISTINCT seller_id) AS n_sellers,
        SUM(price)                AS revenue,
        SUM(freight_value)        AS freight
    FROM order_items
    GROUP BY order_id
),
main_item AS MATERIALIZED (
    -- the most expensive item defines the order's main seller and category
    SELECT * FROM (
        SELECT
            oi.order_id,
            oi.seller_id,
            COALESCE(t.product_category_name_english, p.product_category_name, 'unknown') AS category,
            p.product_weight_g,
            p.product_photos_qty,
            ROW_NUMBER() OVER (PARTITION BY oi.order_id
                               ORDER BY oi.price DESC, oi.order_item_id) AS rn
        FROM order_items oi
        LEFT JOIN products p ON p.product_id = oi.product_id
        LEFT JOIN category_translation t
                             ON t.product_category_name = p.product_category_name
    )
    WHERE rn = 1
),
pay AS (
    -- total paid and max installments per order
    SELECT
        order_id,
        SUM(payment_value)        AS paid,
        MAX(payment_installments) AS installments
    FROM payments
    GROUP BY order_id
),
pay_main AS MATERIALIZED (
    -- payment type that covered the largest share of the order
    SELECT * FROM (
        SELECT
            order_id,
            payment_type,
            ROW_NUMBER() OVER (PARTITION BY order_id
                               ORDER BY payment_value DESC) AS rn
        FROM payments
    )
    WHERE rn = 1
)
SELECT
    o.order_id,
    c.customer_unique_id,
    m.seller_id,
    c.customer_state,
    s.seller_state,
    (c.customer_state = s.seller_state)                         AS same_state,
    o.order_purchase_timestamp                                  AS purchase_ts,
    strftime('%Y-%m', o.order_purchase_timestamp)               AS purchase_month,
    o.order_approved_at                                         AS approved_ts,
    o.order_delivered_carrier_date                              AS carrier_ts,
    o.order_delivered_customer_date                             AS delivered_ts,
    o.order_estimated_delivery_date                             AS estimated_ts,
    -- delivery timing (days)
    julianday(o.order_delivered_customer_date)
      - julianday(o.order_purchase_timestamp)                   AS days_to_deliver,
    julianday(o.order_estimated_delivery_date)
      - julianday(o.order_purchase_timestamp)                   AS days_promised,
    julianday(date(o.order_delivered_customer_date))
      - julianday(date(o.order_estimated_delivery_date))        AS delay_days,
    (date(o.order_delivered_customer_date)
      > date(o.order_estimated_delivery_date))                  AS is_late,
    -- basket
    i.n_items,
    i.n_sellers,
    i.revenue,
    i.freight,
    m.category,
    m.product_weight_g,
    m.product_photos_qty,
    -- payment
    p.paid,
    p.installments,
    pm.payment_type,
    -- review (NULL when the order has no review)
    r.review_score,
    (r.review_score <= 2)                                       AS is_bad_review,
    r.review_creation_date                                      AS review_ts,
    r.review_comment_message                                    AS review_comment
FROM orders o
JOIN customers c          ON c.customer_id = o.customer_id
JOIN items i              ON i.order_id = o.order_id
JOIN main_item m          ON m.order_id = o.order_id
LEFT JOIN sellers s       ON s.seller_id = m.seller_id
LEFT JOIN pay p           ON p.order_id = o.order_id
LEFT JOIN pay_main pm     ON pm.order_id = o.order_id
LEFT JOIN reviews_clean r ON r.order_id = o.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL;
