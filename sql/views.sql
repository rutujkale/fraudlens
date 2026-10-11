-- FraudLens analytical views (Part 04). CREATE OR REPLACE so re-running is
-- safe. These are for analytics/monitoring ONLY; ML features stay in Python
-- (Part 05). Column sets must not change without a --reset (CREATE OR REPLACE
-- VIEW cannot alter an existing view's column list).

-- Transactions joined to their customer, merchant and location context.
-- Deliberately NO labels join (a label-free enriched view is safe to expose).
CREATE OR REPLACE VIEW curated.transactions_enriched AS
SELECT
    t.transaction_id,
    t.customer_id,
    t.account_id,
    t.timestamp,
    t.amount,
    t.merchant_category,
    t.transaction_type,
    t.channel,
    t.device_id,
    c.archetype,
    c.income_band,
    c.home_region,
    m.merchant_risk_group,
    l.city      AS tx_city,
    l.region_group AS tx_region,
    hl.city     AS home_city
FROM raw.transactions t
JOIN raw.customers c  ON c.customer_id = t.customer_id
LEFT JOIN raw.merchants m ON m.merchant_id = t.merchant_id
JOIN raw.locations l  ON l.location_id = t.location_id
LEFT JOIN raw.locations hl ON hl.location_id = c.home_location_id;

-- 1. Daily volume / fraud activity. Joins labels (analytics only).
CREATE OR REPLACE VIEW curated.daily_summary AS
SELECT
    t.timestamp::date                                   AS activity_date,
    COUNT(*)                                            AS tx_count,
    COUNT(*) FILTER (WHERE l.is_fraud)                  AS fraud_count,
    ROUND(AVG(l.is_fraud::int)::numeric, 6)             AS fraud_rate,
    SUM(t.amount)                                       AS total_amount,
    COALESCE(SUM(t.amount) FILTER (WHERE l.is_fraud), 0) AS fraud_amount
FROM raw.transactions t
JOIN raw.labels l ON l.transaction_id = t.transaction_id
GROUP BY 1;

-- 2. Fraud by scenario and drift period (A/B). Period B = timestamp at/after the
--    config-driven boundary in curated.drift_boundary (NULL boundary => all A).
CREATE OR REPLACE VIEW curated.scenario_summary AS
WITH b AS (
    SELECT COALESCE((SELECT boundary_ts FROM curated.drift_boundary LIMIT 1),
                    'infinity'::timestamp) AS boundary_ts
)
SELECT
    l.fraud_scenario,
    CASE WHEN t.timestamp >= b.boundary_ts THEN 'B' ELSE 'A' END AS drift_period,
    COUNT(*)                                    AS tx_count,
    ROUND(AVG(t.amount)::numeric, 2)            AS avg_amount,
    ROUND(
        COUNT(*)::numeric
        / SUM(COUNT(*)) OVER (
            PARTITION BY CASE WHEN t.timestamp >= b.boundary_ts THEN 'B' ELSE 'A' END
        ), 6
    )                                           AS share_of_period_fraud
FROM raw.transactions t
JOIN raw.labels l ON l.transaction_id = t.transaction_id
CROSS JOIN b
WHERE l.is_fraud
GROUP BY l.fraud_scenario, drift_period;

-- 3. Per-customer rollup (median via percentile_cont; fraud count via labels).
CREATE OR REPLACE VIEW curated.customer_summary AS
SELECT
    t.customer_id,
    COUNT(*)                                                AS tx_count,
    ROUND(AVG(t.amount)::numeric, 2)                        AS avg_amount,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY t.amount)::numeric, 2) AS median_amount,
    COUNT(DISTINCT t.device_id)                             AS distinct_devices,
    COUNT(DISTINCT t.location_id)                           AS distinct_locations,
    MIN(t.timestamp)                                        AS first_tx,
    MAX(t.timestamp)                                        AS last_tx,
    COUNT(*) FILTER (WHERE l.is_fraud)                      AS fraud_count
FROM raw.transactions t
JOIN raw.labels l ON l.transaction_id = t.transaction_id
GROUP BY t.customer_id;

-- 4. Volume and fraud rate by hour-of-day x weekday (1=Mon .. 7=Sun).
CREATE OR REPLACE VIEW curated.hourly_profile AS
SELECT
    EXTRACT(HOUR FROM t.timestamp)::int                     AS hour_of_day,
    EXTRACT(ISODOW FROM t.timestamp)::int                   AS weekday,
    COUNT(*)                                                AS tx_count,
    COUNT(*) FILTER (WHERE l.is_fraud)                      AS fraud_count,
    ROUND(AVG(l.is_fraud::int)::numeric, 6)                 AS fraud_rate
FROM raw.transactions t
JOIN raw.labels l ON l.transaction_id = t.transaction_id
GROUP BY 1, 2;

-- 5. Review-load placeholder for Part 07: daily count of transactions above a
--    high amount percentile. 0.95 is a placeholder cutoff; Part 07 replaces it
--    with the cost-based, capacity-aware policy.
CREATE OR REPLACE VIEW curated.review_load_by_day AS
WITH threshold AS (
    SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount) AS amount_cut
    FROM raw.transactions
)
SELECT
    t.timestamp::date      AS activity_date,
    COUNT(*)               AS tx_count,
    COUNT(*) FILTER (WHERE t.amount >= th.amount_cut) AS high_amount_count
FROM raw.transactions t
CROSS JOIN threshold th
GROUP BY 1;

-- 6. SQL cross-check for the Part 05 rolling-24h feature (count of the same
--    customer's STRICTLY PRIOR transactions in the last 24h). EXCLUDE CURRENT
--    ROW is available since PostgreSQL 11.
--    CAVEAT: with a RANGE frame, rows sharing the current row's timestamp are
--    PEERS and EXCLUDE CURRENT ROW drops only the current physical row, not its
--    peers. Part 02 keeps each customer's timestamps strictly increasing, so
--    peers should not occur; Part 05's Python feature (strict `timestamp < T`)
--    remains the source of truth. See docs/what_went_wrong.md.
CREATE OR REPLACE VIEW curated.rolling_24h_sql_demo AS
SELECT
    t.transaction_id,
    t.customer_id,
    t.timestamp,
    COUNT(*) OVER (
        PARTITION BY t.customer_id
        ORDER BY t.timestamp
        RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
        EXCLUDE CURRENT ROW
    ) AS prior_24h_count
FROM raw.transactions t;
