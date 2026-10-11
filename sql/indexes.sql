-- FraudLens indexes (Part 04). Applied AFTER bulk load (the loader runs this
-- last) because building indexes on a populated table is far faster than
-- maintaining them during COPY. Idempotent: CREATE INDEX IF NOT EXISTS.
--
-- Each index below names the query it serves. Over-indexing a write-heavy raw
-- table is avoided: only the access paths the API (Part 13) and analytics need.

-- The API's customer-history lookup: WHERE customer_id = ? AND timestamp < ?
-- ORDER BY timestamp DESC LIMIT n. Composite order (customer_id, timestamp)
-- lets a single index satisfy filter + range + order (no sort).
CREATE INDEX IF NOT EXISTS idx_transactions_customer_ts
    ON raw.transactions (customer_id, timestamp);

-- Device-history lookup (Part 05/13): WHERE device_id = ? AND timestamp < ?.
CREATE INDEX IF NOT EXISTS idx_transactions_device_ts
    ON raw.transactions (device_id, timestamp);

-- Global time-range scans: analytics views and EDA over a date window.
CREATE INDEX IF NOT EXISTS idx_transactions_ts
    ON raw.transactions (timestamp);

-- BRIN is a compact alternative for the append-only, naturally ordered
-- timestamp column: tiny on disk, good for large sequential time-range scans
-- where a B-tree would be bulky. Kept alongside the B-tree as a comparison.
CREATE INDEX IF NOT EXISTS brin_transactions_ts
    ON raw.transactions USING BRIN (timestamp);

-- Fraud analysis touches only the (rare) fraudulent rows: a partial index keeps
-- it small (a fraction of a percent of the table) and fast.
CREATE INDEX IF NOT EXISTS idx_labels_fraud
    ON raw.labels (transaction_id) WHERE is_fraud;
