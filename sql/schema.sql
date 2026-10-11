-- FraudLens warehouse schema (Part 04). Idempotent: safe to re-run.
-- PostgreSQL is the system of record + analytics layer. Feature logic stays in
-- Python (Part 05) so training and serving cannot diverge.
--
-- Enum values are mirrored as CHECK constraints from
-- src/fraudlens/data_gen/archetypes.py (categories/types/channels/income bands),
-- fraud.py SCENARIOS and hard_negatives.py HARD_NEGATIVE_TYPES. If those Python
-- constants change, a load will fail loudly here instead of silently diverging.

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS curated;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS model;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- ---------------------------------------------------------------------------
-- raw.* : 1:1 mirror of the simulator parquet tables.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS raw.locations (
    location_id   TEXT PRIMARY KEY,
    country       TEXT,
    state         TEXT,
    city          TEXT,
    latitude      DOUBLE PRECISION,
    longitude     DOUBLE PRECISION,
    region_group  TEXT
);

CREATE TABLE IF NOT EXISTS raw.merchants (
    merchant_id         TEXT PRIMARY KEY,
    merchant_category   TEXT CHECK (merchant_category IN (
        'grocery','fuel','dining','electronics','travel','utilities',
        'healthcare','fashion','entertainment','jewelry','transfers','wallet_topup')),
    region              TEXT,
    merchant_risk_group TEXT CHECK (merchant_risk_group IN ('low','medium','high'))
);

CREATE TABLE IF NOT EXISTS raw.customers (
    customer_id          TEXT PRIMARY KEY,
    age                  INTEGER,
    income_band          TEXT CHECK (income_band IN ('low','mid','mid_high','high')),
    home_region          TEXT,
    home_location_id     TEXT REFERENCES raw.locations(location_id),
    account_open_date    DATE,
    archetype            TEXT CHECK (archetype IN (
        'conservative','regular','high_spend','frequent_digital','business_like')),
    baseline_daily_tx    DOUBLE PRECISION,
    baseline_amount      NUMERIC(14,2),
    amount_sigma         DOUBLE PRECISION,
    -- preferred_categories is a list in parquet; JSONB keeps it queryable while
    -- the stable relational columns stay typed.
    preferred_categories JSONB,
    active_hour_mean     DOUBLE PRECISION,
    active_hour_sd       DOUBLE PRECISION,
    travel_frequency     DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS raw.accounts (
    account_id    TEXT PRIMARY KEY,
    customer_id   TEXT NOT NULL REFERENCES raw.customers(customer_id),
    account_type  TEXT CHECK (account_type IN ('SAVINGS','CURRENT')),
    opening_date  DATE,
    status        TEXT CHECK (status IN ('ACTIVE','CLOSED','DORMANT')),
    base_currency TEXT
);

CREATE TABLE IF NOT EXISTS raw.devices (
    device_id    TEXT PRIMARY KEY,
    first_seen   TIMESTAMP,
    last_seen    TIMESTAMP,
    device_type  TEXT CHECK (device_type IN ('MOBILE','TABLET','DESKTOP')),
    is_shared    BOOLEAN
);

CREATE TABLE IF NOT EXISTS raw.customer_devices (
    customer_id TEXT NOT NULL REFERENCES raw.customers(customer_id),
    device_id   TEXT NOT NULL REFERENCES raw.devices(device_id),
    PRIMARY KEY (customer_id, device_id)
);

CREATE TABLE IF NOT EXISTS raw.transactions (
    transaction_id   TEXT PRIMARY KEY,
    customer_id      TEXT NOT NULL REFERENCES raw.customers(customer_id),
    account_id       TEXT NOT NULL REFERENCES raw.accounts(account_id),
    timestamp        TIMESTAMP NOT NULL,
    amount           NUMERIC(14,2) NOT NULL,
    currency         TEXT CHECK (currency IN ('INR')),
    merchant_id      TEXT REFERENCES raw.merchants(merchant_id),
    merchant_category TEXT CHECK (merchant_category IN (
        'grocery','fuel','dining','electronics','travel','utilities',
        'healthcare','fashion','entertainment','jewelry','transfers','wallet_topup')),
    device_id        TEXT REFERENCES raw.devices(device_id),
    location_id      TEXT NOT NULL REFERENCES raw.locations(location_id),
    recipient_id     TEXT REFERENCES raw.customers(customer_id),
    transaction_type TEXT CHECK (transaction_type IN (
        'PURCHASE','P2P_TRANSFER','BILL_PAY','CASH_WITHDRAWAL')),
    channel          TEXT CHECK (channel IN ('APP','WEB','POS','ATM')),
    created_at       TIMESTAMP
);
-- NOTE: raw.transactions intentionally has NO label columns (is_fraud,
-- fraud_scenario, episode_id, stealth, hard_negative_type, label_available_at).

-- Labels live in their own table, keyed by transaction_id. Never join this into
-- feature tables (Part 05 guard enforces the same rule in Python).
CREATE TABLE IF NOT EXISTS raw.labels (
    transaction_id     TEXT PRIMARY KEY REFERENCES raw.transactions(transaction_id),
    is_fraud           BOOLEAN NOT NULL,
    fraud_scenario     TEXT CHECK (fraud_scenario IN (
        'burst','account_takeover','amount_anomaly','geo_anomaly','structuring')),
    episode_id         TEXT,
    stealth            DOUBLE PRECISION,
    hard_negative_type TEXT CHECK (hard_negative_type IN (
        'new_phone','travel_trip','holiday_splurge','medical_bill',
        'emergency_odd_hour','payday_burst','power_user_burst')),
    label_available_at TIMESTAMP NOT NULL
);

-- ---------------------------------------------------------------------------
-- curated.* : one-row config metadata (populated by the loader, not the
-- simulator). The drift boundary is config-driven; storing it here keeps the
-- SQL views free of hard-coded dates while config stays the source of truth.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS curated.drift_boundary (
    boundary_ts TIMESTAMP  -- NULL when the profile has no drift (all period A)
);

-- ---------------------------------------------------------------------------
-- features.* : point-in-time feature snapshots.
-- ---------------------------------------------------------------------------

-- Trade-off: the wide feature table lives in parquet; here we store the row id
-- plus a JSONB blob so the DB schema stays stable while the feature set evolves
-- (Part 05+). JSONB is queryable by key and indexable (GIN) if needed. The
-- downside is weaker typing/constraints per feature; acceptable because Python
-- (features/registry.py) is the single source of truth for feature logic.
CREATE TABLE IF NOT EXISTS features.feature_snapshot (
    transaction_id            TEXT PRIMARY KEY REFERENCES raw.transactions(transaction_id),
    feature_as_of_timestamp   TIMESTAMP NOT NULL,
    feature_version           TEXT NOT NULL,
    features                  JSONB NOT NULL
);

-- ---------------------------------------------------------------------------
-- model.* : registry + scored predictions.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS model.model_registry (
    model_version    TEXT PRIMARY KEY,
    feature_version  TEXT,
    trained_at       TIMESTAMP,
    training_window  TEXT,
    calibration      TEXT,
    policy_version   TEXT,
    metrics          JSONB
);

CREATE TABLE IF NOT EXISTS model.predictions (
    prediction_id         BIGSERIAL PRIMARY KEY,
    transaction_id        TEXT REFERENCES raw.transactions(transaction_id),
    model_version         TEXT,
    raw_probability       DOUBLE PRECISION,
    calibrated_probability DOUBLE PRECISION,
    risk_score            DOUBLE PRECISION,
    action                TEXT,
    scored_at             TIMESTAMP,
    latency_ms            DOUBLE PRECISION
);

-- ---------------------------------------------------------------------------
-- monitoring.* : metric time series for drift / performance monitoring.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS monitoring.metrics (
    metric_date   DATE NOT NULL,
    model_version TEXT NOT NULL,
    metric_name   TEXT NOT NULL,
    metric_value  DOUBLE PRECISION,
    extra         JSONB,
    PRIMARY KEY (metric_date, model_version, metric_name)
);
