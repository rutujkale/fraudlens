"""Point-in-time behavioural feature engineering (Part 05A).

``build_features`` computes every registry feature for every row of ``tx`` using
only rows strictly earlier in time (``timestamp < T``). It is the **only**
implementation of feature logic; training, the API and Streamlit all reach it
(``serving.compute_features_for_transaction`` appends the live transaction to its
history and calls this same function).

Performance: per customer we keep a sorted timestamp array plus cumulative
count/sum/sum-of-squares arrays; each row's window is located with
``windows.window_bounds`` (the single ``searchsorted`` call site). Cross-customer
device state is processed in global ``(timestamp, transaction_id)`` order with
tie-group snapshots, so same-timestamp rows never see each other.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from fraudlens.features import windows
from fraudlens.features.registry import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    assert_no_target_columns,
)

_EARTH_RADIUS_KM = 6371.0088
_DAY = np.timedelta64(1, "D")
_HOUR = np.timedelta64(1, "h")
_MINUTE = np.timedelta64(1, "m")


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _distinct(values: np.ndarray, start: int, end: int) -> int:
    """Distinct non-null values in ``values[start:end]``."""
    if end <= start:
        return 0
    chunk = values[start:end]
    chunk = chunk[~pd.isna(chunk)]
    return int(np.unique(chunk).size)


def _ratio(num: float, den: float) -> float:
    """``num / den`` or NaN when ``den`` is missing or zero."""
    if den is None or not np.isfinite(den) or den == 0:
        return float("nan")
    return float(num / den)


def _value_map(frame: pd.DataFrame, column: str, key: str = "location_id") -> dict:
    """``key -> column`` lookup, or an empty map when ``column`` is absent.

    Lets an adapter supply a ``locations`` frame without coordinates (Part 11):
    missing columns degrade to all-NaN features instead of a KeyError.
    """
    if column not in frame.columns:
        return {}
    return dict(zip(frame[key], frame[column]))


def _global_device_features(
    ts: np.ndarray, dev: np.ndarray, cust: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-customer device stats: distinct prior users and device age (days).

    Rows are already globally ``(timestamp, transaction_id)``-sorted. For each
    device we walk its rows in order and snapshot state *before* applying a
    timestamp group, so same-timestamp rows never count each other.
    """
    n = len(ts)
    accounts = np.full(n, np.nan)
    age_days = np.full(n, np.nan)
    mask = pd.notna(dev)
    if not mask.any():
        return accounts, age_days

    positions = np.nonzero(mask)[0]
    sub = pd.DataFrame(
        {"device_id": dev[mask], "ts": ts[mask], "cust": cust[mask]}, index=positions
    )

    for _device, group in sub.groupby("device_id", sort=False):
        gpos = group.index.to_numpy()
        gts = group["ts"].to_numpy()
        gcust = group["cust"].to_numpy()
        seen: set = set()
        first: np.datetime64 | None = None
        m = len(gpos)
        k = 0
        while k < m:
            j = k
            while j < m and gts[j] == gts[k]:
                j += 1
            if first is None:
                prior_accounts = 0
                prior_age = float("nan")
            else:
                prior_accounts = len(seen)
                prior_age = float((gts[k] - first) / _DAY)
            for r in range(k, j):
                accounts[gpos[r]] = prior_accounts
                age_days[gpos[r]] = prior_age
            for r in range(k, j):
                seen.add(gcust[r])
            if first is None:
                first = gts[k]
            k = j
    return accounts, age_days


def build_features(tx: pd.DataFrame, entities, cfg) -> pd.DataFrame:
    """Compute every feature for every row of ``tx`` using only prior rows.

    ``entities`` only needs ``.customers`` (home location, account_open_date) and
    ``.locations`` (coordinates, region). Rows are sorted by
    ``(timestamp, transaction_id)`` and returned in that order.
    """
    sim = cfg.simulation
    fc = sim.features
    threshold = sim.reporting_threshold
    band_lo = fc.near_threshold_band[0] * threshold
    band_hi = fc.near_threshold_band[1] * threshold

    tx = tx.sort_values(["timestamp", "transaction_id"], kind="stable").reset_index(drop=True)
    assert_no_target_columns(tx)
    n = len(tx)

    ts = pd.to_datetime(tx["timestamp"]).to_numpy()
    amount = tx["amount"].to_numpy(dtype="float64")
    cust = tx["customer_id"].to_numpy()
    dev = tx["device_id"].to_numpy()
    recip = tx["recipient_id"].to_numpy()
    merch = tx["merchant_id"].to_numpy()
    cat = tx["merchant_category"].to_numpy()

    locations = entities.locations
    lat_by_loc = _value_map(locations, "latitude")
    lon_by_loc = _value_map(locations, "longitude")
    region_by_loc = _value_map(locations, "region_group")
    lat = tx["location_id"].map(lat_by_loc).to_numpy(dtype="float64")
    lon = tx["location_id"].map(lon_by_loc).to_numpy(dtype="float64")
    region = tx["location_id"].map(region_by_loc).to_numpy()

    customers = entities.customers
    home_loc = dict(zip(customers["customer_id"], customers["home_location_id"]))
    home_lat = tx["customer_id"].map(
        {c: lat_by_loc.get(h) for c, h in home_loc.items()}
    ).to_numpy("float64")
    home_lon = tx["customer_id"].map(
        {c: lon_by_loc.get(h) for c, h in home_loc.items()}
    ).to_numpy("float64")
    acct_open = dict(zip(customers["customer_id"], pd.to_datetime(customers["account_open_date"])))
    open_ns = pd.to_datetime(tx["customer_id"].map(acct_open)).to_numpy()

    accounts_seen, device_age = _global_device_features(ts, dev, cust)

    out: dict[str, np.ndarray] = {
        name: np.full(n, np.nan, dtype="float64") for name in NUMERIC_COLUMNS
    }
    for c in CATEGORICAL_COLUMNS:
        out[c] = tx[c].to_numpy()

    wdelta = {int(m): np.timedelta64(int(m), "m") for m in fc.windows_minutes}
    w10, w60, w360, w1440, w10080, w43200 = (
        wdelta[10],
        wdelta[60],
        wdelta[360],
        wdelta[1440],
        wdelta[10080],
        wdelta[43200],
    )
    w7 = np.timedelta64(int(fc.trend_window_days), "D")
    w14 = np.timedelta64(2 * int(fc.trend_window_days), "D")
    half = int(fc.hour_deviation_halfwidth_hours)
    min_prior = int(fc.hour_deviation_min_prior)
    std_floor = float(fc.amount_std_floor)
    speed_cap = float(fc.travel_speed_cap_kmh)
    hour_offsets = np.array([(h - half) % 24 for h in range(0, 2 * half + 1)])

    for idx in tx.groupby("customer_id", sort=False).indices.values():
        idx = np.asarray(idx)
        m = len(idx)
        cts = ts[idx]
        cam = amount[idx]
        cumsum = np.concatenate([[0.0], np.cumsum(cam)])
        cumsq = np.concatenate([[0.0], np.cumsum(cam * cam)])
        cumnear = np.concatenate(
            [[0.0], np.cumsum(((cam >= band_lo) & (cam < band_hi)).astype("float64"))]
        )
        cday = cts.astype("datetime64[D]")
        chour = (cts.astype("datetime64[h]").astype("int64") % 24).astype("int64")
        cdist = cts.astype("datetime64[D]").astype("int64")
        cdev, crecip, cmerch, ccat, cregion = (
            dev[idx],
            recip[idx],
            merch[idx],
            cat[idx],
            region[idx],
        )

        seen_dev: set = set()
        seen_recip: set = set()
        seen_merch: set = set()
        seen_cat: set = set()
        seen_region: set = set()
        dev_count: dict = {}
        hour_hist = np.zeros(24, dtype="int64")

        # Rows are processed in equal-timestamp groups. ``prior_end`` is the
        # exclusive end of the strictly-prior rows (== #rows with ts < t), so a
        # same-timestamp peer is never visible to the row next to it.
        prior_end = 0
        r = 0
        while r < m:
            g_end = r + 1
            while g_end < m and cts[g_end] == cts[r]:
                g_end += 1
            t = cts[r]
            s10, e10 = windows.window_bounds(cts, t, w10)
            s60, e60 = windows.window_bounds(cts, t, w60)
            s360, e360 = windows.window_bounds(cts, t, w360)
            s1440, e1440 = windows.window_bounds(cts, t, w1440)
            s7d, e7d = windows.window_bounds(cts, t, w10080)
            s30d, e30d = windows.window_bounds(cts, t, w43200)
            s7, _ = windows.window_bounds(cts, t, w7)
            s14, _ = windows.window_bounds(cts, t, w14)

            c60 = e60 - s60
            c1440 = e1440 - s1440
            c7d = e7d - s7d
            c30d = e30d - s30d
            c_prev7 = s7 - s14
            prev = prior_end - 1

            for i in range(r, g_end):
                a = cam[i]
                out["amount"][idx[i]] = a
                out["log_amount"][idx[i]] = math.log1p(a)
                out["hour"][idx[i]] = chour[i]
                dow = int((cdist[i] + 3) % 7)
                out["day_of_week"][idx[i]] = dow
                out["is_weekend"][idx[i]] = 1 if dow >= 5 else 0
                out["is_night"][idx[i]] = 1 if (chour[i] >= 22 or chour[i] < 6) else 0

                # recency
                if prior_end > 0:
                    out["minutes_since_prev_tx"][idx[i]] = float((t - cts[prev]) / _MINUTE)
                    out["days_since_last_tx"][idx[i]] = float((t - cts[prev]) / _DAY)
                out["n_prior_tx"][idx[i]] = prior_end
                out["history_days"][idx[i]] = float((t - open_ns[idx[i]]) / _DAY)
                out["is_cold_start"][idx[i]] = 1 if prior_end == 0 else 0

                # frequency
                out["tx_count_10m"][idx[i]] = e10 - s10
                out["tx_count_1h"][idx[i]] = c60
                out["tx_count_6h"][idx[i]] = e360 - s360
                out["tx_count_24h"][idx[i]] = c1440
                out["tx_count_7d"][idx[i]] = c7d
                out["active_days_7d"][idx[i]] = _distinct(cday, s7d, e7d)
                out["active_days_30d"][idx[i]] = _distinct(cday, s30d, e30d)
                out["velocity_ratio_1h_vs_24h"][idx[i]] = _ratio(c60, c1440)

                # amount sums
                out["amount_sum_1h"][idx[i]] = cumsum[e60] - cumsum[s60]
                out["amount_sum_24h"][idx[i]] = cumsum[e1440] - cumsum[s1440]
                out["max_amount_24h"][idx[i]] = cam[s1440:e1440].max() if c1440 else float("nan")
                out["count_near_threshold_24h"][idx[i]] = int(cumnear[e1440] - cumnear[s1440])

                # amount deviation
                sum7d = cumsum[e7d] - cumsum[s7d]
                sum30d = cumsum[e30d] - cumsum[s30d]
                sumsq30d = cumsq[e30d] - cumsq[s30d]
                avg7 = sum7d / c7d if c7d else float("nan")
                avg30 = sum30d / c30d if c30d else float("nan")
                std30 = (
                    math.sqrt(max(sumsq30d / c30d - avg30 * avg30, 0.0)) if c30d else float("nan")
                )
                out["avg_amount_7d"][idx[i]] = avg7
                out["avg_amount_30d"][idx[i]] = avg30
                out["std_amount_30d"][idx[i]] = std30
                out["amount_ratio_to_7d_mean"][idx[i]] = _ratio(a, avg7)
                out["amount_ratio_to_30d_mean"][idx[i]] = _ratio(a, avg30)
                out["amount_zscore_30d"][idx[i]] = (
                    (a - avg30) / max(std30, std_floor) if c30d else float("nan")
                )
                out["amount_vs_lifetime_median"][idx[i]] = (
                    _ratio(a, float(np.median(cam[:prior_end]))) if prior_end else float("nan")
                )

                # merchant / category
                if not pd.isna(cmerch[i]):
                    out["new_merchant_flag"][idx[i]] = 0 if cmerch[i] in seen_merch else 1
                    out["merchant_freq_30d"][idx[i]] = _ratio(
                        np.count_nonzero(cmerch[s30d:e30d] == cmerch[i]), c30d
                    )
                if not pd.isna(ccat[i]):
                    out["new_category_flag"][idx[i]] = 0 if ccat[i] in seen_cat else 1
                    out["category_freq_30d"][idx[i]] = _ratio(
                        np.count_nonzero(ccat[s30d:e30d] == ccat[i]), c30d
                    )

                # device
                if not pd.isna(cdev[i]):
                    out["is_new_device"][idx[i]] = 0 if cdev[i] in seen_dev else 1
                    out["tx_from_device_prior"][idx[i]] = dev_count.get(cdev[i], 0)
                    out["device_count_30d"][idx[i]] = np.count_nonzero(cdev[s30d:e30d] == cdev[i])
                    out["accounts_seen_on_device"][idx[i]] = accounts_seen[idx[i]]
                    out["device_age_days"][idx[i]] = device_age[idx[i]]

                # recipient
                out["unique_recipients_1h"][idx[i]] = _distinct(crecip, s60, e60)
                out["unique_recipients_24h"][idx[i]] = _distinct(crecip, s1440, e1440)
                if not pd.isna(crecip[i]):
                    out["is_new_recipient"][idx[i]] = 0 if crecip[i] in seen_recip else 1

                # geographic
                out["distance_from_home_km"][idx[i]] = _haversine(
                    lat[idx[i]], lon[idx[i]], home_lat[idx[i]], home_lon[idx[i]]
                )
                if prior_end > 0:
                    d_prev = _haversine(lat[idx[i]], lon[idx[i]], lat[idx[prev]], lon[idx[prev]])
                    dt_h = (t - cts[prev]) / _HOUR
                    out["distance_from_prev_km"][idx[i]] = d_prev
                    if dt_h > 0:
                        out["travel_speed_kmh"][idx[i]] = min(d_prev / float(dt_h), speed_cap)
                out["new_region_flag"][idx[i]] = 0 if cregion[i] in seen_region else 1
                out["region_freq_30d"][idx[i]] = _ratio(
                    np.count_nonzero(cregion[s30d:e30d] == cregion[i]), c30d
                )

                # temporal
                if prior_end >= min_prior:
                    bins = (chour[i] + hour_offsets) % 24
                    out["hour_deviation"][idx[i]] = float(hour_hist[bins].sum() / prior_end)

                # trend
                prev7_sum = cumsum[s7] - cumsum[s14]
                out["activity_change_7d_vs_prev_7d"][idx[i]] = _ratio(c7d, c_prev7)
                out["amount_change_7d_vs_prev_7d"][idx[i]] = _ratio(sum7d, prev7_sum)

            # update running state once the whole timestamp group is done
            for i in range(r, g_end):
                if not pd.isna(cdev[i]):
                    seen_dev.add(cdev[i])
                    dev_count[cdev[i]] = dev_count.get(cdev[i], 0) + 1
                if not pd.isna(crecip[i]):
                    seen_recip.add(crecip[i])
                if not pd.isna(cmerch[i]):
                    seen_merch.add(cmerch[i])
                if not pd.isna(ccat[i]):
                    seen_cat.add(ccat[i])
                seen_region.add(cregion[i])
                hour_hist[chour[i]] += 1
            prior_end = g_end
            r = g_end

    features = pd.DataFrame(out)
    result = pd.DataFrame(
        {
            "transaction_id": tx["transaction_id"].to_numpy(),
            "feature_as_of_timestamp": tx["timestamp"].to_numpy(),
        }
    )
    result = pd.concat([result, features[FEATURE_COLUMNS]], axis=1)
    negative_history = int((features["history_days"] < 0).sum())
    if negative_history:  # pragma: no cover - guard
        raise AssertionError(
            f"history_days < 0 for {negative_history} rows: transactions precede "
            "account_open_date (see Part 02b)"
        )
    assert_no_target_columns(result)
    return result
