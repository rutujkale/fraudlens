"""HTTP client for the FraudLens scoring API (Part 14).

The app holds **no** feature or scoring logic: every score comes from the Part 13
API. Tests inject a fake client with :func:`set_client`, so the pages never hit a
live server in CI.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Protocol

import httpx


def _resolve_api_url() -> str:
    """The API base URL: ``FRAUDLENS_API_URL`` env var, then Streamlit Cloud secrets.

    Community Cloud exposes secrets through ``st.secrets`` (``.streamlit/secrets.toml``),
    not the process environment, and reads this at import time — so the secret is copied
    into ``os.environ`` here. The local default is unchanged.
    """
    url = os.environ.get("FRAUDLENS_API_URL")
    if url:
        return url
    if "streamlit" in sys.modules:
        try:
            import streamlit as st

            url = st.secrets.get("FRAUDLENS_API_URL")
        except Exception:
            url = None
    return url or "http://127.0.0.1:8000"


DEFAULT_URL = _resolve_api_url()
API_COMMAND = "python scripts/run_api.py --profile dev"


class ApiError(RuntimeError):
    """The scoring API is unreachable or returned an error."""


class ApiClient(Protocol):
    """The slice of the API the pages use."""

    def health(self) -> dict[str, Any]: ...
    def model_info(self) -> dict[str, Any]: ...
    def predict(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    def predict_batch(self, transactions: list[dict[str, Any]]) -> dict[str, Any]: ...
    def batch_csv(self, data: bytes, filename: str = "batch.csv") -> dict[str, Any]: ...
    def customer_summary(self, customer_id: str, before: str | None = None) -> dict[str, Any]: ...


class HttpApiClient:
    """Thin ``httpx`` wrapper; every method raises :class:`ApiError` on failure."""

    def __init__(self, base_url: str | None = None, timeout: float = 60.0) -> None:
        self.base_url = (base_url or DEFAULT_URL).rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = httpx.request(
                method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs
            )
        except httpx.HTTPError as exc:
            raise ApiError(f"API request failed: {exc}") from exc
        if response.status_code >= 400:
            raise ApiError(f"API returned {response.status_code}: {response.text[:300]}")
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def model_info(self) -> dict[str, Any]:
        return self._request("GET", "/model-info")

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/predict", json=payload)

    def predict_batch(self, transactions: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request("POST", "/predict/batch", json={"transactions": transactions})

    def batch_csv(self, data: bytes, filename: str = "batch.csv") -> dict[str, Any]:
        return self._request(
            "POST", "/predict/batch/csv", files={"file": (filename, data, "text/csv")}
        )

    def customer_summary(self, customer_id: str, before: str | None = None) -> dict[str, Any]:
        params = {"before": before} if before else None
        return self._request("GET", f"/customer/{customer_id}/summary", params=params)


_CLIENT: ApiClient | None = None


def set_client(client: ApiClient | None) -> None:
    """Override the module client (tests, offline demos)."""
    global _CLIENT
    _CLIENT = client


def get_client() -> ApiClient:
    """The active client, or a default HTTP client against ``FRAUDLENS_API_URL``."""
    return _CLIENT if _CLIENT is not None else HttpApiClient()
