# coding=utf-8

from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional
from urllib.parse import quote


YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=5d&interval=1d"
YAHOO_HISTORY_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=6mo&interval=1d"
YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

EASTMONEY_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://quote.eastmoney.com/",
}
EASTMONEY_QUOTE_URL = (
    "https://push2.eastmoney.com/api/qt/stock/get?"
    "fltt=2&invt=2&secid={secid}&fields=f43,f58,f60"
)
EASTMONEY_HISTORY_URL = (
    "https://push2his.eastmoney.com/api/qt/stock/kline/get?"
    "secid={secid}&fields1=f1,f2,f3,f4,f5,f6&"
    "fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&"
    "klt=101&fqt=1&beg={beg}&end={end}&lmt=180&ut=fa5fd1943c7b386f172d6893dbfba10b"
)


def describe_symbol(symbol: str) -> Dict[str, str]:
    normalized = str(symbol or "").strip().upper()
    market = ""
    pure_code = normalized
    secid = ""
    currency = ""

    if "." in normalized:
        pure, suffix = normalized.rsplit(".", 1)
        if suffix in {"SS", "SH"} and pure.isdigit():
            pure_code = pure.zfill(6)
            market = "sh"
            secid = f"1.{pure_code}"
            currency = "CNY"
        elif suffix == "SZ" and pure.isdigit():
            pure_code = pure.zfill(6)
            market = "sz"
            secid = f"0.{pure_code}"
            currency = "CNY"
        elif suffix == "HK" and pure.isdigit():
            pure_code = pure.zfill(5)
            market = "hk"
            secid = f"116.{pure_code}"
            currency = "HKD"
        else:
            pure_code = pure
            market = suffix.lower()
    elif normalized.isalpha():
        market = "us"
        currency = "USD"

    return {
        "symbol": normalized,
        "market": market,
        "pure_code": pure_code,
        "secid": secid,
        "currency": currency,
    }


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _eastmoney_price(value: Any) -> Optional[float]:
    parsed = _safe_float(value)
    if parsed in (None, 0):
        return None
    # Eastmoney quote fields store prices as integer values scaled by 100.
    return parsed / 100.0


def _fetch_yahoo_quote(
    symbol: str,
    timeout: float,
    request_get: Callable[..., Any],
) -> Optional[Dict[str, Any]]:
    url = YAHOO_CHART_URL.format(symbol=quote(symbol, safe=""))
    response = request_get(url, headers=YAHOO_HEADERS, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    result = ((payload.get("chart") or {}).get("result") or [None])[0]
    if not isinstance(result, dict):
        return None

    meta = result.get("meta") or {}
    indicators = result.get("indicators") or {}
    quote_data = ((indicators.get("quote") or [None])[0]) or {}
    closes = [
        value
        for value in (quote_data.get("close") or [])
        if isinstance(value, (int, float))
    ]

    current_price = meta.get("regularMarketPrice")
    previous_close = meta.get("chartPreviousClose")

    if current_price in (None, 0) and closes:
        current_price = closes[-1]
    if previous_close in (None, 0) and len(closes) >= 2:
        previous_close = closes[-2]

    if current_price in (None, 0) or previous_close in (None, 0):
        return None

    pct_change = ((float(current_price) - float(previous_close)) / float(previous_close)) * 100.0
    return {
        "symbol": symbol,
        "label": meta.get("shortName") or meta.get("longName") or symbol,
        "currency": meta.get("currency", ""),
        "current_price": float(current_price),
        "previous_close": float(previous_close),
        "pct_change": pct_change,
    }


def _fetch_eastmoney_quote(
    symbol_meta: Dict[str, str],
    timeout: float,
    request_get: Callable[..., Any],
) -> Optional[Dict[str, Any]]:
    secid = symbol_meta.get("secid")
    if not secid:
        return None

    url = EASTMONEY_QUOTE_URL.format(secid=secid)
    response = request_get(url, headers=EASTMONEY_HEADERS, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") or {}

    current_price = _eastmoney_price(data.get("f43"))
    previous_close = _eastmoney_price(data.get("f60"))
    if current_price in (None, 0) or previous_close in (None, 0):
        return None

    pct_change = ((current_price - previous_close) / previous_close) * 100.0
    return {
        "symbol": symbol_meta["symbol"],
        "label": data.get("f58") or symbol_meta["symbol"],
        "currency": symbol_meta.get("currency", ""),
        "current_price": current_price,
        "previous_close": previous_close,
        "pct_change": pct_change,
    }


def fetch_symbol_quote(
    symbol: str,
    timeout: float,
    request_get: Callable[..., Any],
) -> Optional[Dict[str, Any]]:
    symbol_meta = describe_symbol(symbol)
    if symbol_meta.get("secid"):
        try:
            quote_data = _fetch_eastmoney_quote(symbol_meta, timeout, request_get)
            if quote_data:
                return quote_data
        except Exception:
            pass

    try:
        return _fetch_yahoo_quote(symbol_meta["symbol"], timeout, request_get)
    except Exception:
        return None


def _fetch_yahoo_history(
    symbol: str,
    timeout: float,
    request_get: Callable[..., Any],
) -> Optional[Dict[str, Any]]:
    url = YAHOO_HISTORY_URL.format(symbol=quote(symbol, safe=""))
    response = request_get(url, headers=YAHOO_HEADERS, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    result = ((payload.get("chart") or {}).get("result") or [None])[0]
    if not isinstance(result, dict):
        return None

    meta = result.get("meta") or {}
    indicators = result.get("indicators") or {}
    quote_data = ((indicators.get("quote") or [None])[0]) or {}
    closes = quote_data.get("close") or []
    highs = quote_data.get("high") or []
    lows = quote_data.get("low") or []

    series = []
    max_len = max(len(closes), len(highs), len(lows))
    for index in range(max_len):
        close = closes[index] if index < len(closes) else None
        high = highs[index] if index < len(highs) else close
        low = lows[index] if index < len(lows) else close
        if not isinstance(close, (int, float)):
            continue
        if not isinstance(high, (int, float)):
            high = close
        if not isinstance(low, (int, float)):
            low = close
        series.append(
            {
                "close": float(close),
                "high": float(high),
                "low": float(low),
            }
        )

    if len(series) < 60:
        return None

    return {
        "symbol": symbol,
        "label": meta.get("shortName") or meta.get("longName") or symbol,
        "series": series,
    }


def _fetch_eastmoney_history(
    symbol_meta: Dict[str, str],
    timeout: float,
    request_get: Callable[..., Any],
) -> Optional[Dict[str, Any]]:
    secid = symbol_meta.get("secid")
    if not secid:
        return None

    end = datetime.now().strftime("%Y%m%d")
    begin = (datetime.now() - timedelta(days=210)).strftime("%Y%m%d")
    url = EASTMONEY_HISTORY_URL.format(secid=secid, beg=begin, end=end)
    response = request_get(url, headers=EASTMONEY_HEADERS, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") or {}
    klines = data.get("klines") or []

    series = []
    for line in klines:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        close = _safe_float(parts[2])
        high = _safe_float(parts[3])
        low = _safe_float(parts[4])
        if close is None:
            continue
        if high is None:
            high = close
        if low is None:
            low = close
        series.append(
            {
                "close": close,
                "high": high,
                "low": low,
            }
        )

    if len(series) < 60:
        return None

    return {
        "symbol": symbol_meta["symbol"],
        "label": data.get("name") or symbol_meta["symbol"],
        "series": series,
    }


def fetch_symbol_history(
    symbol: str,
    timeout: float,
    request_get: Callable[..., Any],
) -> Optional[Dict[str, Any]]:
    symbol_meta = describe_symbol(symbol)
    if symbol_meta.get("secid"):
        try:
            history = _fetch_eastmoney_history(symbol_meta, timeout, request_get)
            if history:
                return history
        except Exception:
            pass

    try:
        return _fetch_yahoo_history(symbol_meta["symbol"], timeout, request_get)
    except Exception:
        return None