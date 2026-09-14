from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import threading
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, parse, request


@dataclass
class HTTPResponse:
    status_code: int
    headers: Dict[str, str]
    content: bytes
    url: str

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)


class HTTPRequestError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Transparent on-disk response cache (deterministic environment replay).
#
# Goal: collapse run-to-run noise from non-deterministic external services
# (search, SEC, market data, readers) so that score changes can be attributed
# to the model/training rather than to environment jitter. This layer is pure
# infrastructure: it never reasons, it only replays identical requests.
#
# Modes (env FIRE_AGENT_HTTP_CACHE):
#   off (default) : no caching, original live behavior.
#   rw            : read+write. Hit -> serve cached. Miss -> fetch live, then
#                   store ONLY successful 2xx responses. A different request
#                   (e.g. a reformulated query) is always a miss and goes live,
#                   so the model's freedom to search better is fully preserved.
#   ro            : read-only. Hit -> serve cached. Miss -> fetch live but DO
#                   NOT store. Never blocks a run; used to freeze a curated
#                   cache without polluting it.
#
# The cache key is sha256(version, method, final_url, body) and deliberately
# excludes auth headers (API keys) so the same logical query always matches.
# Only 2xx responses are cached: transient errors/timeouts are never frozen,
# so they can still recover on a rerun.
# ---------------------------------------------------------------------------

_CACHE_LOCK = threading.Lock()


def _http_cache_mode() -> str:
    return (os.getenv("FIRE_AGENT_HTTP_CACHE", "off") or "off").strip().lower()


def _http_cache_dir() -> Path:
    raw = (os.getenv("FIRE_AGENT_HTTP_CACHE_DIR", "") or "").strip()
    if raw:
        return Path(raw)
    return Path.cwd() / ".fire_http_cache"


def _http_cache_key(method: str, url: str, body: Optional[bytes]) -> str:
    version = (os.getenv("FIRE_AGENT_HTTP_CACHE_VERSION", "v1") or "v1").strip()
    digest = hashlib.sha256()
    digest.update(version.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(method.upper().encode("utf-8"))
    digest.update(b"\x00")
    digest.update(url.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(body or b"")
    return digest.hexdigest()


def _http_cache_load(key: str) -> Optional[HTTPResponse]:
    path = _http_cache_dir() / f"{key}.json"
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        content = base64.b64decode(payload["content_b64"])
    except (OSError, ValueError, KeyError):
        return None
    return HTTPResponse(
        status_code=int(payload.get("status_code", 200)),
        headers=dict(payload.get("headers") or {}),
        content=content,
        url=str(payload.get("url") or ""),
    )


def _http_cache_store(key: str, response: HTTPResponse) -> None:
    cache_dir = _http_cache_dir()
    payload = {
        "status_code": response.status_code,
        "headers": response.headers,
        "content_b64": base64.b64encode(response.content).decode("ascii"),
        "url": response.url,
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    path = cache_dir / f"{key}.json"
    tmp = cache_dir / f"{key}.json.tmp.{os.getpid()}.{threading.get_ident()}"
    with _CACHE_LOCK:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with tmp.open("w", encoding="utf-8") as handle:
                handle.write(serialized)
            tmp.replace(path)
        except OSError:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass


def http_get(url: str, *, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> HTTPResponse:
    return http_request("GET", url, headers=headers, params=params, timeout=timeout)


def http_post(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    json_body: Any = None,
    data: Any = None,
    timeout: int = 30,
) -> HTTPResponse:
    return http_request("POST", url, headers=headers, json_body=json_body, data=data, timeout=timeout)


def http_request(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json_body: Any = None,
    data: Any = None,
    timeout: int = 30,
) -> HTTPResponse:
    request_headers = dict(headers or {})
    if params:
        query = parse.urlencode({key: value for key, value in params.items() if value is not None})
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}{query}"

    body: Optional[bytes] = None
    if json_body is not None:
        body = json.dumps(json_body).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    elif data is not None:
        body = data.encode("utf-8") if isinstance(data, str) else data

    cache_mode = _http_cache_mode()
    cache_key: Optional[str] = None
    if cache_mode in {"rw", "ro"}:
        cache_key = _http_cache_key(method, url, body)
        cached = _http_cache_load(cache_key)
        if cached is not None:
            return cached

    req = request.Request(url, data=body, headers=request_headers, method=method.upper())
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            response_headers = dict(response.headers.items())
            result = HTTPResponse(
                status_code=response.status,
                headers=response_headers,
                content=_decode_content(raw, response_headers),
                url=response.geturl(),
            )
    except error.HTTPError as exc:
        raw = exc.read()
        message = _decode_content(raw, dict(exc.headers.items())).decode("utf-8", errors="replace")
        raise HTTPRequestError(f"HTTP {exc.code} {exc.reason}: {message[:1000]}") from exc
    except error.URLError as exc:
        raise HTTPRequestError(f"HTTP request failed for {url}: {exc.reason}") from exc

    # Only persist successful responses, and only in read+write mode, so that
    # transient failures are never frozen into the cache.
    if cache_mode == "rw" and cache_key is not None and 200 <= result.status_code < 300:
        _http_cache_store(cache_key, result)
    return result


def _decode_content(raw: bytes, headers: Dict[str, str]) -> bytes:
    encoding = (headers.get("Content-Encoding") or headers.get("content-encoding") or "").lower()
    if encoding == "gzip":
        return gzip.decompress(raw)
    if encoding == "deflate":
        return zlib.decompress(raw)
    return raw
